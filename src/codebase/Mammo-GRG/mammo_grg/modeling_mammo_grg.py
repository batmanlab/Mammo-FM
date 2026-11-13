"""Llama report generator conditioned on four Mammo-FM feature maps.

Modified from LLaVA: https://github.com/haotian-liu/LLaVA
Copyright 2023 Haotian Liu, Apache-2.0.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, LlamaModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from .projector import build_vision_projector


class MammoGRGConfig(LlamaConfig):
    """Configuration compatible with the released ``llava_llama`` files."""

    model_type = "llava_llama"


class MammoGRGModel(LlamaModel):
    config_class = MammoGRGConfig

    def __init__(self, config: MammoGRGConfig):
        super().__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.mm_projector = build_vision_projector(config)

    def initialize_projector(self, projector_type: str, n_queries: int) -> None:
        self.config.mm_vision_tower = "Mammo-FM_BatmanlabTrained_CLIP_last.tar"
        self.config.use_mm_proj = True
        self.config.mm_projector_type = projector_type
        self.config.mm_hidden_size = 2048
        self.config.mm_context_size = 2048
        self.config.mm_n_queries = n_queries
        if not hasattr(self, "mm_projector"):
            self.mm_projector = build_vision_projector(self.config)


class MammoGRGForCausalLM(LlamaForCausalLM):
    config_class = MammoGRGConfig

    def __init__(self, config: MammoGRGConfig):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MammoGRGModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # The released PEFT adapter lists this historical auxiliary head among
        # its target modules even though the LM-only run did not use its loss.
        # Retaining it is required to load that adapter without key drift.
        self.lesion_head = nn.Linear(config.hidden_size, 4)
        self.post_init()

    def get_model(self) -> MammoGRGModel:
        return self.model

    def encode_images(self, images: torch.Tensor) -> list[torch.Tensor]:
        expected_rank = 6
        if images.ndim != expected_rank or images.shape[1:3] != (4, 1):
            raise ValueError(
                "Expected embeddings shaped [batch,4,1,2048,48,29], "
                f"received {tuple(images.shape)}"
            )
        image_features = images.squeeze(2).permute(0, 1, 3, 4, 2)
        batch, views, height, width, channels = image_features.shape
        if (views, channels, height, width) != (4, 2048, 48, 29):
            raise ValueError(f"Unexpected Mammo-FM feature geometry: {tuple(images.shape)}")

        projected_views = []
        for view_index in range(4):
            view = image_features[:, view_index].reshape(batch, height * width, channels)
            view = self.model.mm_projector(view)
            view = view + self.model.mm_projector.view_positional_embeddings[view_index][None, None]
            projected_views.append(view)
        return projected_views

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        if images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        image_features = self.encode_images(images)
        view_token_ids = getattr(self.config, "view_token_ids", [128260, 128261, 128262, 128263])
        token_to_view = {int(token_id): index for index, token_id in enumerate(view_token_ids)}

        original_labels = labels
        original_position_ids = position_ids
        original_attention_mask = attention_mask
        attention_mask = (
            torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        )
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        new_embeddings = []
        new_labels = []
        for batch_index, current_ids in enumerate(input_ids):
            image_positions = torch.where(current_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            if len(image_positions) != 4:
                raise ValueError(f"Every sample must contain exactly four <image> tokens; found {len(image_positions)}")

            boundaries = [-1, *image_positions, current_ids.shape[0]]
            id_segments = [current_ids[boundaries[i] + 1 : boundaries[i + 1]] for i in range(5)]
            label_segments = [labels[batch_index][boundaries[i] + 1 : boundaries[i + 1]] for i in range(5)]
            lengths = [segment.shape[0] for segment in id_segments]
            text_embeddings = torch.split(self.model.embed_tokens(torch.cat(id_segments)), lengths, dim=0)

            sample_embeddings = []
            sample_labels = []
            for index in range(5):
                sample_embeddings.append(text_embeddings[index])
                sample_labels.append(label_segments[index])
                if index == 4:
                    continue
                position = image_positions[index]
                preceding_token = int(current_ids[position - 1])
                if preceding_token not in token_to_view:
                    raise ValueError("Each <image> token must immediately follow one of the four view tokens")
                features = image_features[token_to_view[preceding_token]][batch_index]
                sample_embeddings.append(features)
                sample_labels.append(
                    torch.full((features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
                )
            new_embeddings.append(torch.cat(sample_embeddings))
            new_labels.append(torch.cat(sample_labels))

        max_model_length = getattr(self.config, "tokenizer_model_max_length", None)
        if max_model_length:
            new_embeddings = [item[:max_model_length] for item in new_embeddings]
            new_labels = [item[:max_model_length] for item in new_labels]

        max_length = max(item.shape[0] for item in new_embeddings)
        batch_size = len(new_embeddings)
        hidden_size = new_embeddings[0].shape[-1]
        padded_embeddings = torch.zeros(
            (batch_size, max_length, hidden_size),
            device=new_embeddings[0].device,
            dtype=new_embeddings[0].dtype,
        )
        padded_labels = torch.full(
            (batch_size, max_length), IGNORE_INDEX, device=labels.device, dtype=labels.dtype
        )
        new_attention = torch.zeros((batch_size, max_length), dtype=torch.bool, device=input_ids.device)
        new_positions = torch.zeros((batch_size, max_length), dtype=torch.long, device=input_ids.device)
        for index, (embeddings, target) in enumerate(zip(new_embeddings, new_labels)):
            length = embeddings.shape[0]
            padded_embeddings[index, :length] = embeddings
            padded_labels[index, :length] = target
            new_attention[index, :length] = True
            new_positions[index, :length] = torch.arange(length, device=input_ids.device)

        return (
            None,
            None if original_position_ids is None else new_positions,
            None if original_attention_mask is None else new_attention.to(original_attention_mask.dtype),
            past_key_values,
            padded_embeddings,
            None if original_labels is None else padded_labels,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes=None,
        return_dict: Optional[bool] = None,
        cache_position=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes
                )
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, inputs=None, images=None, image_sizes=None, **kwargs) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("Direct inputs_embeds are not supported")
        if images is not None:
            inputs, position_ids, attention_mask, _, inputs_embeds, _ = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs, position_ids, attention_mask, None, None, images, image_sizes
                )
            )
        else:
            inputs_embeds = self.model.embed_tokens(inputs)
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("llava_llama", MammoGRGConfig, exist_ok=True)
AutoModelForCausalLM.register(MammoGRGConfig, MammoGRGForCausalLM, exist_ok=True)
