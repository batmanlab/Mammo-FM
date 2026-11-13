import os
from typing import Dict
import torch

from .efficientnet_custom import EfficientNet
from .image_classifier import LinearClassifier
from .image_encoder import HuggingfaceImageEncoder, ResNet, EfficientNet_Mammo
from .projection import LinearProjectionHead, MLPProjectionHead
from .text_encoder import HuggingfaceTextEncoder
from transformers import AutoModelForMaskedLM, AutoConfig, AutoModel

# Import ModernBERT modules to register them
try:
    import transformers
    from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
    from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM
    
    # Register the model type if not already registered
    if "modernbert" not in transformers.CONFIG_MAPPING:
        transformers.CONFIG_MAPPING.register("modernbert", ModernBertConfig)
        transformers.MODEL_FOR_MASKED_LM_MAPPING.register(ModernBertConfig, ModernBertForMaskedLM)
    print("Successfully imported and registered ModernBERT modules")
except ImportError:
    print("Warning: Could not import ModernBERT modules. Make sure you have the latest transformers version.")


def load_image_encoder(config_image_encoder: Dict):
    if config_image_encoder["source"].lower() == "huggingface":
        cache_dir = config_image_encoder[
            "cache_dir"] if "cache_dir" in config_image_encoder else "~/.cache/huggingface/hub"
        gradient_checkpointing = (
            config_image_encoder[
                "gradient_checkpointing"] if "gradient_checkpointing" in config_image_encoder else False
        )
        model_type = config_image_encoder["model_type"] if "model_type" in config_image_encoder else "vit"
        _image_encoder = HuggingfaceImageEncoder(
            name=config_image_encoder["name"],
            pretrained=config_image_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            model_type=model_type,
            local_files_only=os.path.exists(
                os.path.join(cache_dir, f'models--{config_image_encoder["name"].replace("/", "--")}')),
        )
    elif (
            config_image_encoder["source"].lower() == "cnn" and (
            config_image_encoder["name"].lower() == "tf_efficientnet_b5_ns" or
            config_image_encoder["name"].lower() == "tf_efficientnetv2_s"
    )):
        _image_encoder = EfficientNet_Mammo(name=config_image_encoder["name"])
    elif (
            config_image_encoder["source"].lower() == "cnn" and
            config_image_encoder["name"].lower() == "tf_efficientnetv2-detect"
    ):
        _image_encoder = EfficientNet.from_pretrained("efficientnet-b2", num_classes=1)
        _image_encoder.out_dim = 1408
    elif (
            config_image_encoder["source"].lower() == "cnn" and
            config_image_encoder["name"].lower() == "tf_efficientnet_b5_ns-detect"
    ):
        _image_encoder = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
        _image_encoder.out_dim = 2048
    elif (
            config_image_encoder["source"].lower() == "cnn" and (
            config_image_encoder["name"].lower() == "resnet152" or
            config_image_encoder["name"].lower() == "resnet101"
    )):
        _image_encoder = ResNet(name=config_image_encoder["name"])

    else:
        raise KeyError(f"Not supported image encoder: {config_image_encoder}")
    return _image_encoder


def load_text_encoder(config_text_encoder: Dict, vocab_size: int):
    if config_text_encoder["source"].lower() == "huggingface":
        cache_dir = config_text_encoder["cache_dir"]
        gradient_checkpointing = config_text_encoder["gradient_checkpointing"]
        _text_encoder = HuggingfaceTextEncoder(
            name=config_text_encoder["name"],
            vocab_size=vocab_size,
            pretrained=config_text_encoder["pretrained"],
            gradient_checkpointing=gradient_checkpointing,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(
                os.path.join(cache_dir, f'models--{config_text_encoder["name"].replace("/", "--")}')),
            trust_remote_code=config_text_encoder["trust_remote_code"],
        )
    elif config_text_encoder["source"].lower() == "ft":
        print("\n" + "="*50)
        print("Loading ModernBERT text encoder...")
        print(f"Loading from checkpoint: {config_text_encoder['cache_dir']}")
        
        try:
            _text_encoder = AutoModel.from_pretrained(
                config_text_encoder["cache_dir"],
                trust_remote_code=config_text_encoder.get("trust_remote_code", True),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            # Add out_dim attribute - ModernBERT uses hidden_size as embedding dimension
            _text_encoder.out_dim = _text_encoder.config.hidden_size
            print("Successfully loaded model from checkpoint!")
            print(f"Model config: {_text_encoder.config}")
            print(f"Output dimension: {_text_encoder.out_dim}")
            print(f"Model device: {next(_text_encoder.parameters()).device}")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"Error loading ModernBERT: {str(e)}")
            print(f"Error type: {type(e)}")
            raise
            
    else:
        raise KeyError(f"Not supported text encoder: {config_text_encoder}")
    return _text_encoder


def load_projection_head(embedding_dim: int, config_projection_head: Dict):
    if config_projection_head["name"].lower() == "mlp":
        projection_head = MLPProjectionHead(
            embedding_dim=embedding_dim, projection_dim=config_projection_head["proj_dim"],
            dropout=config_projection_head["dropout"]
        )
    elif config_projection_head["name"].lower() == "linear":
        projection_head = LinearProjectionHead(embedding_dim=embedding_dim,
                                               projection_dim=config_projection_head["proj_dim"])
    else:
        raise KeyError(f"Not supported text encoder: {config_projection_head}")
    return projection_head


def load_image_classifier(config_image_classifier: Dict, feature_dim: int):
    if config_image_classifier["name"].lower() == "linear":
        _image_classifier = LinearClassifier(feature_dim=feature_dim, num_class=config_image_classifier["n_class"])
    else:
        raise KeyError(f"Not supported image classifier: {config_image_classifier}")

    return _image_classifier
