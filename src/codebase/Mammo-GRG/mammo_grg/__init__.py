"""Core model code for Mammo-GRG.

The multimodal architecture is derived from LLaVA and the four-view adaptation
used to train Mammo-GRG.  See the per-file attribution notices.
"""

from .modeling_mammo_grg import MammoGRGConfig, MammoGRGForCausalLM

__all__ = ["MammoGRGConfig", "MammoGRGForCausalLM"]
