import torch
import dataclasses
from dataclasses import dataclass, field
from typing import List, Union, Optional, Callable, Any

@dataclasses.dataclass
class TrainConfig:
    """
    Configuration for training parameters.
    
    Attributes:
        lr (float): Learning rate for the optimizer.
        seed (int): Random seed for reproducibility.
        dtype (torch.dtype): Data type for model parameters.
        mean_center (bool): Whether to center input data to zero mean.
        target_norm (float): Target norm for weight normalization, None for no normalization.
        bias_init_median (bool): If True, initialize bias using median values.
        beta1 (float): Exponential decay rate for first moment estimates in Adam optimizer.
        beta2 (float): Exponential decay rate for second moment estimates in Adam optimizer.
        eps (float): Small constant for numerical stability in Adam optimizer.
        weight_decay (float): L2 regularization coefficient.
        scheduler (int): Learning rate scheduler type (1 for cosine decay).
        decay_time (float): Fraction of total epochs after which learning rate reaches its minimum.
        epochs (int): Total number of training epochs.
        batch_size (int): Number of samples per batch.
        clip_grad (float): Maximum gradient norm for gradient clipping.
        check_dead (int): Frequency (in iterations) to check for dead neurons.
        print_freq (int): Frequency (in iterations) to print training metrics.
        num_workers (int): Number of worker processes for data loading.
    """
    lr: float = 0.001
    seed: int = 12345
    dtype: Any = torch.float32
    mean_center: bool = True
    target_norm: Optional[float] = None
    bias_init_median: bool = False
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 6.25e-10
    weight_decay: float = 0.0
    scheduler: int = 1
    decay_time: float = 0.8
    epochs: int = 30
    batch_size: int = 4096
    clip_grad: float = 1.0
    check_dead: int = 1000
    print_freq: int = 100
    num_workers: int = 0

@dataclasses.dataclass
class LossConfig:
    """
    Configuration for loss function parameters.
    
    Attributes:
        reconstruction_loss (str): Type of reconstruction loss ('nmse' for Normalized Mean Squared Error).
        sparse_loss (str): Type of sparsity loss ('l1' for L1 norm).
        sparse_weight (float): Weight coefficient for the sparsity loss term.
    """
    reconstruction_loss: str = "mse"
    sparse_loss: str = "l1"
    sparse_weight: float = 0.0

@dataclasses.dataclass
class ModelConfig:
    """
    Configuration for model parameters.
    
    Attributes:
        use_matryoshka (bool): Whether to use matryoshka (nested) architecture.
        n_inputs (int): Number of input features.
        n_latents (int): Number of latent dimensions.
        activation (str): Activation function type (e.g., 'ReLU', 'TopKReLU_64').
        tied (bool): Whether to use tied weights for encoder and decoder.
        normalize (bool): Whether to normalize the latent representations.
        init_method (str): Weight initialization method.
        latent_soft_cap (float): Soft cap value for latent activations (0.0 for no cap).
        nesting_list (List[int] or int): List of nested dimensions or single value for matryoshka networks.
        relative_importance (List[int] or str): Relative importance weights for nested features 
                                              ('UW' for uniform, 'RW' for linear reverse).
        max_nesting (int): Maximum nesting level for matryoshka architecture. If `max_nesting` is lower
                            than the `nesting_list`, we will use n_latents as the max_nesting.
    """
    use_matryoshka: bool = False
    n_inputs: int = 1024
    n_latents: int = 1
    activation: str = "ReLU"
    tied: bool = False
    normalize: bool = False
    init_method: str = "kaiming"
    latent_soft_cap: float = 0.0
    nesting_list: Union[List[int], int] = 32
    relative_importance: Union[List[int], str] = "UW"
    max_nesting: int = 256#512

# Helper functions to create default instances
def default_relusae_training() -> TrainConfig:
    return TrainConfig(lr=5e-5)

def default_relusae_loss() -> LossConfig:
    return LossConfig(sparse_loss="l1", sparse_weight=0.003)

def default_relusae_model() -> ModelConfig:
    return ModelConfig(use_matryoshka=False, activation="ReLU")

@dataclasses.dataclass
class ReLUSAEConfig:
    """
    Configuration for ReLU-based Sparse Autoencoder.
    
    This configuration uses standard ReLU activation with L1 sparsity regularization.
    Optimized with a relatively small learning rate (5e-5) and L1 sparsity weight of 0.03.
    """
    training: TrainConfig = field(default_factory=default_relusae_training)
    loss: LossConfig = field(default_factory=default_relusae_loss)
    model: ModelConfig = field(default_factory=default_relusae_model)

def default_topksae_training() -> TrainConfig:
    return TrainConfig(lr=5e-4)

def default_topksae_loss() -> LossConfig:
    return LossConfig(sparse_weight=0.0)

def default_topksae_model() -> ModelConfig:
    return ModelConfig(use_matryoshka=False, activation="TopKReLU_64")

@dataclasses.dataclass
class TopKSAEConfig:
    """
    Configuration for Top-K Sparse Autoencoder.
    
    This configuration uses TopKReLU_64 activation which only keeps the top 64 activations
    and zeros out the rest, enforcing sparsity directly within the activation function
    rather than through a loss term. Uses a higher learning rate (5e-4) and no explicit
    sparsity weight.
    """
    training: TrainConfig = field(default_factory=default_topksae_training)
    loss: LossConfig = field(default_factory=default_topksae_loss)
    model: ModelConfig = field(default_factory=default_topksae_model)

def default_batchtopksae_training() -> TrainConfig:
    return TrainConfig(lr=5e-4)

def default_batchtopksae_loss() -> LossConfig:
    return LossConfig(sparse_weight=0.0)

def default_batchtopksae_model() -> ModelConfig:
    return ModelConfig(use_matryoshka=False, activation="BatchTopKReLU_32")

@dataclasses.dataclass
class BatchTopKSAEConfig:
    """
    Configuration for Batch Top-K Sparse Autoencoder.
    
    Similar to TopKSAE but uses BatchTopKReLU_32 activation which performs top-k
    selection across the batch dimension, potentially allowing more dynamic
    sparsity patterns. Uses a higher learning rate (5e-4) and no explicit
    sparsity weight.
    """
    training: TrainConfig = field(default_factory=default_batchtopksae_training)
    loss: LossConfig = field(default_factory=default_batchtopksae_loss)
    model: ModelConfig = field(default_factory=default_batchtopksae_model)

def default_msae_uw_training() -> TrainConfig:
    return TrainConfig(lr=1e-4, mean_center=False)

def default_msae_uw_loss() -> LossConfig:
    return LossConfig(reconstruction_loss="mse", sparse_weight=0.0)

def default_msae_uw_model() -> ModelConfig:
    return ModelConfig(
        use_matryoshka=True,
        activation="TopKReLU",
        nesting_list=64,
        relative_importance="UW",
    )

@dataclasses.dataclass
class MSAE_UWConfig:
    """
    Configuration for Matryoshka Sparse Autoencoder with Uniform Weighting.
    
    This configuration uses a TopKReLU activation with matryoshka nesting of 64
    and uniform weighting (UW) scheme for the nested features. No explicit sparsity
    regularization is used, as the TopKReLU enforces sparsity directly.
    """
    training: TrainConfig = field(default_factory=default_msae_uw_training)
    loss: LossConfig = field(default_factory=default_msae_uw_loss)
    model: ModelConfig = field(default_factory=default_msae_uw_model)

def default_msae_rw_training() -> TrainConfig:
    return TrainConfig(lr=1e-4, mean_center=False)

def default_msae_rw_loss() -> LossConfig:
    return LossConfig(reconstruction_loss="mse", sparse_weight=0.0)

def default_msae_rw_model() -> ModelConfig:
    return ModelConfig(
        use_matryoshka=True,
        activation="TopKReLU",
        nesting_list=64,
        relative_importance="RW",
    )

@dataclasses.dataclass
class MSAE_RWConfig:
    """
    Configuration for Matryoshka Sparse Autoencoder with Reverse Weighting.
    
    Similar to MSAE_UW but uses reciprocal weighting (RW) scheme for the nested features,
    which applies different importance to different levels of the nested representation.
    This can help prioritize certain features in the hierarchical representation.
    """
    training: TrainConfig = field(default_factory=default_msae_rw_training)
    loss: LossConfig = field(default_factory=default_msae_rw_loss)
    model: ModelConfig = field(default_factory=default_msae_rw_model)
    

def get_config(model_name: str):
    """
    Get the configuration object for the specified model name.
    
    Args:
        model_name (str): The name of the model.
    
    Returns:
        dataclass: The configuration object for the specified model.
        
    Raises:
        ValueError: If an unknown model name is provided.
    """
    if model_name == "ReLUSAE":
        return ReLUSAEConfig()
    elif model_name == "TopKSAE":
        return TopKSAEConfig()
    elif model_name == "BatchTopKSAE":
        return BatchTopKSAEConfig()
    elif model_name == "MSAE_UW":
        return MSAE_UWConfig()
    elif model_name == "MSAE_RW":
        return MSAE_RWConfig()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
