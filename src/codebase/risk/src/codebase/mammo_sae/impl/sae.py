from typing import Callable, Any, List, Optional, Tuple, Dict, Union
from functools import partial
import warnings

import numpy as np
import torch
import torch.nn as nn

from .utils import normalize_data, JumpReLUFunction, StepFunction

"""
Sparse Autoencoder (SAE) Implementation

This module implements various sparse autoencoder architectures and activation functions
designed to learn interpretable features in high-dimensional data.
"""

class SoftCapping(nn.Module):
    """
    Soft capping layer to prevent latent activations from growing excessively large.
    
    This layer applies a scaled tanh transformation that smoothly saturates values
    without hard truncation, helping stabilize training.
    
    Args:
        soft_cap (float): The scale factor for the tanh transformation
    """
    def __init__(self, soft_cap):
        super(SoftCapping, self).__init__()
        self.soft_cap = soft_cap

    def forward(self, logits):
        """
        Apply soft capping to input values.
        
        Args:
            logits (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Soft-capped values with range approximately [-soft_cap, soft_cap]
        """
        return self.soft_cap * torch.tanh(logits / self.soft_cap)


class TopK(nn.Module):
    """
    Top-K activation function that only keeps the K largest activations per sample.
    
    This activation enforces sparsity by zeroing out all but the k highest values in each
    input vector. Can optionally use absolute values for selection and apply a subsequent
    activation function.
    
    Args:
        k (int): Number of activations to keep
        act_fn (Callable, optional): Secondary activation function to apply to the kept values.
                                   Defaults to nn.ReLU().
        use_abs (bool, optional): If True, selection is based on absolute values. Defaults to False.
    """
    def __init__(self, k: int, act_fn: Callable = nn.ReLU(), use_abs: bool = False) -> None:
        super().__init__()
        self.k = k
        self.act_fn = act_fn
        self.use_abs = use_abs
        # print(f"Top_K used: {self.k}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that keeps only the top-k activations for each sample.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, features]
            
        Returns:
            torch.Tensor: Sparse output tensor with same shape as input, where all but
                        the top k values (per sample) are zero
        """
        if self.use_abs:
            x = torch.abs(x)
        
        # Get indices of top-k values along feature dimension
        _, indices = torch.topk(x, k=self.k, dim=-1)
        # Gather the corresponding values from the original input
        values = torch.gather(x, -1, indices)
            
        # Apply the activation function to the selected values
        activated_values = self.act_fn(values)
        # Create a tensor of zeros and place the activated values at the correct positions
        result = torch.zeros_like(x)
        result.scatter_(-1, indices, activated_values)
        
        # Verify sparsity constraint is met
        assert (result != 0.0).sum(dim=-1).max() <= self.k
        return result
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation mode forward pass that doesn't enforce sparsity.
        
        Used for computing full activations during evaluation or visualization.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after applying activation function (without top-k filtering)
        """
        if self.use_abs:
            x = torch.abs(x)
        
        x = self.act_fn(x)
        return x


class BatchTopK(TopK):
    """
    Batch-wide Top-K activation function that selects K largest activations across the entire batch.
    
    Unlike standard TopK which operates per sample, this selects the k*batch_size highest
    activations across all samples in the batch, potentially allowing some samples to have
    more activations than others based on relative magnitudes.
    
    Args:
        k (int): Target number of activations to keep per sample (actual number may vary)
        act_fn (Callable, optional): Secondary activation function. Defaults to nn.Identity().
        use_abs (bool, optional): If True, selection is based on absolute values. Defaults to False.
    """
    def __init__(self, k: int, act_fn: Callable = nn.Identity(), use_abs: bool = False) -> None:
        # Call the parent class constructor
        super().__init__(k, act_fn, use_abs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that keeps the top-k activations across the entire batch.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, features]
            
        Returns:
            torch.Tensor: Sparse output tensor with the same shape as input, where only
                        approximately k*batch_size values are non-zero across the entire batch
        """
        # Get batch size
        batch_size = x.shape[0]
        
        # Calculate total number of values to keep
        total_k = min(self.k * batch_size, x.numel())
        
        # Use absolute values if requested for selection
        values = torch.abs(x) if self.use_abs else x
        
        # Store original shape and flatten
        flat_values = values.flatten()
        flat_x = x.flatten()
        
        # Get indices of top-k elements across the entire batch
        _, indices = torch.topk(flat_values, k=total_k, dim=-1)
        
        # Create output tensor of zeros and place original values at correct positions
        flat_result = torch.zeros_like(flat_x)
        
        # Apply activation function to selected values and place them in the result
        activated_values = self.act_fn(flat_x[indices])
        flat_result.scatter_(-1, indices, activated_values)
        
        # Reshape back to original shape
        result = flat_result.reshape(values.shape)
        
        return result
    
    def forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluation mode forward pass that doesn't enforce sparsity.
        
        Used for computing full activations during evaluation or visualization.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output after applying activation function (without top-k filtering)
        """
        x = torch.abs(x) if self.use_abs else x
        x = self.act_fn(x)
        return x


class JumpReLU(nn.Module):
    """
    JumpReLU activation with learned thresholds.
    
    This activation implements a soft version of a threshold-based activation function,
    where values below a learned threshold are suppressed. The bandwidth parameter
    controls the sharpness of the transition at the threshold.
    
    Args:
        hidden_dim (int): Dimension of the input tensor
        init_threshold (float, optional): Initial threshold value. Defaults to 0.001.
        bandwidth (float, optional): Controls the transition sharpness. Defaults to 0.001.
    """
    def __init__(self, hidden_dim: int, init_threshold: float=0.001, bandwidth: float=0.001) -> None:
        """
        Initialize JumpReLU activation with specified parameters.

        Args:
            hidden_dim (int): Dimension of the input tensor
            init_threshold (float, optional): Initial threshold for the JUMP mechanism. Defaults to 0.001.
            bandwidth (float, optional): Controls transition sharpness. Defaults to 0.001.
        """
        super().__init__()
        self.log_threshold = nn.Parameter(torch.full((hidden_dim,), np.log(init_threshold)))
        self.bandwidth = bandwidth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Applies ReLU followed by the JUMP mechanism.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Activated tensor
        """
        x_relu = torch.relu(x)
        out = JumpReLUFunction.apply(x_relu, self.log_threshold, self.bandwidth)
        return out

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass used during training.
        
        Uses a step function approximation for computing gradients.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Activated tensor with gradient-friendly step function
        """
        return StepFunction.apply(x, self.log_threshold, self.bandwidth)


# Mapping of activation function names to their corresponding classes
ACTIVATIONS_CLASSES = {
    "ReLU": nn.ReLU,
    "JumpReLU": JumpReLU,
    "Identity": nn.Identity,
    "TopK": partial(TopK, act_fn=nn.Identity()),
    "TopKReLU": partial(TopK, act_fn=nn.ReLU()),
    "TopKabs": partial(TopK, use_abs=True, act_fn=nn.Identity()),
    "TopKabsReLU": partial(TopK, use_abs=True, act_fn=nn.ReLU()),
    "BatchTopK": partial(BatchTopK, act_fn=nn.Identity()),
    "BatchTopKReLU": partial(BatchTopK, act_fn=nn.ReLU()),
    "BatchTopKabs": partial(BatchTopK, use_abs=True, act_fn=nn.Identity()),
    "BatchTopKabsReLU": partial(BatchTopK, use_abs=True, act_fn=nn.ReLU()),
}


def get_activation(activation: str) -> nn.Module:
    """
    Factory function to create activation function instances by name.
    
    Handles special cases like parameterized activations (e.g., TopK_64).
    
    Args:
        activation (str): Name of the activation function, with optional parameter
                         (e.g., "TopKReLU_64" for TopKReLU with k=64)
                         
    Returns:
        nn.Module: Instantiated activation function
    """
    if "_" in activation:
        parts = activation.split("_")
        activation = parts[0]
        arg = parts[-1]  # Take the last part as the argument
        if "TopK" in activation:
            return ACTIVATIONS_CLASSES[activation](k=int(arg))
        elif "JumpReLU" in activation:
            return ACTIVATIONS_CLASSES[activation](hidden_dim=int(arg))
    return ACTIVATIONS_CLASSES[activation]()


class Autoencoder(nn.Module):
    """
    Sparse autoencoder base class.
    
    Implements the standard sparse autoencoder architecture:
        latents = activation(encoder(x - pre_bias) + latent_bias)
        recons = decoder(latents) + pre_bias
        
    Includes various options for controlling activation functions, weight initialization,
    and feature normalization.
    
    Attributes:
        n_latents (int): Number of latent features (neurons)
        n_inputs (int): Dimensionality of the input data
        tied (bool): Whether decoder weights are tied to encoder weights
        normalize (bool): Whether to normalize input data
        encoder (nn.Parameter): Encoder weight matrix [n_inputs, n_latents]
        decoder (nn.Parameter): Decoder weight matrix [n_latents, n_inputs] (if not tied)
        pre_bias (nn.Parameter): Input bias/offset [n_inputs]
        latent_bias (nn.Parameter): Latent bias [n_latents]
        activation (nn.Module): Activation function for the latent layer
        latents_activation_frequency (torch.Tensor): Tracks how often neurons activate
    """

    def __init__(
        self, n_latents: int, n_inputs: int, activation: Callable = nn.ReLU(), tied: bool = False, normalize: bool = False,
        bias_init: torch.Tensor or float = 0.0, init_method: str = "kaiming", latent_soft_cap: float = 30.0, threshold: torch.Tensor = None,
        *args, **kwargs
    ) -> None:
        """
        Initialize the sparse autoencoder.
        
        Args:
            n_latents (int): Dimension of the autoencoder latent space
            n_inputs (int): Dimensionality of the original data
            activation (Callable or str): Activation function or name
            tied (bool, optional): Whether to tie encoder and decoder weights. Defaults to False.
            normalize (bool, optional): Whether to normalize input data. Defaults to False.
            bias_init (torch.Tensor or float, optional): Initial bias value. Defaults to 0.0.
            init_method (str, optional): Weight initialization method. Defaults to "kaiming".
            latent_soft_cap (float, optional): Soft cap value for latent activations. Defaults to 30.0.
            threshold (torch.Tensor, optional): Threshold for JumpReLU. Defaults to None.
        """
        super().__init__()
        if isinstance(activation, str):
            activation = get_activation(activation)
        
        # Store configuration
        self.tied = tied
        self.n_latents = n_latents
        self.n_inputs = n_inputs
        self.init_method = init_method
        self.bias_init = bias_init
        self.normalize = normalize
        
        # Initialize parameters
        self.pre_bias = nn.Parameter(torch.full((n_inputs,), bias_init) if isinstance(bias_init, float) else bias_init)
        self.encoder = nn.Parameter(torch.zeros((n_inputs, n_latents)))
        self.latent_bias = nn.Parameter(torch.zeros(n_latents,))
        
        # For tied weights, decoder is derived from encoder
        if tied:
            self.register_parameter('decoder', None)
        else:
            self.decoder = nn.Parameter(torch.zeros((n_latents, n_inputs)))
        
        # Set up activation functions
        self.latent_soft_cap = SoftCapping(latent_soft_cap) if latent_soft_cap > 0 else nn.Identity()
        self.activation = activation
        
        # Set threshold for JumpReLU if needed
        if isinstance(self.activation, JumpReLU) and threshold is not None:
            self.activation.log_threshold = threshold
            
        self.dead_activations = activation
        
        # Initialize weights
        self._init_weights()

        # Set up activation tracking
        self.latents_activation_frequency: torch.Tensor
        self.register_buffer(
            "latents_activation_frequency", torch.zeros(n_latents, dtype=torch.int64, requires_grad=False)
        )
        self.num_updates = 0
        
        self.dead_latents = []

    def get_and_reset_stats(self) -> torch.Tensor:
        """
        Get activation statistics and reset the counters.
        
        Returns:
            torch.Tensor: Proportion of samples that activated each neuron
        """
        activations = self.latents_activation_frequency.detach().cpu().float() / self.num_updates
        self.latents_activation_frequency.zero_()
        self.num_updates = 0
        return activations
    
    @torch.no_grad()
    def _init_weights(self, norm=0.1, neuron_indices: List[int] = None) -> None:
        """
        Initialize network weights.
        
        Args:
            norm (float, optional): Target norm for the weights. Defaults to 0.1.
            neuron_indices (List[int], optional): Indices of neurons to initialize.
                                                       If None, initialize all neurons.
        
        Raises:
            ValueError: If invalid initialization method is specified
        """
        if self.init_method not in ["kaiming", "xavier", "uniform", "normal"]:
            raise ValueError(f"Invalid init_method: {self.init_method}")
        
        # Use transposed encoder if weights are tied
        if self.tied:
            decoder_weight = self.encoder.t()
        else:
            decoder_weight = self.decoder
        
        # Initialize with specified method
        if self.init_method == "kaiming":
            new_W_dec = (nn.init.kaiming_uniform_(torch.zeros_like(decoder_weight), nonlinearity='relu'))
        elif self.init_method == "xavier":
            new_W_dec = (nn.init.xavier_uniform_(torch.zeros_like(decoder_weight), gain=nn.init.calculate_gain('relu')))
        elif self.init_method == "uniform":
            new_W_dec = (nn.init.uniform_(torch.zeros_like(decoder_weight), a=-1, b=1))
        elif self.init_method == "normal":
            new_W_dec = (nn.init.normal_(torch.zeros_like(decoder_weight)))
        else:
            raise ValueError(f"Invalid init_method: {self.init_method}")

        # Normalize to target norm
        new_W_dec *= (norm / new_W_dec.norm(p=2, dim=-1, keepdim=True))
        
        # Initialize bias to zero
        new_l_bias = (torch.zeros_like(self.latent_bias))
        
        # Transpose for encoder
        new_W_enc = new_W_dec.t().clone()
        
        # Apply initialization to all or specific neurons
        if neuron_indices is None:
            if not self.tied:
                self.decoder.data = new_W_dec
            self.encoder.data = new_W_enc
            self.latent_bias.data = new_l_bias
        else:
            if not self.tied:
                self.decoder.data[neuron_indices] = new_W_dec[neuron_indices]
            self.encoder.data[:, neuron_indices] = new_W_enc[:, neuron_indices]
            self.latent_bias.data[neuron_indices] = new_l_bias[neuron_indices]
    
    @torch.no_grad()
    def project_grads_decode(self):
        """
        Project out components of decoder gradient that would change its norm.
        
        This helps maintain normalized decoder norms during training.
        """
        if self.tied:
            weights = self.encoder.data.T
            grad = self.encoder.grad.T
        else:
            weights = self.decoder.data
            grad = self.decoder.grad
        
        # Project out the component parallel to weights
        grad_proj = (grad * weights).sum(dim=-1, keepdim=True) * weights
        
        # Update gradients
        if self.tied:
            self.encoder.grad -= grad_proj.T
        else:
            self.decoder.grad -= grad_proj
        
    @torch.no_grad()
    def scale_to_unit_norm(self) -> None:
        """
        Scale decoder rows to unit norm, and adjust other parameters accordingly.
        
        This normalization helps with feature interpretability and training stability.
        """
        eps = torch.finfo(self.decoder.dtype).eps
        
        # Normalize tied or untied weights
        if self.tied:
            norm = self.encoder.data.T.norm(p=2, dim=-1, keepdim=True) + eps
            self.encoder.data.T /= norm
        else:
            norm = self.decoder.data.norm(p=2, dim=-1, keepdim=True) + eps
            self.decoder.data /= norm
            self.encoder.data *= norm.t()
        
        # Scale biases accordingly
        self.latent_bias.data *= norm.squeeze()
        
        # Adjust JumpReLU thresholds if present
        if isinstance(self.activation, JumpReLU):
            cur_threshold = torch.exp(self.activation.log_threshold.data)
            self.activation.log_threshold.data = torch.log(cur_threshold * norm.squeeze())
   
    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activation latent values.
        
        Args:
            x (torch.Tensor): Input data [batch, n_inputs]
        
        Returns:
            torch.Tensor: Pre-activation latent values [batch, n_latents]
        """
        x = x - self.pre_bias
        latents_pre_act_full = x @ self.encoder + self.latent_bias
        return latents_pre_act_full

    def preprocess(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Preprocess input data, optionally normalizing it.
        
        Args:
            x (torch.Tensor): Input data [batch, n_inputs]
            
        Returns:
            tuple: (preprocessed_data, normalization_info)
                - preprocessed_data: Processed input data
                - normalization_info: Dict with normalization parameters (if normalize=True)
        """
        if not self.normalize:
            return x, dict()
        x_processed, mu, std = normalize_data(x)
        return x_processed, dict(mu=mu, std=std)

    def encode(self, x: torch.Tensor, topk_number: int = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Encode input data to latent representations.
        
        Args:
            x (torch.Tensor): Input data [batch, n_inputs]
            topk_number (int, optional): Number of top-k activations to keep (for inference).
                                              Defaults to None.
        
        Returns:
            tuple: (encoded, full_encoded, info)
                - encoded: Latent activations with sparsity constraints [batch, n_latents]
                - full_encoded: Latent activations without sparsity (for analysis) [batch, n_latents]
                - info: Normalization information dictionary
        """
        x, info = self.preprocess(x)
        pre_encoded = self.encode_pre_act(x)
        encoded = self.activation(pre_encoded)
        
        # Get full activations (for analysis) depending on activation type
        if isinstance(self.activation, TopK):
            full_encoded = self.activation.forward_eval(pre_encoded)
        else:
            full_encoded = torch.clone(encoded)
        
        # Apply topk filtering for inference if requested
        if topk_number is not None:
            _, indices = torch.topk(full_encoded, k=topk_number, dim=-1)
            values = torch.gather(full_encoded, -1, indices)
            full_encoded = torch.zeros_like(full_encoded)
            full_encoded.scatter_(-1, indices, values)
        
        # Apply soft capping to both outputs
        caped_encoded = self.latent_soft_cap(encoded)
        capped_full_encoded = self.latent_soft_cap(full_encoded)
        
        return caped_encoded, capped_full_encoded, info

    def decode(self, latents: torch.Tensor, info: Dict[str, Any] = None) -> torch.Tensor:
        """
        Decode latent representations to reconstructed inputs.
        
        Args:
            latents (torch.Tensor): Latent activations [batch, n_latents]
            info (Dict[str, Any], optional): Normalization information. Defaults to None.
        
        Returns:
            torch.Tensor: Reconstructed input data [batch, n_inputs]
        """
        # Decode using tied or untied weights
        if self.tied:
            ret = latents @ self.encoder.t() + self.pre_bias
        else:
            ret = latents @ self.decoder + self.pre_bias
            
        # Denormalize if needed
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
            
        return ret

    @torch.no_grad()
    def update_latent_statistics(self, latents: torch.Tensor) -> None:
        """
        Update statistics on latent activations.
        
        Args:
            latents (torch.Tensor): Latent activations [batch, n_latents]
        """
        self.num_updates += latents.shape[0]
        current_activation_frequency = (latents != 0).to(torch.int64).sum(dim=0)
        self.latents_activation_frequency += current_activation_frequency

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x (torch.Tensor): Input data [batch, n_inputs]
        
        Returns:
            tuple: (recons, latents, all_recons, all_latents)
                - recons: Reconstructed data [batch, n_inputs]
                - latents: Latent activations [batch, n_latents]
                - all_recons: Reconstructed data without sparsity constraints (for analysis)
                - all_latents: Latent activations without sparsity constraints (for analysis)
        """
        # Preprocess data
        x_processed, info = self.preprocess(x)
        
        # Compute pre-activations
        latents_pre_act = self.encode_pre_act(x_processed)
        
        # Apply activation function
        latents = self.activation(latents_pre_act)
        latents_caped = self.latent_soft_cap(latents)

        # Decode to reconstruction
        recons = self.decode(latents_caped, info)
        
        # Update activation statistics
        self.update_latent_statistics(latents_caped)
        
        # Handle different activation function types for analysis outputs
        if isinstance(self.activation, TopK):
            # For TopK, return both sparse and full activations
            all_latents = self.activation.forward_eval(latents_pre_act)
            all_latents_caped = self.latent_soft_cap(all_latents)
            all_recons = self.decode(all_latents_caped, info)
            return recons, latents_caped, all_recons, all_latents_caped
        elif isinstance(self.activation, JumpReLU) and self.training:
            # For JumpReLU in training mode, use special training activations
            loss_latents = self.activation.forward_train(latents)
            return recons, loss_latents, recons, latents_caped
        else:
            # For other activations, return the same for both
            return recons, latents_caped, recons, latents_caped


class MatryoshkaAutoencoder(Autoencoder):
    """
    Matryoshka Sparse Autoencoder.
    
    This extends the base Autoencoder with a nested structure of latent representations,
    where different numbers of features can be used depending on computational budget
    or desired level of detail.
    
    The model uses multiple TopK activations with different k values and maintains
    relative importance weights for each level of the hierarchy.
    """
    def __init__(
        self, n_latents: int, n_inputs: int, activation: str = "TopKReLU", tied: bool = False, normalize: bool = False,
        bias_init: torch.Tensor or float = 0.0, init_method: str = "kaiming", latent_soft_cap: float = 30.0,
        nesting_list: List[int] = [16, 32], relative_importance: List[float] = None, *args, **kwargs
    ) -> None:
        """
        Initialize the Matryoshka Sparse Autoencoder.
        
        Args:
            n_latents (int): Dimension of the autoencoder latent space
            n_inputs (int): Dimensionality of the original data
            activation (str, optional): Base activation function name. Defaults to "TopKReLU".
            tied (bool, optional): Whether to tie encoder and decoder weights. Defaults to False.
            normalize (bool, optional): Whether to normalize input data. Defaults to False.
            bias_init (torch.Tensor or float, optional): Initial bias value. Defaults to 0.0.
            init_method (str, optional): Weight initialization method. Defaults to "kaiming".
            latent_soft_cap (float, optional): Soft cap value for latent activations. Defaults to 30.0.
            nesting_list (List[int], optional): List of k values for nested representations. Defaults to [16, 32].
            relative_importance (List[float], optional): Importance weights for each nesting level.
                                                              Defaults to equal weights.
        """
        # Initialize nesting hierarchy
        self.nesting_list = sorted(nesting_list)
        self.relative_importance = relative_importance if relative_importance is not None else [1.0] * len(nesting_list)
        assert len(self.relative_importance) == len(self.nesting_list)
        
        # Ensure activation is TopK-based
        if "TopK" not in activation:
            warnings.warn(f"MatryoshkaAutoencoder: activation {activation} is not a TopK activation. We are changing it to TopKReLU")
            activation = "TopKReLU"
        
        # Initialize with base activation
        base_activation = activation + f"_{self.nesting_list[0]}"
        super().__init__(n_latents, n_inputs, base_activation, tied, normalize, bias_init, init_method, latent_soft_cap)
        
        # Create multiple activations with different k values
        self.activation = nn.ModuleList(
            [get_activation(activation + f"_{nesting}") for nesting in self.nesting_list]
        )
    
    def encode(self, x: torch.Tensor, topk_number: int = None) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """
        Encode input data to multiple latent representations with different sparsity levels.
        
        Args:
            x (torch.Tensor): Input data [batch, n_inputs]
            topk_number (int, optional): Number of top-k activations to keep (for inference).
                                              Defaults to None.
        
        Returns:
            tuple: (encoded_list, last_encoded, info)
                - encoded_list: List of latent activations with different sparsity levels
                - last_encoded: The least sparse latent activations (from largest k value)
                - info: Normalization information dictionary
        """
        x, info = self.preprocess(x)
        pre_encoded = self.encode_pre_act(x)
        
        # Apply each activation function in the hierarchy
        encoded = [activation(pre_encoded) for activation in self.activation]
        caped_encoded = [self.latent_soft_cap(enc) for enc in encoded]
        
        # Apply additional top-k filtering for inference if requested
        if topk_number is not None:
            last_encoded = caped_encoded[-1]
            _, indices = torch.topk(last_encoded, k=topk_number, dim=-1)
            values = torch.gather(last_encoded, -1, indices)
            last_encoded = torch.zeros_like(last_encoded)
            last_encoded.scatter_(-1, indices, values)
        else:
            last_encoded = caped_encoded[-1]
        
        return caped_encoded, last_encoded, info
    
    def decode(self, latents: List[torch.Tensor], info: Dict[str, Any] = None) -> List[torch.Tensor]:
        """
        Decode multiple latent representations to reconstructions.
        
        Args:
            latents (List[torch.Tensor]): List of latent activations at different sparsity levels
            info (Dict[str, Any], optional): Normalization information. Defaults to None.
        
        Returns:
            List[torch.Tensor]: List of reconstructed inputs at different sparsity levels
        """
        # Decode each latent representation
        if self.tied:
            ret = [latent @ self.encoder.t() + self.pre_bias for latent in latents]
        else:
            ret = [latent @ self.decoder + self.pre_bias for latent in latents]
            
        # Denormalize if needed
        if self.normalize:
            assert info is not None
            ret = [re * info["std"] + info["mu"] for re in ret]
            
        return ret

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Matryoshka autoencoder.
        
        Args:
            x (torch.Tensor): Input data [batch, n_inputs]
        
        Returns:
            tuple: (recons_list, latents_list, final_recon, final_latent)
                - recons_list: List of reconstructions at different sparsity levels
                - latents_list: List of latent activations at different sparsity levels
                - final_recon: Reconstruction from the largest k value
                - final_latent: Latent activations from the largest k value
        """
        # Preprocess data
        x_processed, info = self.preprocess(x)
        latents_pre_act = self.encode_pre_act(x_processed)
        
        # Apply each activation in the hierarchy
        latents = [activation(latents_pre_act) for activation in self.activation]
        assert len(latents) == len(self.activation)
        latents_caped = [self.latent_soft_cap(latent) for latent in latents]

        # Decode each level
        recons = self.decode(latents_caped, info)
        assert len(recons) == len(latents)
        
        # Update activation statistics using the largest k
        self.update_latent_statistics(latents_caped[-1])
        
        # Get full activations for analysis
        all_latents = self.activation[0].forward_eval(latents_pre_act)
        all_latents_caped = self.latent_soft_cap(all_latents)
        all_recons = self.decode([all_latents_caped], info)[0]
        
        # Return all reconstructions and the final ones
        return recons, latents_caped, all_recons, all_latents_caped


def load_model(path):
    """
    Load a saved sparse autoencoder model from a file.
    
    This function parses the filename to extract model configuration parameters
    and then loads the saved model weights.
    
    Args:
        path (str): Path to the saved model file (.pt)
        
    Returns:
        tuple: (model, data_mean_center, data_normalized, scaling_factor)
            - model: The loaded Autoencoder model
            - mean_center: Boolean indicating if data was mean-centered
            - target_norm: Target normalization factor for the data
    """
    # Extract configuration from filename
    path_head = path.split("/")[-1]
    path_name = path_head[:path_head.find(".pt")]
    path_name_spited = path_name.split("_")
    
    n_latents = int(path_name_spited.pop(0))
    n_inputs = int(path_name_spited.pop(0))
    activation = path_name_spited.pop(0)
    if "TopK" in activation:
        activation += "_" + path_name_spited.pop(0)
    elif "ReLU" == activation:
        path_name_spited.pop(0)
    if "UW" in path_name_spited[0] or "RW" in path_name_spited[0]:
        path_name_spited.pop(0)
    tied = False if path_name_spited.pop(0) == "False" else True
    normalize = False if path_name_spited.pop(0) == "False" else True
    latent_soft_cap = float(path_name_spited.pop(0))
    
    # Create and load the model
    model = Autoencoder(n_latents, n_inputs, activation, tied=tied, normalize=normalize, latent_soft_cap=latent_soft_cap)
    model_state_dict = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(model_state_Dict['model'])
    mean_center = model_state_Dict['mean_center']
    scaling_factor = model_state_Dict['scaling_factor']
    target_norm = model_state_Dict['target_norm']
    return model, mean_center, scaling_factor, target_norm


class SAE(nn.Module):
    def __init__(self, path: str) -> None:
        """
        Initialize the Sparse Autoencoder (SAE) model.
        
        Args:
            path (str): Path to the saved model file (.pt)
        """
        super().__init__()
        self.model, mean, scaling_factor, _ = load_model(path)
        self.register_buffer("mean", mean.clone().detach() if isinstance(mean, torch.Tensor) else torch.tensor(mean))   
        self.register_buffer("scaling_factor", torch.tensor(scaling_factor)) 
    
    @property
    def input_dim(self) -> int:
        """Return input dimension of the model."""
        return self.model.n_inputs
    
    @property
    def latent_dim(self) -> int:
        """Return latent dimension of the model."""
        return self.model.n_latents

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input data (mean-centering and scaling).
        
        Args:
            x: Input tensor
            
        Returns:
            Preprocessed tensor
        """
        # Mean-center and scale the input
        x = (x - self.mean) * self.scaling_factor
        return x
    
    def postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Post-process output data (denormalization).
        
        Args:
            x: Output tensor
            
        Returns:
            Denormalized tensor
        """
        # Rescale and mean-center the output
        x = x / self.scaling_factor + self.mean
        return x
    
    def encode(self, x: torch.Tensor, topk: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input data to latent representation.
        
        Args:
            x: Input tensor
            topk (int, optional): Number of top-k activations to keep. Defaults to -1 (no sparsity).
            
        Returns:
            Encoded latents and full latents
        """
        # Preprocess input
        x = self.preprocess(x)
        
        # Validate topk constrain
        if topk > 0 and topk < self.model.n_latents:
            topk_number = topk
        else:
            topk_number = None
            
        # Encode using the model
        latents, full_latents, _ = self.model.encode(x, topk_number=topk_number)
        
        return latents, full_latents
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to input space.
        
        Args:
            latents: Latent tensor
            
        Returns:
            Reconstructed input tensor
        """
        # Decode using the model
        reconstructed = self.model.decode(latents)
        
        # Postprocess output
        reconstructed = self.postprocess(reconstructed)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor, topk: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the SAE.
        
        Args:
            x: Input tensor
            topk (int, optional): Number of top-k activations to keep. Defaults to -1 (no sparsity).

        Returns:
            - Post-processed reconstructed tensor
            - Reconstructed tensor
            - Full latent activations
        """
        # Encode to latent space
        _, full_latents = self.encode(x, topk=topk)
        
        # Decode back to input space
        reconstructed = self.model.decode(full_latents)
        
        # Postprocess output
        post_reconstructed = self.postprocess(reconstructed)
        
        # Return reconstructed, post_reconstructed, full_latents
        return post_reconstructed, reconstructed, full_latents
