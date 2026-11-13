import math
import warnings
from tqdm import tqdm

import torch
import random
import numpy as np

"""
Sparse Autoencoder (SAE) Utilities

This module provides utility functions and classes for training and using
Sparse Autoencoders, including dataset handling, learning rate schedulers,
custom activation functions, and various mathematical operations.
"""

class SAEDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset implementation for Sparse Autoencoders.
    
    This class loads data from memory-mapped numpy arrays to efficiently handle
    large datasets without loading everything into memory at once. It also
    handles preprocessing like mean centering and normalization.
    
    The class automatically parses dataset dimensions from the filename,
    which is expected to contain the data shape as the last two underscored
    components (e.g., "dataset_name_10000_768.npy" for 10000 vectors of size 768).
    
    Args:
        data_path (str): Path to the memory-mapped numpy array file
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        mean_center (bool, optional): Whether to center the data by subtracting the mean.
                                     Defaults to False.
        target_norm (float, optional): Target norm for normalization. If None, uses sqrt(vector_size).
                                     If 0.0, no normalization is applied. Defaults to None.
    """
    def __init__(self, data_path: str, dtype: torch.dtype = torch.float32, mean_center: bool = False, target_norm: float = None):
        # Parse vector dimensions from filename
        parts = data_path.split("/")[-1].split(".")[0].split("_")
        self.len, self.vector_size = map(int, parts[-2:])
        
        # Set core attributes
        self.dtype = dtype
        # Use np.load with mmap_mode to properly handle .npy file headers
        self.data = np.load(data_path, mmap_mode='r')
        
        # Special case for representation files (already preprocessed)
        if "repr" in data_path:
            self.mean = torch.zeros(self.vector_size, dtype=dtype)
            self.mean_center = False
            self.scaling_factor = 1.0
            return

        # Set preprocessing configuration
        self.mean_center = mean_center
        self.target_norm = np.sqrt(self.vector_size) if target_norm is None else target_norm

        # Compute statistics if needed
        if self.mean_center or self.target_norm != 0.0:
            self._compute_statistics()
        else:
            self.mean = torch.zeros(self.vector_size, dtype=dtype)
            self.scaling_factor = 1.0

    def _compute_statistics(self, batch_size: int = 10000):
        """
        Compute dataset statistics (mean and scaling factor) in memory-efficient batches.
        
        Args:
            batch_size (int, optional): Number of samples to process at once. Defaults to 10000.
        """
        # Compute mean if mean centering is enabled
        if self.mean_center:
            mean_acc = np.zeros(self.vector_size, dtype=np.float64)  # Use double precision
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy()
                mean_acc += np.sum(batch, axis=0, dtype=np.float64)  # Use double precision for sum
                total += (end - start)

            self.mean = torch.from_numpy((mean_acc / total).astype(np.float32)).to(self.dtype)
        else:
            self.mean = torch.zeros(self.vector_size, dtype=self.dtype)

        # Compute scaling factor if normalization is enabled
        if self.target_norm != 0.0:
            squared_norm_sum = 0.0
            total = 0

            for start in range(0, self.len, batch_size):
                end = min(start + batch_size, self.len)
                batch = self.data[start:end].copy().astype(np.float64)  # Use double precision
                # Center the batch if needed
                batch = batch - self.mean.numpy().astype(np.float64)
                squared_norm_sum += np.sum(np.square(batch))
                total += (end - start)

            avg_squared_norm = squared_norm_sum / total
            self.scaling_factor = float(self.target_norm / np.sqrt(avg_squared_norm))
        else:
            self.scaling_factor = 1.0

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.len
    
    def process_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Process data for the autoencoder (subtract mean and apply scaling).
        
        Args:
            data (torch.Tensor): Input data tensor
            
        Returns:
            torch.Tensor: Processed data tensor
        """        
        data.sub_(self.mean)
        data.mul_(self.scaling_factor)
        
        return data
    
    def unprocess_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the processing of data (apply inverse scaling and add mean).
        
        Args:
            data (torch.Tensor): Input data tensor
            
        Returns:
            torch.Tensor: Unprocessed data tensor
        """        
        data.div_(self.scaling_factor)
        data.add_(self.mean)
        
        return data

    @torch.no_grad()
    def __getitem__(self, idx):
        """
        Get a preprocessed data sample at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            torch.Tensor: Preprocessed data sample
        """
        torch_data = torch.tensor(self.data[idx])
        output = self.process_data(torch_data.clone())
        return output.to(self.dtype)


class LinearDecayLR(torch.optim.lr_scheduler.LambdaLR):
    """
    Learning rate scheduler with a constant phase followed by linear decay.
    
    The learning rate remains constant for a specified fraction of total epochs,
    then decays linearly to zero for the remaining epochs.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to adjust
        total_epochs (int): Total number of training epochs
        decay_time (float, optional): Fraction of total epochs before decay starts.
                                     Defaults to 0.8 (80% of training).
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """
    def __init__(self, optimizer, total_epochs, decay_time = 0.8, last_epoch=-1):
        def lr_lambda(epoch):
            if epoch < int(decay_time * total_epochs):
                return 1.0
            return max(0.0, (total_epochs - epoch) / ((1-decay_time) * total_epochs))
        
        super().__init__(optimizer, lr_lambda, last_epoch)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with warmup and cosine annealing.
    
    This scheduler implements:
    1. Linear warmup from initial_lr (max_lr * final_lr_factor) to max_lr during the warmup epoch
    2. Cosine annealing from max_lr to final_lr (max_lr * final_lr_factor) for the remaining epochs
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to adjust
        max_lr (float): Maximum learning rate after warmup
        total_epochs (int): Total number of training epochs
        warmup_epoch (int, optional): Number of warmup epochs. Defaults to 1.
        final_lr_factor (float, optional): Ratio of final LR to max LR. Defaults to 0.1.
        last_epoch (int, optional): The index of the last epoch. Defaults to -1.
    """
    def __init__(self, optimizer, max_lr, total_epochs, warmup_epoch=1, 
                 final_lr_factor=0.1, last_epoch=-1):
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.warmup_epoch = warmup_epoch
        self.initial_lr = max_lr * final_lr_factor
        self.final_lr = max_lr * final_lr_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Calculate the learning rate for the current epoch.
        
        Returns:
            list: Learning rates for each parameter group
        """
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")

        # During warmup (first epoch)
        if self.last_epoch < self.warmup_epoch:
            # Linear interpolation from initial_lr to max_lr
            alpha = self.last_epoch / self.warmup_epoch
            return [self.initial_lr + (self.max_lr - self.initial_lr) * alpha 
                    for _ in self.base_lrs]
        
        # After warmup - Cosine annealing
        else:
            # Adjust epoch count to start cosine annealing after warmup
            current = self.last_epoch - self.warmup_epoch
            total = self.total_epochs - self.warmup_epoch
            
            # Implement cosine annealing
            cosine_factor = (1 + math.cos(math.pi * current / total)) / 2
            return [self.final_lr + (self.max_lr - self.final_lr) * cosine_factor 
                    for _ in self.base_lrs]


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed (int): The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """
    Determine the best available device for PyTorch computation.
    
    Returns:
        torch.device: The selected device (CUDA if available, MPS on Apple Silicon, CPU otherwise)
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return torch.device(device)


from typing import Tuple
def normalize_data(x: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize input data to zero mean and unit variance.
    
    Args:
        x (torch.Tensor): Input tensor to normalize
        eps (float, optional): Small constant for numerical stability. Defaults to 1e-5.
        
    Returns:
        tuple: (normalized_data, mean, std)
            - normalized_data: Data normalized to zero mean and unit variance
            - mean: Mean of the original data (for denormalization)
            - std: Standard deviation of the original data (for denormalization)
    """
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


@torch.no_grad()
def geometric_median(dataset: torch.utils.data.Dataset, eps: float = 1e-5, 
                    device: torch.device = torch.device("cpu"), 
                    max_number: int = 925117, max_iter: int = 1000) -> torch.Tensor:
    """
    Compute the geometric median of a dataset using Weiszfeld's algorithm.
    
    The geometric median is a generalization of the median to multiple dimensions
    and is robust to outliers. This implementation uses iterative approximation
    with early stopping based on convergence.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset to compute median for
        eps (float, optional): Convergence threshold. Defaults to 1e-5.
        device (torch.device, optional): Computation device. Defaults to CPU.
        max_number (int, optional): Maximum number of samples to use. Defaults to 925117.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        
    Returns:
        torch.Tensor: The geometric median vector
    """
    # Sample a subset of the dataset if it's large
    indices = torch.randperm(len(dataset))[:min(len(dataset), max_number)]
    X = dataset[indices]
    
    # Move data to device
    try:
        X = X.to(device)
    except Exception as e:
        warnings.warn(f"Error moving dataset to device: {device}, using default device {X.device}")
    
    # Initialize with arithmetic mean
    y = torch.mean(X, dim=0)
    progress_bar = tqdm(range(max_iter), desc="Geometric Median Iteration", leave=False)
    
    # Weiszfeld's algorithm
    for _ in progress_bar:
        # Compute distances to current estimate
        D = torch.norm(X - y, dim=1)
        nonzeros = (D != 0)  # Avoid division by zero
        
        # Compute weights for non-zero distances
        Dinv = 1 / D[nonzeros]
        Dinv_sum = torch.sum(Dinv)
        W = Dinv / Dinv_sum
        
        # Weighted average of points
        T = torch.sum(W.view(-1, 1) * X[nonzeros], dim=0)
        
        # Handle special case when some points equal the current estimate
        num_zeros = len(X) - torch.sum(nonzeros)
        if num_zeros == 0:
            # No points equal the current estimate
            y1 = T
        else:
            # Some points equal the current estimate
            R = T * Dinv_sum / (Dinv_sum - num_zeros)
            r = torch.norm(R - y)
            progress_bar.set_postfix({"r": r.item()})
            if r < eps:
                return y
            y1 = R
        
        # Check convergence
        if torch.norm(y - y1) < eps:
            return y1
        
        y = y1
    
    # Return best estimate after max iterations
    return y


def calculate_vector_mean(dataset: torch.utils.data.Dataset,
                          batch_size: int = 10000,
                          num_workers: int = 4) -> torch.Tensor:
    """
    Efficiently calculate the mean of vectors in a dataset.
    
    This function processes the dataset in batches to handle large datasets
    that might not fit in memory all at once.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset containing vectors
        batch_size (int, optional): Batch size for processing. Defaults to 10000.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        
    Returns:
        torch.Tensor: Mean vector of the dataset
    """
    # Use DataLoader to efficiently iterate through the dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False  # No need to shuffle for calculating mean
    )
    
    # Initialize sum and count
    vector_sum = torch.zeros_like(dataset[0])
    count = 0
    
    # Iterate through batches
    for batch in tqdm(dataloader, desc="Calculating Mean Vector", leave=False):
        batch_count = batch.size(0)
        vector_sum += batch.sum(dim=0)
        count += batch_count
    
    # Calculate mean
    mean_vector = vector_sum / count
    
    return mean_vector


class RectangleFunction(torch.autograd.Function):
    """
    Custom autograd function that implements a rectangle function.
    
    This function outputs 1.0 for inputs between -0.5 and 0.5, and 0.0 elsewhere.
    The gradient is non-zero only within this interval.
    
    Used as a building block for other activation functions with custom gradients.
    """
    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the rectangle function.
        
        Args:
            ctx: Context for saving variables for backward
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with values in {0.0, 1.0}
        """
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the rectangle function.
        
        Args:
            ctx: Context with saved variables
            grad_output (torch.Tensor): Gradient from subsequent layers
            
        Returns:
            torch.Tensor: Gradient with respect to input
        """
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) or (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(torch.autograd.Function):
    """
    Custom autograd function implementing a thresholded ReLU with learnable threshold.
    
    This activation function passes values through only if they exceed a learned threshold.
    It has custom gradients for both the input and the threshold parameter.
    """
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        """
        Forward pass of the JumpReLU function.
        
        Args:
            ctx: Context for saving variables for backward
            x (torch.Tensor): Input tensor
            log_threshold (torch.Tensor): Log of the threshold value (learned parameter)
            bandwidth (float): Bandwidth parameter for gradient approximation
            
        Returns:
            torch.Tensor: Output tensor
        """
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the JumpReLU function.
        
        Args:
            ctx: Context with saved variables
            grad_output (torch.Tensor): Gradient from subsequent layers
            
        Returns:
            tuple: (input_gradient, threshold_gradient, None)
        """
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        
        # Gradient with respect to x
        x_grad = (x > threshold).float() * grad_output
        
        # Gradient with respect to threshold
        # Uses rectangle function to approximate the dirac delta
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(torch.autograd.Function):
    """
    Custom autograd function implementing a step function with learnable threshold.
    
    This activation function outputs 1 for values above a threshold and 0 otherwise.
    It has custom gradients for both the input and the threshold parameter.
    """
    @staticmethod
    def forward(ctx, x, log_threshold, bandwidth):
        """
        Forward pass of the step function.
        
        Args:
            ctx: Context for saving variables for backward
            x (torch.Tensor): Input tensor
            log_threshold (torch.Tensor): Log of the threshold value (learned parameter)
            bandwidth (float): Bandwidth parameter for gradient approximation
            
        Returns:
            torch.Tensor: Binary output tensor with values in {0.0, 1.0}
        """
        ctx.save_for_backward(x, log_threshold, torch.tensor(bandwidth))
        threshold = torch.exp(log_threshold)
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the step function.
        
        Args:
            ctx: Context with saved variables
            grad_output (torch.Tensor): Gradient from subsequent layers
            
        Returns:
            tuple: (input_gradient, threshold_gradient, None)
        """
        x, log_threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        threshold = torch.exp(log_threshold)
        
        # No gradient with respect to x (step function)
        x_grad = torch.zeros_like(x)
        
        # Gradient with respect to threshold
        # Uses rectangle function to approximate the dirac delta
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        
        return x_grad, threshold_grad, None  # None for bandwidth
