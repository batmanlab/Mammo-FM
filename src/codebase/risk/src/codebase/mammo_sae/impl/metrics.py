import torch


def calculate_similarity_metrics(original_matrix: torch.Tensor, reconstruction_matrix: torch.Tensor):
    """
    Calculate cosine similarity and Euclidean distance between original and reconstructed vectors.
    
    Computes the average cosine similarity and Euclidean distance for corresponding pairs of
    vectors in the original and reconstruction matrices.
    
    Args:
        original_matrix (torch.Tensor): Original data matrix of shape [batch_size, feature_dim]
        reconstruction_matrix (torch.Tensor): Reconstructed data matrix of shape [batch_size, feature_dim]
        
    Returns:
        tuple:
            - torch.Tensor: Mean cosine similarity (higher values indicate better reconstruction)
            - torch.Tensor: Mean Euclidean distance (lower values indicate better reconstruction)
    """
    # Calculate cosine similarity for each pair
    # First normalize the vectors
    original_norm = original_matrix.norm(dim=-1, keepdim=True)
    reconstruction_norm = reconstruction_matrix.norm(dim=-1, keepdim=True)
    
    original_normalized = original_matrix / original_norm
    reconstruction_normalized = reconstruction_matrix / reconstruction_norm
    
    # Calculate dot product of normalized vectors
    cosine_similarities = reconstruction_normalized @ original_normalized.T
    cosine_similarities = torch.diagonal(cosine_similarities)
    
    # Calculate Euclidean distance for each pair
    euclidean_distances = torch.norm(original_matrix - reconstruction_matrix, dim=-1)
    
    return torch.mean(cosine_similarities), torch.mean(euclidean_distances)


def identify_dead_neurons(latent_bias: torch.Tensor, threshold: float = 10**(-5.5)) -> torch.Tensor:
    """
    Identify dead neurons based on their bias values.
    
    Dead neurons are those with bias magnitudes below a specified threshold,
    indicating that they may not be activating significantly during training.
    
    Args:
        latent_bias (torch.Tensor): Bias vector for latent neurons
        threshold (float, optional): Threshold below which a neuron is considered dead.
                                     Defaults to 10^(-5.5).
    
    Returns:
        torch.Tensor: Indices of dead neurons
    """
    dead_neurons = torch.where(torch.abs(latent_bias) < threshold)[0]
    return dead_neurons


def hsic_unbiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute the unbiased Hilbert-Schmidt Independence Criterion (HSIC).
    
    This implementation follows Equation 5 in Song et al. (2012), which provides
    an unbiased estimator of HSIC. This measure quantifies the dependency between
    two sets of variables represented by their kernel matrices.
    
    Reference: 
        Song, L., Smola, A., Gretton, A., & Borgwardt, K. (2012).
        "A dependence maximization view of clustering."
        https://jmlr.csail.mit.edu/papers/volume13/song12a/song12a.pdf
    
    Args:
        K (torch.Tensor): First kernel matrix of shape [n, n]
        L (torch.Tensor): Second kernel matrix of shape [n, n]
        
    Returns:
        torch.Tensor: Unbiased HSIC value (scalar)
    """
    m = K.shape[0]

    # Zero out the diagonal elements of K and L
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    # Compute HSIC using the formula in Equation 5
    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute the biased Hilbert-Schmidt Independence Criterion (HSIC).
    
    This is the original form used in Centered Kernel Alignment (CKA).
    It's computationally simpler than the unbiased version but may have
    statistical bias, especially for small sample sizes.
    
    Args:
        K (torch.Tensor): First kernel matrix of shape [n, n]
        L (torch.Tensor): Second kernel matrix of shape [n, n]
        
    Returns:
        torch.Tensor: Biased HSIC value (scalar)
    """
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def cknna(feats_A: torch.Tensor, feats_B: torch.Tensor, topk: int = 10, 
         distance_agnostic: bool = False, unbiased: bool = True) -> float:
    """
    Compute the Centered Kernel Nearest Neighbor Alignment (CKNNA). From:
    https://github.com/minyoungg/platonic-rep/blob/4dd084e1b96804ddd07ae849658fbb69797e319b/metrics.py#L180
    
    CKNNA is a variant of CKA that only considers k-nearest neighbors when computing
    similarity. This makes it more robust to outliers and more sensitive to local
    structure in the data.
    
    Args:
        feats_A (torch.Tensor): First feature matrix of shape [n_samples, n_features_A]
        feats_B (torch.Tensor): Second feature matrix of shape [n_samples, n_features_B]
        topk (int, optional): Number of nearest neighbors to consider. Defaults to 10.
        distance_agnostic (bool, optional): If True, only considers binary neighborhood
                                           membership without weighting by similarity.
                                           Defaults to False.
        unbiased (bool, optional): If True, uses unbiased HSIC estimator. 
                                  Defaults to True.
    
    Returns:
        float: CKNNA similarity score between 0 and 1, where higher values
               indicate greater similarity between the feature spaces
               
    Raises:
        ValueError: If topk is less than 2
    """
    n = feats_A.shape[0]
            
    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")
    
    if topk is None:
        topk = feats_A.shape[0] - 1
                        
    # Compute kernel matrices (linear kernels)
    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    device = feats_A.device

    def similarity(K, L, topk):
        """
        Compute similarity based on nearest neighbor intersection.
        
        This inner function computes similarity between two kernel matrices
        based on their shared nearest neighbor structure.
        """                    
        if unbiased:            
            # Fill diagonal with -inf to exclude self-similarity when finding topk
            K_hat = K.clone().fill_diagonal_(float("-inf"))
            L_hat = L.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = K, L

        # Get topk indices for each row
        _, topk_K_indices = torch.topk(K_hat, topk, dim=1)
        _, topk_L_indices = torch.topk(L_hat, topk, dim=1)
        
        # Create masks for nearest neighbors
        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)
        
        # Intersection of nearest neighbors
        mask = mask_K * mask_L
                    
        if distance_agnostic:
            # Simply count shared neighbors without considering similarity values
            sim = mask * 1.0
        else:
            # Compute HSIC on the masked kernel matrices
            if unbiased:
                sim = hsic_unbiased(mask * K, mask * L)
            else:
                sim = hsic_biased(mask * K, mask * L)
        return sim

    # Compute similarities
    sim_kl = similarity(K, L, topk)  # Cross-similarity
    sim_kk = similarity(K, K, topk)  # Self-similarity of K
    sim_ll = similarity(L, L, topk)  # Self-similarity of L
            
    # Normalized similarity (similar to correlation)
    return sim_kl.item() / (torch.sqrt(sim_kk * sim_ll) + 1e-6).item()


def explained_variance_full(
    original_input: torch.Tensor,
    reconstruction: torch.Tensor,
) -> float:
    """
    Computes the explained variance between the original input and its reconstruction.

    The explained variance is a measure of how much of the variance in the original input
    is captured by the reconstruction. It is calculated as:
        1 - (variance of the reconstruction error / total variance of the original input)

    Args:
        original_input (torch.Tensor): The original input tensor.
        reconstruction (torch.Tensor): The reconstructed tensor.

    Returns:
        float: The explained variance score, a value between 0 and 1.
            A value of 1 indicates perfect reconstruction.
    """
    variance = (original_input - reconstruction).var(dim=-1)
    total_variance = original_input.var(dim=-1)
    return variance / total_variance


def explained_variance(
    original_input: torch.Tensor,
    reconstruction: torch.Tensor,
) -> float:
    """
    Computes the explained variance between the original input and its reconstruction.

    The explained variance is a measure of how much of the variance in the original input
    is captured by the reconstruction. It is calculated as:
        1 - (variance of the reconstruction error / total variance of the original input)

    Args:
        original_input (torch.Tensor): The original input tensor.
        reconstruction (torch.Tensor): The reconstructed tensor.

    Returns:
        float: The explained variance score, a value between 0 and 1.
            A value of 1 indicates perfect reconstruction.
    """
    return explained_variance_full(original_input, reconstruction).mean(dim=-1).item()


def orthogonal_decoder(decoder: torch.Tensor) -> float:
    """
    Compute the degree of non-orthogonality in decoder weights.
    
    This metric measures how close the decoder feature vectors are to being
    orthogonal to each other. Lower values indicate more orthogonal features,
    which is often desirable for sparse representation learning.
    
    Args:
        decoder (torch.Tensor): Decoder weight matrix of shape [n_latents, n_inputs]
        
    Returns:
        float: Orthogonality score (lower is better, 0 means perfectly orthogonal)
    """
    # Compute dot products between all pairs of decoder vectors
    logits = decoder @ decoder.T
    
    # Create a mask to only consider off-diagonal elements
    I = 1 - torch.eye(decoder.shape[0], device=decoder.device, dtype=decoder.dtype)
    
    # Compute mean squared dot product (excluding diagonal)
    return ((logits * I) ** 2).mean().item()


def normalized_mean_absolute_error(
    original_input: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    """
    Compute normalized mean absolute error between original and reconstructed data.
    
    This metric normalizes the MAE by the mean absolute value of the original input,
    making it scale-invariant and more comparable across different datasets.
    
    Args:
        original_input (torch.Tensor): Original input data of shape [batch, n_inputs]
        reconstruction (torch.Tensor): Reconstructed data of shape [batch, n_inputs]
        
    Returns:
        torch.Tensor: Normalized MAE for each sample in the batch
    """
    return (
        torch.abs(reconstruction - original_input).mean(dim=1) / 
        torch.abs(original_input).mean(dim=1)
    )


def normalized_mean_squared_error(
    original_input: torch.Tensor,
    reconstruction: torch.Tensor,
) -> torch.Tensor:
    """
    Compute normalized mean squared error between original and reconstructed data.
    
    This metric normalizes the MSE by the mean squared value of the original input,
    making it scale-invariant and more comparable across different datasets.
    Also known as the Fraction of Variance Unexplained (FVU).
    
    Args:
        original_input (torch.Tensor): Original input data of shape [batch, n_inputs]
        reconstruction (torch.Tensor): Reconstructed data of shape [batch, n_inputs]
        
    Returns:
        torch.Tensor: Normalized MSE for each sample in the batch
    """
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1) / 
        (original_input**2).mean(dim=1)
    )


def l0_messure(sample: torch.Tensor) -> torch.Tensor:
    """
    Compute the L0 measure (sparsity) of feature activations.
    
    The L0 measure counts the proportion of zero elements in the activation,
    providing a direct measure of sparsity. Higher values indicate greater
    sparsity (more zeros).
    
    Note: The function name contains a spelling variant ("messure" vs "measure")
    but is kept for backward compatibility.
    
    Args:
        sample (torch.Tensor): Activation tensor of shape [batch, n_features]
        
    Returns:
        torch.Tensor: Proportion of zero elements for each sample in the batch
    """
    return (sample == 0).float().mean(dim=1)