import torch
import inspect
import argparse
import logging
from tqdm import tqdm
from dataclasses import asdict

from .metrics import calculate_similarity_metrics, identify_dead_neurons, orthogonal_decoder, cknna, explained_variance
from .utils import SAEDataset, set_seed, get_device, geometric_median, calculate_vector_mean, LinearDecayLR, CosineWarmupScheduler
from .config import get_config
from .sae import Autoencoder, MatryoshkaAutoencoder
from .loss import SAELoss

"""
Sparse Autoencoder (SAE) Training Script

This script provides a complete pipeline for training various types of sparse autoencoder models,
including standard SAEs with different activation functions and Matryoshka SAEs with nested
feature hierarchies. It handles training, evaluation, and model saving with configurable
hyperparameters.
"""

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the SAE training script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments with the following fields:
            - dataset_train: Path to the training dataset
            - dataset_test: Path to the testing/validation dataset
            - model: Model architecture to train (e.g., "ReLUSAE", "TopKSAE")
            - activation: Activation function to use
            - epochs: Number of training epochs
            - learning_rate: Initial learning rate
            - expansion_factor: Ratio of latent dimensions to input dimensions
    """
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder (SAE) models")
    parser.add_argument("-dt", "--dataset_train", type=str, required=True, 
                       help="Path to training dataset file (.npy)")
    parser.add_argument("-ds", "--dataset_test", type=str, required=True, 
                       help="Path to testing/validation dataset file (.npy)")
    parser.add_argument("-dm", "--dataset_second_modality", type=str, default=None,
                       help="Path to second modality dataset file (.npy)")
    parser.add_argument("-m", "--model", type=str, required=True, 
                       choices=["ReLUSAE", "TopKSAE", "BatchTopKSAE", "MSAE_UW", "MSAE_RW"], 
                       help="Model architecture to train")
    parser.add_argument("-a", "--activation", type=str, required=True, 
                       help="Activation function (e.g., 'ReLU_003', 'TopKReLU_64')")
    parser.add_argument("-e", "--epochs", type=int, default=100, 
                       help="Number of training epochs")
    parser.add_argument("-ef", "--expansion_factor", type=float, default=1.0,
                       help="Ratio of latent dimensions to input dimensions")
    parser.add_argument("--max_nesting", type=int, default=None,
                       help="Maximum nesting level for MSAE. Default uses n_latents (full hierarchy).")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to load for continued training/fine-tuning.")
    parser.add_argument("--lr", type=float, default=None,
                       help="Override learning rate (useful for fine-tuning with lower lr).")
    parser.add_argument("--freeze_neurons", type=str, default=None,
                       help="Comma-separated list of neuron indices to freeze during training (e.g., '4793,2204,15918,9384,2565,13702')")

    return parser.parse_args()


def eval(model, eval_loader, loss_fn, device, cfg):
    # Evaluation phase
    loss_all = 0.0
    recon_loss_all = 0.0
    sparse_loss_all = 0.0
    cknna_score_sparse_sum = 0.0
    cknna_score_all_sum = 0.0
    fvu_score_all_sum = 0.0
    fvu_score_sparse_sum = 0.0
    diagonal_cs_sparse_sum = 0.0
    diagonal_cs_all_sum = 0.0
    mae_distance_sparse_sum = 0.0
    mae_distance_all_sum = 0.0
    od_sum = 0.0
    sparsity_sparse_sum = 0.0
    sparsity_all_sum = 0.0
    
    # Switch to evaluation mode
    model.eval()
    for step, embeddings in enumerate(tqdm(eval_loader, desc="Evaluation")):
        embeddings = embeddings.to(device)
        
        # Forward pass without gradient computation
        with torch.no_grad():
            recons_sparse, repr_sparse, recons_all, repr_all = model(embeddings)
            loss, recon_loss, sparse_loss = loss_fn(recons_all, embeddings, repr_all)
        
        if cfg.model.use_matryoshka:
            recons_sparse = recons_sparse[0]
            repr_sparse = repr_sparse[0]
        
        # Accumulate loss metrics
        loss_all += loss.item()
        recon_loss_all += recon_loss.item()
        sparse_loss_all += sparse_loss.item()
        
        # Accumulate CKNNA scores
        cknna_score_sparse_sum += cknna(recons_sparse, embeddings)
        cknna_score_all_sum += cknna(recons_all, embeddings)
        
        # Accumulate similarity metrics
        fvu_score_sparse_sum += explained_variance(embeddings, recons_sparse)
        fvu_score_all_sum += explained_variance(embeddings, recons_all)
        distance_sparse = calculate_similarity_metrics(embeddings, recons_sparse)
        distance_all = calculate_similarity_metrics(embeddings, recons_all)
        diagonal_cs_sparse_sum += distance_sparse[0]
        mae_distance_sparse_sum += distance_sparse[1]
        diagonal_cs_all_sum += distance_all[0]
        mae_distance_all_sum += distance_all[1]
        
        # Accumulate orthogonality measure
        od_sum += orthogonal_decoder(model.decoder)
        
        # Accumulate sparsity measures
        sparsity_sparse_sum += (repr_sparse == 0.0).float().mean(axis=-1).mean()
        sparsity_all_sum += (repr_all == 0.0).float().mean(axis=-1).mean()
    
    # Log evaluation metrics (averaged over batches)
    logger.info("Evaluation results:")
    logger.info(f"  Loss: {loss_all / len(eval_loader):.6f}")
    logger.info(f"  Reconstruction Loss: {recon_loss_all / len(eval_loader):.6f}")
    logger.info(f"  Sparsity Loss: {sparse_loss_all / len(eval_loader):.6f}")
    logger.info(f"  FVU Sparse: {fvu_score_sparse_sum / len(eval_loader):.4f}")
    logger.info(f"  FVU All: {fvu_score_all_sum / len(eval_loader):.4f}")
    logger.info(f"  CKNNA Sparse: {cknna_score_sparse_sum / len(eval_loader):.4f}")
    logger.info(f"  CKNNA All: {cknna_score_all_sum / len(eval_loader):.4f}")
    logger.info(f"  Cosine Similarity Sparse: {diagonal_cs_sparse_sum / len(eval_loader):.4f}")
    logger.info(f"  MAE Distance Sparse: {mae_distance_sparse_sum / len(eval_loader):.4f}")
    logger.info(f"  Cosine Similarity All: {diagonal_cs_all_sum / len(eval_loader):.4f}")
    logger.info(f"  MAE Distance All: {mae_distance_all_sum / len(eval_loader):.4f}")
    logger.info(f"  Sparsity Sparse: {sparsity_sparse_sum / len(eval_loader):.4f}")
    logger.info(f"  Sparsity All: {sparsity_all_sum / len(eval_loader):.4f}")
    logger.info(f"  Orthogonal Decoder Loss: {od_sum / len(eval_loader):.6f}")

def main(args):
    """
    Main training function for Sparse Autoencoders.
    
    This function handles the complete training pipeline:
    1. Setting up configuration based on model type and arguments
    2. Loading and preparing datasets
    3. Initializing the appropriate model (standard or Matryoshka SAE)
    4. Setting up loss function, optimizer, and learning rate scheduler
    5. Executing the training loop with periodic evaluation
    6. Tracking metrics including reconstruction quality, sparsity, and dead neurons
    7. Saving the trained model with relevant metadata
    
    Args:
        args (argparse.Namespace): Command line arguments from parse_args()
    """
    logger.info("Starting training with the following arguments:")
    logger.info(args)
    
    # Get configuration based on model type
    cfg = get_config(args.model)
    cfg.training.epochs = args.epochs
    
    # Set the random seed for reproducibility
    set_seed(cfg.training.seed)
    
    # Set the device (GPU/CPU)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load datasets
    train_ds = SAEDataset(
        args.dataset_train, 
        dtype=cfg.training.dtype, 
        mean_center=cfg.training.mean_center, 
        target_norm=cfg.training.target_norm
    )
    eval_ds = SAEDataset(
        args.dataset_test, 
        dtype=cfg.training.dtype, 
        mean_center=cfg.training.mean_center, 
        target_norm=cfg.training.target_norm
    )
    logger.info(f"Training dataset length: {len(train_ds)}, Evaluation dataset length: {len(eval_ds)}, Embedding size: {train_ds.vector_size}")
    logger.info(f"Training dataset mean center: {train_ds.mean.mean()}, Scaling factor: {train_ds.scaling_factor} with target norm {train_ds.target_norm}")
    logger.info(f"Evaluation dataset mean center: {eval_ds.mean.mean()}, Scaling factor: {eval_ds.scaling_factor} with target norm {eval_ds.target_norm}")
    assert train_ds.vector_size == eval_ds.vector_size, "Training and evaluation datasets must have the same embedding size"
    if args.dataset_second_modality is not None:
        eval_ds_second = SAEDataset(
            args.dataset_second_modality,
            dtype=cfg.training.dtype,
            mean_center=cfg.training.mean_center,
            target_norm=cfg.training.target_norm
        )
        logger.info(f"Second modality dataset mean center: {eval_ds_second.mean.mean()}, Scaling factor: {eval_ds_second.scaling_factor} with target norm {eval_ds_second.target_norm}")
        assert train_ds.vector_size == eval_ds_second.vector_size, "Training and second modality datasets must have the same embedding size"
    
    # Set model parameters based on dataset and arguments
    cfg.model.n_inputs = train_ds.vector_size
    
    # Calculate number of latent dimensions using expansion factor
    cfg.model.n_latents = int(args.expansion_factor * train_ds.vector_size)
    logger.info(f"Expansion factor: {args.expansion_factor}, Latent dimensions: {cfg.model.n_latents}")
    
    # Extract l1 from ReLU if applied
    if args.model == "ReLUSAE" and "_" in args.activation:
        args.activation, sparse_weight = args.activation.split("_")
        cfg.loss.sparse_weight = float(f"0.{sparse_weight}")
        logger.info(f"Changing sparsity weight value to {cfg.loss.sparse_weight}")
        
    # Override activation if specified in arguments
    if args.activation:
        cfg.model.activation = args.activation

    # Configure Matryoshka SAE parameters if applicable
    if cfg.model.use_matryoshka:
        # Max nesting: use CLI arg if provided, otherwise default to n_latents (full hierarchy)
        if args.max_nesting is not None:
            max_nesting = min(args.max_nesting, cfg.model.n_latents)
        else:
            max_nesting = cfg.model.n_latents  # Default: full hierarchy up to n_latents
        
        # Generate nesting list if a single value was provided
        if isinstance(cfg.model.nesting_list, int):
            logger.info(f"Generating nesting list from {cfg.model.nesting_list} to {max_nesting}")
            start = [cfg.model.nesting_list]
            while start[-1] < max_nesting:
                new_k = start[-1] * 2
                if new_k > max_nesting:
                    break
                start.append(new_k)
            
            if max_nesting not in start:
                start.append(max_nesting)
            cfg.model.nesting_list = start
        
        # Set importance weights for different nesting levels
        if cfg.model.relative_importance == "RW":
            # Reverse weighting - higher weight for larger k values
            cfg.model.relative_importance = list(reversed(list(range(1, len(cfg.model.nesting_list)+1))))
        elif cfg.model.relative_importance == "UW":
            # Uniform weighting - equal weight for all k values
            cfg.model.relative_importance = [1.0] * len(cfg.model.nesting_list)

        logger.info(f"Using Matryoshka with nesting list: {cfg.model.nesting_list} and weighting function: {cfg.model.relative_importance}")
    else:
        logger.info(f"Using standard SAE with {cfg.model.activation} activation")
    
    # Calculate bias initialization (median or zero)
    logger.info(f"Calculating bias initialization with median: {cfg.training.bias_init_median}")
    bias_init = 0.0
    if cfg.training.bias_init_median:
        # Use geometric median of a subset of data points for robustness
        bias_init = geometric_median(train_ds, device=device, max_number=len(train_ds)//10)
    logger.info(f"Bias initialization: {bias_init}")
    
    # Initialize the appropriate model type
    if cfg.model.use_matryoshka:
        model = MatryoshkaAutoencoder(bias_init=bias_init, **asdict(cfg.model))
    else:
        model = Autoencoder(bias_init=bias_init, **asdict(cfg.model))
    model = model.to(device)

    # Load checkpoint if provided (for fine-tuning/continued training)
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
        logger.info("Checkpoint loaded successfully")
        # Optionally load mean_center and scaling_factor if they exist
        if 'mean_center' in checkpoint:
            logger.info(f"Loaded mean_center from checkpoint")
        if 'scaling_factor' in checkpoint:
            logger.info(f"Loaded scaling_factor from checkpoint")

    # Override learning rate if specified (useful for fine-tuning)
    if args.lr is not None:
        cfg.training.lr = args.lr
        logger.info(f"Overriding learning rate to {args.lr}")

    # Parse frozen neurons if specified
    frozen_neurons = []
    if args.freeze_neurons is not None:
        frozen_neurons = [int(n.strip()) for n in args.freeze_neurons.split(',')]
        logger.info(f"Freezing {len(frozen_neurons)} neurons: {frozen_neurons}")

    # Prepare loss function
    # Use zeros or calculate mean from dataset depending on config
    mean_input = torch.zeros((train_ds.vector_size,), dtype=cfg.training.dtype)
    if not cfg.training.mean_center:
        mean_input = calculate_vector_mean(train_ds, num_workers=cfg.training.num_workers)
    
    mean_input = mean_input.to(device)
    loss_fn = SAELoss(
        reconstruction_loss=cfg.loss.reconstruction_loss,
        sparse_loss=cfg.loss.sparse_loss,
        sparse_weight=cfg.loss.sparse_weight,
        mean_input=mean_input,
    )
    
    # Prepare the optimizer with adaptive settings based on device
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and "cuda" in device.type
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training.lr, 
        betas=(cfg.training.beta1, cfg.training.beta2), 
        eps=cfg.training.eps, 
        weight_decay=cfg.training.weight_decay, 
        fused=use_fused
    )
    
    # Prepare the learning rate scheduler
    if cfg.training.scheduler == 1:
        # Linear decay scheduler
        scheduler = LinearDecayLR(optimizer, cfg.training.epochs, decay_time=cfg.training.decay_time)
    elif cfg.training.scheduler == 2:
        # Cosine annealing with warmup
        scheduler = CosineWarmupScheduler(
            optimizer, 
            max_lr=cfg.training.lr, 
            warmup_epoch=1, 
            final_lr_factor=0.1, 
            total_epochs=cfg.training.epochs
        )
    else:
        # No scheduler
        scheduler = None
    
    # Prepare the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True,
        drop_last=True  # Drop last incomplete batch to avoid CKNNA issues with small batches
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_ds,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False
    )
    if args.dataset_second_modality is not None:
        eval_loader_second = torch.utils.data.DataLoader(
            eval_ds_second,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
            shuffle=False
        )

    # Training loop
    global_step = 0
    numb_of_dead_neurons = 0
    dead_neurons = []
    
    for epoch in range(cfg.training.epochs):
        model.train()
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs}")
        
        # Training loop for current epoch
        for step, embeddings in enumerate(tqdm(train_loader, desc="Training")):
            optimizer.zero_grad()
            global_step += 1
            embeddings = embeddings.to(device)
            
            # Forward pass through model
            recons_sparse, repr_sparse, recons_all, repr_all = model(embeddings)
            
            # Compute loss based on model type
            if cfg.model.use_matryoshka:
                # For Matryoshka models, compute weighted loss across all nesting levels
                sparse_loss = loss_fn(recons_all, embeddings, repr_all)[-1]
                recon_loss = loss_fn(recons_sparse[0], embeddings, repr_all)[1]
                
                # Weight reconstruction losses by relative importance
                loss_recon_all = torch.tensor(0., requires_grad=True, device=device)
                for i in range(len(recons_sparse)):
                    current_loss = loss_fn(recons_sparse[i], embeddings, repr_sparse[i])[1]
                    loss_recon_all = loss_recon_all + current_loss * model.relative_importance[i]

                # Normalize by sum of weights
                loss = loss_recon_all / sum(model.relative_importance)
                
                # Use first nesting level for metrics
                repr_sparse = repr_sparse[0]
                recons_sparse = recons_sparse[0]
            else:
                # Standard SAE loss computation
                loss, recon_loss, sparse_loss = loss_fn(recons_sparse, embeddings, repr_sparse)
            
            # Backpropagation
            loss.backward()

            # Zero gradients for frozen neurons (preserve their learned representations)
            # encoder shape: [n_inputs, n_latents] - columns are neurons
            # decoder shape: [n_latents, n_inputs] - rows are neurons
            if frozen_neurons:
                with torch.no_grad():
                    for neuron_idx in frozen_neurons:
                        if neuron_idx < model.encoder.shape[1]:
                            model.encoder.grad[:, neuron_idx] = 0
                            model.latent_bias.grad[neuron_idx] = 0
                            if model.decoder is not None:
                                model.decoder.grad[neuron_idx, :] = 0

            # Weight normalization and gradient projection
            model.scale_to_unit_norm()
            model.project_grads_decode()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad)
            
            # Update model parameters
            optimizer.step()
            
            # Detach tensors for metric calculation
            recons_sparse, recons_all, embeddings = recons_sparse.detach(), recons_all.detach(), embeddings.detach()
            
            # Calculate evaluation metrics
            # CKNNA (Centered Kernel Nearest Neighbor Alignment) scores
            cknna_score_sparse = cknna(recons_sparse, embeddings)
            cknna_score_all = cknna(recons_all, embeddings)
            
            # FVU (Explained Variance) metric
            fvu_score_sparse = explained_variance(embeddings, recons_sparse)
            fvu_score_all = explained_variance(embeddings, recons_all)
            
            # Reconstruction quality metrics
            diagonal_cs_sparse, mae_distance_sparse = calculate_similarity_metrics(recons_sparse, embeddings)
            diagonal_cs_all, mae_distance_all = calculate_similarity_metrics(recons_all, embeddings)
            
            # Orthogonality of decoder features
            od = orthogonal_decoder(model.decoder)
            
            # Sparsity measurements
            sparsity_sparse = (repr_sparse == 0.0).float().mean(axis=-1).mean()
            sparsity_all = (repr_all == 0.0).float().mean(axis=-1).mean()

            # Representation Metrics
            repr_norm = repr_all.norm(dim=-1).mean().item()
            repr_max = repr_all.max(dim=-1).values.mean().item()

            # Check for dead neurons periodically
            if global_step % cfg.training.check_dead == 0:
                activations = model.get_and_reset_stats()
                dead_neurons = identify_dead_neurons(activations).numpy().tolist()
                numb_of_dead_neurons = len(dead_neurons)
                logger.info(f"Number of dead neurons: {numb_of_dead_neurons}")
            
            # Log metrics periodically
            if global_step % cfg.training.print_freq == 0:
                logger.info(f"Epoch: {epoch+1}, Step: {step}, Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}, Sparse Loss: {sparse_loss.item():.6f}")
                logger.info(f"FVU Sparse: {fvu_score_sparse:.4f}, FVU All: {fvu_score_all:.4f}")
                logger.info(f"CKNNA Sparse: {cknna_score_sparse:.4f}, CKNNA All: {cknna_score_all:.4f}")
                logger.info(f"Cosine Similarity Sparse: {diagonal_cs_sparse:.4f}, MAE Distance Sparse: {mae_distance_sparse:.4f}")
                logger.info(f"Cosine Similarity All: {diagonal_cs_all:.4f}, MAE Distance All: {mae_distance_all:.4f}")
                logger.info(f"Sparsity Sparse: {sparsity_sparse:.4f}, Sparsity All: {sparsity_all:.4f}")
                logger.info(f"Orthogonal Decoder Loss: {od:.6f}, Representation norm {repr_norm:.4f} and max {repr_max:.2f}")
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        # Log epoch summary
        lr_rate = scheduler.get_last_lr()[0] if scheduler else cfg.training.lr
        logger.info(f"Epoch: {epoch+1}, Learning Rate: {lr_rate:.6f}, Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}, Sparse Loss: {sparse_loss.item():.6f}, Dead neurons: {numb_of_dead_neurons}")
    
        # Evaluate the model on the validation set
        eval(model, eval_loader, loss_fn, device, cfg)

        if args.dataset_second_modality is not None:
            # Evaluate on the second modality dataset
            eval(model, eval_loader_second, loss_fn, device, cfg)
    
    # Save the trained model
    # For Matryoshka models, append the first nesting level to activation name
    model_appendix = ""
    if args.model == "ReLUSAE":
        activation = f"{args.activation}_{str(cfg.loss.sparse_weight).split('.')[1]}"
    else:
        activation = cfg.model.activation
    
    if cfg.model.use_matryoshka:
        activation += f"_{model.nesting_list[0]}"
        if "RW" in args.model:
            model_appendix = "_RW"
        else:
            model_appendix = "_UW"
        
    # Construct filename with key hyperparameters
    model_params = f"{cfg.model.n_latents}_{cfg.model.n_inputs}_{activation}{model_appendix}_{cfg.model.tied}_{cfg.model.normalize}_{cfg.model.latent_soft_cap}"
    dataset_name = args.dataset_train.split("/")[-1].split(".")[0]
    save_path = f"{model_params}_{dataset_name}.pth"
    
    # Save model state and preprocessing parameters
    torch.save({
        "model": model.state_dict(),
        "mean_center": train_ds.mean,
        "scaling_factor": train_ds.scaling_factor,
        "target_norm": train_ds.target_norm,
    }, save_path)
    
    logger.info(f"Model saved to {save_path}")
        
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
