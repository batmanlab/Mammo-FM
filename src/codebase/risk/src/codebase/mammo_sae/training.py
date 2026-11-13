"""Portable Matryoshka training with training-only normalization statistics."""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .core import SpatialSAE
from .data import read_features
from .impl.sae import MatryoshkaAutoencoder


class Features(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return torch.tensor(self.values[index], dtype=torch.float32)


def statistics(values, mean_center=False, target_norm=None):
    dimension = values.shape[1]
    total = np.zeros(dimension, dtype=np.float64)
    squares = 0.0
    for start in range(0, len(values), 8192):
        chunk = np.asarray(values[start:start+8192], dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError("Training features contain nonfinite values")
        total += chunk.sum(0)
        squares += np.square(chunk).sum()
    mean = total / len(values) if mean_center else np.zeros(dimension)
    mean_square = max(0.0, squares / len(values) - np.square(mean).sum())
    target = np.sqrt(dimension) if target_norm is None else float(target_norm)
    if target < 0 or not np.isfinite(target):
        raise ValueError("target_norm must be finite and nonnegative")
    if target > 0 and mean_square <= 0:
        raise ValueError("Cannot normalize zero-variance features")
    scale = target / np.sqrt(mean_square) if target > 0 else 1.0
    return torch.tensor(mean, dtype=torch.float32), float(scale), float(target)


@torch.no_grad()
def evaluate(sae, loader):
    sums = np.zeros(len(sae.model.nesting_list), dtype=np.float64)
    count = 0
    sae.eval()
    for x in loader:
        x = x.to(sae.mean.device)
        if not torch.isfinite(x).all():
            raise ValueError("Validation features contain nonfinite values")
        levels, _, _ = sae.model.encode((x - sae.mean) * sae.scale)
        for i, z in enumerate(levels):
            sums[i] += ((sae.decode(z) - x) ** 2).sum().item()
        count += x.numel()
    return (sums / count).tolist()


def train(train_path, val_path, output, config, epochs=30, batch_size=4096,
          lr=1e-4, seed=12345, device="cpu", mean_center=False, target_norm=None):
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs, batch_size and lr must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    values, _ = read_features(train_path)
    validation, _ = read_features(val_path)
    if values.shape[1] != validation.shape[1] or values.shape[1] != config["n_inputs"]:
        raise ValueError("Training, validation and model input dimensions must match")
    if Path(train_path).resolve() == Path(val_path).resolve():
        raise ValueError("Training and validation must be different files")
    mean, scale, target = statistics(values, mean_center, target_norm)
    sae = SpatialSAE(MatryoshkaAutoencoder(**config), mean, scale).to(device)
    loader = DataLoader(Features(values), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(Features(validation), batch_size=batch_size)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=0.0,
                                 betas=(0.9, 0.999), eps=6.25e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    run = dict(model_config=config, epochs=epochs, batch_size=batch_size, lr=lr,
               seed=seed, mean_center=mean_center, target_norm=target,
               train_features=str(Path(train_path).resolve()),
               val_features=str(Path(val_path).resolve()),
               selection="mean validation MSE over nesting levels", torch_version=str(torch.__version__))
    (output / "run_config.json").write_text(json.dumps(run, indent=2))
    best = float("inf")
    with (output / "history.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "val_mse"] + ["val_mse_k%d" % k for k in sae.model.nesting_list])
        for epoch in range(epochs):
            sae.train()
            total, n = 0.0, 0
            for x in loader:
                x = x.to(device)
                normalized = (x - sae.mean) * sae.scale
                optimizer.zero_grad(set_to_none=True)
                levels, _, _ = sae.model.encode(normalized)
                reconstructions = sae.model.decode(levels)
                weights = sae.model.relative_importance
                loss = sum(w * (recon - normalized).square().mean()
                           for w, recon in zip(weights, reconstructions)) / sum(weights)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                optimizer.step()
                total += loss.item()*len(x)
                n += len(x)
            errors = evaluate(sae, val_loader)
            score = float(np.mean(errors))
            writer.writerow([epoch+1, total/n, score] + errors)
            stream.flush()
            checkpoint = dict(model=sae.model.state_dict(), model_config=config,
                              mean_center=sae.mean.detach().cpu(), scaling_factor=scale,
                              target_norm=target, epoch=epoch+1, val_mse=score, run_config=run)
            torch.save(checkpoint, output / "last.pt")
            if score < best:
                best = score
                torch.save(checkpoint, output / "best.pt")
            scheduler.step()
            print("epoch=%d train_loss=%.6g val_mse=%.6g" % (epoch+1, total/n, score))
    return sae.eval()
