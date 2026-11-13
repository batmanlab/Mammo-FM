"""Losses shared by Stage 1 and Stage 2."""

import torch
import torch.nn.functional as F


def binary_kl_from_probs(student_logits, teacher_probs, epsilon=1e-7):
    """KL(Bernoulli(teacher) || Bernoulli(student)), averaged over all heads."""
    if student_logits.shape != teacher_probs.shape:
        raise ValueError(
            f"Student and teacher must have the same shape; got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_probs.shape)}"
        )
    student_probs = torch.sigmoid(student_logits.float()).clamp(epsilon, 1.0 - epsilon)
    teacher_probs = teacher_probs.float().clamp(epsilon, 1.0 - epsilon)
    loss = teacher_probs * torch.log(teacher_probs / student_probs)
    loss += (1.0 - teacher_probs) * torch.log(
        (1.0 - teacher_probs) / (1.0 - student_probs)
    )
    return loss.mean()


def masked_weighted_bce(student_logits, labels, label_mask, pos_weight):
    """Five-horizon BCE, excluding outcomes that are unknown because of censoring."""
    if student_logits.shape != labels.shape or labels.shape != label_mask.shape:
        raise ValueError(
            "Logits, labels, and masks must have identical [batch, 5] shapes; "
            f"got {tuple(student_logits.shape)}, {tuple(labels.shape)}, "
            f"and {tuple(label_mask.shape)}"
        )
    if student_logits.ndim != 2 or student_logits.shape[1] != 5:
        raise ValueError(f"Expected five risk heads, got {tuple(student_logits.shape)}")
    pos_weight = pos_weight.float().reshape(-1)
    if pos_weight.numel() != 5:
        raise ValueError(f"Expected five positive weights, got {pos_weight.numel()}")

    element_loss = F.binary_cross_entropy_with_logits(
        student_logits.float(),
        labels.float(),
        pos_weight=pos_weight,
        reduction="none",
    )
    mask = label_mask.float()
    return (element_loss * mask).sum() / mask.sum().clamp_min(1.0)
