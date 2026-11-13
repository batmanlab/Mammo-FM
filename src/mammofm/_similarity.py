def _embedding_rows(value, name):
    import sys

    import numpy as np

    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        if value.dtype == torch.bool or value.is_complex():
            raise TypeError(f"{name} must contain real numeric values, not boolean or complex values")
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(f"{name} must be a dense, non-quantized tensor")
        value = value.detach().to(device="cpu", dtype=torch.float64).resolve_conj().resolve_neg().numpy()

    def contains_boolean(item):
        if isinstance(item, (bool, np.bool_)):
            return True
        if isinstance(item, np.ndarray) and item.dtype.kind == "b":
            return True
        if isinstance(item, (list, tuple)):
            return any(contains_boolean(element) for element in item)
        return False

    if contains_boolean(value):
        raise TypeError(f"{name} must not contain boolean values")
    try:
        rows = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error
    if rows.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    if rows.ndim not in (1, 2):
        raise ValueError(f"{name} must have shape [D] or [N, D]")
    if rows.size == 0:
        raise ValueError(f"{name} must not be empty")
    rows = rows.astype(np.result_type(rows.dtype, np.float64), copy=False)
    if rows.ndim == 1:
        rows = rows[None, :]
    if not np.isfinite(rows).all():
        raise ValueError(f"{name} must contain only finite values")
    return rows


def _normalize_rows(rows, name):
    import numpy as np

    scale = np.max(np.abs(rows), axis=1, keepdims=True)
    if (scale == 0).any():
        raise ValueError(f"{name} must not contain a zero vector")
    scaled = rows / scale
    return scaled / np.sqrt(np.sum(scaled * scaled, axis=1, keepdims=True))


def zero_shot(image_embeddings, positive_embeddings, negative_embeddings):
    import numpy as np

    images = _embedding_rows(image_embeddings, "image_embeddings")
    positive = _embedding_rows(positive_embeddings, "positive_embeddings")
    negative = _embedding_rows(negative_embeddings, "negative_embeddings")
    if images.shape[1] != positive.shape[1] or images.shape[1] != negative.shape[1]:
        raise ValueError("Image, positive, and negative embeddings must have the same dimension")
    images = _normalize_rows(images, "image_embeddings")
    positive = _normalize_rows(positive, "positive_embeddings")
    negative = _normalize_rows(negative, "negative_embeddings")
    positive = _normalize_rows(positive.mean(axis=0, keepdims=True), "positive_embeddings mean")[0]
    negative = _normalize_rows(negative.mean(axis=0, keepdims=True), "negative_embeddings mean")[0]
    positive_cosine = [float(value) for value in np.clip(images @ positive, -1.0, 1.0)]
    negative_cosine = [float(value) for value in np.clip(images @ negative, -1.0, 1.0)]
    return {
        "positive_cosine": positive_cosine,
        "negative_cosine": negative_cosine,
        "score": [positive - negative for positive, negative in zip(positive_cosine, negative_cosine)],
        "metadata": {
            "scoring": "positive_cosine - negative_cosine",
        },
    }
