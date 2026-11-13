"""Portable pure-pixel operations from Mammo-FM's unified DICOM converter.

These functions retain the CPU conventions in
``src/preprocessing/preprocess_images.py``. Keep pixel parity with that source
when changing either implementation. This is not anatomical segmentation or
the complete DICOM VOI LUT algorithm.
"""

import cv2
import numpy as np


def _first_number(value, default=None):
    if value is None:
        return default
    if not isinstance(value, (str, bytes)) and hasattr(value, '__len__'):
        value = value[0]
    result = float(value)
    if not np.isfinite(result):
        raise ValueError('Nonfinite DICOM intensity metadata')
    return result


def _legacy_indices(mask):
    """Keep the original boundary convention and all tied longest runs."""
    if not mask.any():
        return np.arange(len(mask))
    left = np.arange(len(mask))
    left[mask] = 0
    left = np.maximum.accumulate(left)
    right = np.arange(len(mask))
    right[mask[::-1]] = 0
    right = len(mask) - 1 - np.maximum.accumulate(right)[::-1]
    lengths = right - left - 1
    return np.flatnonzero(lengths == lengths.max())


def _central_slice(length):
    start, stop = length // 2 - int(length * 0.4), length // 2 + int(length * 0.4)
    return slice(start, stop) if stop > start else slice(None)


def extract_breast(image):
    """Legacy central-strip crop, not a lesion or anatomical segmentation mask."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError('Breast extraction requires a nonempty 2D image')
    detection = np.where(image <= 40, 0, image)
    cols = _legacy_indices(detection[_central_slice(image.shape[0])].std(axis=0) != 0)
    detection = detection[:, cols]
    rows = _legacy_indices(detection[:, _central_slice(len(cols))].std(axis=1) != 0)
    return image[rows][:, cols]


def preprocess_pixels(pixels, metadata, width=912, height=1520, border=5,
                      intensity='minmax', fallback='percentile', p_lo=0.01,
                      p_hi=0.99, breast_crop=True):
    """Return an 8-bit grayscale PNG array; dimensions are height by width.

    minmax: raw-pixel MONOCHROME1 inversion, then min-max scaling.
    window: legacy linear center +/- width/2 mapping after slope/intercept,
    or the selected fallback, followed by display-space inversion. This is
    intentionally not the complete DICOM VOI LUT/window-function algorithm.
    """
    if width < 1 or height < 1 or border < 0:
        raise ValueError('Dimensions must be positive and border nonnegative')
    if intensity not in ('minmax', 'window') or fallback not in ('minmax', 'percentile'):
        raise ValueError('Unsupported intensity mode or fallback')
    if not 0 <= p_lo < p_hi <= 1:
        raise ValueError('Require 0 <= p_lo < p_hi <= 1')
    photo = str(getattr(metadata, 'PhotometricInterpretation', ''))
    if photo not in ('MONOCHROME1', 'MONOCHROME2'):
        raise ValueError('Only MONOCHROME1/2 grayscale DICOMs are supported')
    if int(getattr(metadata, 'NumberOfFrames', 1)) != 1 or int(getattr(metadata, 'SamplesPerPixel', 1)) != 1:
        raise ValueError('Only single-frame grayscale DICOMs are supported')
    pixels = np.asarray(pixels)
    if pixels.ndim != 2 or min(pixels.shape) <= 2 * border:
        raise ValueError('Expected a 2D image larger than twice the border')
    if not np.issubdtype(pixels.dtype, np.number) or np.iscomplexobj(pixels):
        raise ValueError('Expected real numeric image pixels')
    if not np.isfinite(pixels).all():
        raise ValueError('Nonfinite image pixels')
    data = pixels.astype(np.float64 if intensity == 'minmax' else np.float32)
    if not np.isfinite(data).all():
        raise ValueError('Image pixels exceed the processing dtype range')
    if border:
        data = data[border:-border, border:-border]
    if intensity == 'minmax':
        if photo == 'MONOCHROME1':
            data = data.max() - data
        data -= data.min()
        maximum = data.max()
        data = data / maximum * 255.0 if maximum > 0 else np.zeros_like(data)
    else:
        data = data * _first_number(getattr(metadata, 'RescaleSlope', None), 1.0)
        data = data + _first_number(getattr(metadata, 'RescaleIntercept', None), 0.0)
        center = _first_number(getattr(metadata, 'WindowCenter', None))
        window = _first_number(getattr(metadata, 'WindowWidth', None))
        if center is not None and window is not None and window > 1e-6:
            low, high = center - window / 2.0, center + window / 2.0
        elif fallback == 'minmax':
            low, high = float(data.min()), float(data.max())
        else:
            low, high = (float(value) for value in np.quantile(data, [p_lo, p_hi]))
        data = (np.clip(data, low, high) - low) / (high - low + 1e-6) * 255.0
        if photo == 'MONOCHROME1':
            data = 255.0 - data
    if not np.isfinite(data).all():
        raise ValueError('Intensity processing produced nonfinite pixels')
    image = np.clip(data, 0, 255).astype(np.uint8)
    if breast_crop:
        image = extract_breast(image)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
