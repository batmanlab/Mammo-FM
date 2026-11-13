"""Local single-image loading and checkpoint-compatible inference transforms."""

import os
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image

from ._preprocessing import _first_number, preprocess_pixels

DEFAULT_MEAN = 0.3089279
DEFAULT_STD = 0.25053555408335154
KNOWN_DATASETS = frozenset({'rsna', 'vindr', 'cmmd', 'nlbreast', 'lumina'})


def normalize_dataset(name):
    """Normalize dataset aliases without concealing unknown dataset names.

    Unknown names remain distinct so diagnosis can reject them; feature
    extraction uses the general min-max route for any unknown name or None.
    """
    if name is None:
        return 'general'
    if not isinstance(name, str):
        raise TypeError('dataset must be a string or None')
    normalized = name.strip().lower().replace('-', '').replace('_', '').replace(' ', '')
    aliases = {'vindrmammo': 'vindr', 'nlbs': 'nlbreast',
               'nlbreastscreen': 'nlbreast', 'nlbreastscreening': 'nlbreast'}
    return aliases.get(normalized, normalized or 'general')


def load_image(image, dataset=None):
    """Load one local DICOM or preprocessed 8-bit PNG, returning RGB and metadata.

    DICOM conversion is in memory; no temporary PNG, patient metadata, input
    filename, or source path is returned. PNG contrast and crop are untouched.
    PNGs must already have the intended mammographic preprocessing. 16-bit
    PNGs are rejected to avoid silent truncation of diagnostic intensities.
    """
    if not isinstance(image, (str, os.PathLike)):
        raise TypeError('img must be a local path to a single DICOM or 8-bit PNG image')
    path = Path(image).expanduser()
    if not path.is_file():
        raise FileNotFoundError('img must point to an existing image file')
    dataset = normalize_dataset(dataset)
    with path.open('rb') as handle:
        is_png = handle.read(8) == b'\x89PNG\r\n\x1a\n'
    if is_png:
        with Image.open(path) as source:
            if source.format != 'PNG' or source.mode not in {'L', 'RGB', 'P', '1'}:
                raise ValueError('PNG input must be 8-bit grayscale or RGB (16-bit and alpha images are unsupported)')
            if 'transparency' in source.info:
                raise ValueError('Transparent PNG images are unsupported')
            original_size = [source.height, source.width]
            output = source.convert('RGB')
            output.info.clear()
        return output, {
            'input_format': 'png', 'dataset': dataset,
            'input_shape': original_size, 'output_shape': [output.height, output.width, 3],
            'intensity': 'already_preprocessed', 'border_removed': 0,
            'breast_crop': False, 'dicom_conversion': False,
        }
    if path.suffix.lower() not in {'.dcm', '.dicom', '.ima', ''}:
        raise ValueError('img must be a DICOM (.dcm, .dicom, .ima or extensionless) or PNG file')

    import pydicom

    dicom = pydicom.dcmread(str(path))
    if int(getattr(dicom, 'NumberOfFrames', 1)) != 1 or int(getattr(dicom, 'SamplesPerPixel', 1)) != 1:
        raise ValueError('Only single-frame grayscale DICOMs are supported')
    if str(getattr(dicom, 'PhotometricInterpretation', '')) not in {'MONOCHROME1', 'MONOCHROME2'}:
        raise ValueError('Only MONOCHROME1/2 grayscale DICOMs are supported')
    try:
        pixels = dicom.pixel_array
    except Exception as error:
        raise RuntimeError(
            'DICOM pixel decoding failed. Compressed DICOMs need a compatible '
            'pydicom decoder: pylibjpeg with pylibjpeg-libjpeg for JPEG/lossless '
            'JPEG, or pylibjpeg-openjpeg for JPEG2000. Install mammofm[dicom].'
        ) from error
    intensity = 'window' if dataset == 'nlbreast' else 'minmax'
    processed = preprocess_pixels(pixels, dicom, intensity=intensity)
    details = {
        'input_format': 'dicom', 'dataset': dataset,
        'input_shape': list(pixels.shape), 'output_shape': [1520, 912, 3],
        'intensity': intensity, 'border_removed': 5, 'breast_crop': True,
        'dicom_conversion': True, 'dicom_resize_interpolation': 'area',
    }
    if intensity == 'window':
        center = _first_number(getattr(dicom, 'WindowCenter', None))
        width = _first_number(getattr(dicom, 'WindowWidth', None))
        details['window_source'] = ('dicom' if center is not None and width is not None
                                    and width > 1e-6 else 'percentile_1_99')
    return Image.fromarray(processed).convert('RGB'), details


def convert_dicom_to_png(source, destination, dataset=None):
    """Convert one supported mammography DICOM into a metadata-free 8-bit PNG.

    This is the file-writing wrapper around the package's unified DICOM
    preprocessing route.  It deliberately rejects PNG input so callers cannot
    accidentally recompress an already-preprocessed image.  The returned
    metadata never contains DICOM tags or source/destination paths.
    """
    output = Path(destination).expanduser()
    if output.suffix.lower() != '.png':
        raise ValueError('destination must end in .png')
    if output.exists():
        raise FileExistsError('destination PNG already exists')
    image, details = load_image(source, dataset)
    if details['input_format'] != 'dicom':
        raise ValueError('convert_dicom_to_png expects a DICOM source')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        image.save(temporary, format='PNG', optimize=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(details)


def inference_spec(branch='clip', dataset=None):
    """Describe the branch-specific published model's input geometry."""
    if branch not in {'classifier', 'clip', 'features'}:
        raise ValueError("branch must be 'classifier', 'clip' or 'features'")
    height, width = 1520, 912
    return {
        'branch': branch, 'height': height, 'width': width,
        'resize_interpolation': 'linear', 'normalization': 'per_image_minmax_then_mean_std',
        'mean': DEFAULT_MEAN, 'std': DEFAULT_STD,
    }


def inference_tensor(image, branch='clip', dataset=None, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """Return a float32 CPU tensor [1, 3, height, width] without augmentation.

    ``image`` is the PIL RGB image returned by ``load_image``. All supported
    branches use 1520x912. Already matching images are not resized. Constant
    inputs map to zero before mean/std scaling.
    """
    if not isinstance(image, Image.Image) or image.mode != 'RGB':
        raise TypeError('inference_tensor expects the PIL RGB image returned by load_image')
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError('mean must be finite and std must be finite and positive')
    spec = inference_spec(branch, dataset)
    pixels = np.asarray(image)
    if pixels.shape[:2] != (spec['height'], spec['width']):
        pixels = cv2.resize(pixels, (spec['width'], spec['height']), interpolation=cv2.INTER_LINEAR)
    pixels = pixels.astype(np.float32)
    pixels -= pixels.min()
    maximum = pixels.max()
    if maximum > 0:
        pixels /= maximum
    pixels = (pixels - mean) / std

    import torch

    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).unsqueeze(0)
