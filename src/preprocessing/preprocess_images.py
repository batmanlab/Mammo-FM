#!/usr/bin/env python3
"""Dataset-independent, CPU-only mammography DICOM-to-PNG conversion.

Uses the existing NumPy breast-extraction and intensity conventions. It does
not generate labels, split folds, transform detection boxes, or anonymize data.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
import fnmatch
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import pydicom


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


def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def plan_inputs(input_root, output_root, csv_path=None, path_column='image_path', patterns=None):
    """Plan unique, collision-free outputs before creating any image files."""
    if not input_root.is_dir():
        raise ValueError('Input root must be an existing directory')
    if output_root.exists() and not output_root.is_dir():
        raise ValueError('Output root must be a directory')
    if _inside(input_root, output_root):
        raise ValueError('Output root must not equal or contain the input root')
    sources = []
    if csv_path:
        with csv_path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            if path_column not in (reader.fieldnames or []):
                raise ValueError('CSV is missing the requested path column')
            for number, row in enumerate(reader, start=2):
                value = (row.get(path_column) or '').strip()
                if not value:
                    raise ValueError('Empty CSV path at row {}'.format(number))
                path = Path(value)
                sources.append(path if path.is_absolute() else input_root / path)
    else:
        patterns = patterns or ['*.dcm', '*.dicom']
        for directory, dirs, files in os.walk(str(input_root), followlinks=False):
            dirs[:] = sorted(d for d in dirs if not _inside((Path(directory) / d).resolve(), output_root))
            for name in sorted(files):
                path = Path(directory) / name
                relative = path.relative_to(input_root).as_posix()
                if any(fnmatch.fnmatchcase(relative.lower() if '/' in p else name.lower(), p.lower()) for p in patterns):
                    sources.append(path)
    jobs, seen, destinations = [], set(), set()
    for source in sources:
        source = source.resolve()
        if not _inside(source, input_root) or _inside(source, output_root):
            raise ValueError('An input path escapes the input root or lies in the output tree')
        if source in seen:
            continue
        destination = output_root / source.relative_to(input_root).with_suffix('.png')
        if destination.is_symlink():
            raise ValueError('PNG destinations must not be symbolic links')
        destination = destination.resolve()
        if not _inside(destination, output_root):
            raise ValueError('An output symlink escapes the output root')
        if destination in destinations:
            raise ValueError('Multiple inputs map to the same PNG; use distinct source stems')
        seen.add(source)
        destinations.add(destination)
        jobs.append((source, destination))
    if not jobs:
        raise ValueError('No input files selected')
    return jobs


def _save_png(destination, image, overwrite):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.preprocess-', suffix='.png', dir=str(destination.parent))
    os.close(fd)
    try:
        if not cv2.imwrite(temporary, image):
            raise OSError('PNG encoding failed')
        if overwrite:
            os.replace(temporary, str(destination))
        else:
            os.link(temporary, str(destination))  # Atomic no-clobber publication.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _convert_job(job):
    source, destination, options, existing = job
    cv2.setNumThreads(1)
    try:
        if destination.exists():
            if not destination.is_file():
                raise ValueError('PNG destination exists but is not a regular file')
            if existing == 'skip':
                return 'skipped', ''
            if existing == 'error':
                raise FileExistsError('PNG exists; choose --existing skip or overwrite explicitly')
        dataset = pydicom.dcmread(str(source))
        if int(getattr(dataset, 'NumberOfFrames', 1)) != 1 or int(getattr(dataset, 'SamplesPerPixel', 1)) != 1:
            raise ValueError('Only single-frame grayscale DICOMs are supported')
        if str(getattr(dataset, 'PhotometricInterpretation', '')) not in ('MONOCHROME1', 'MONOCHROME2'):
            raise ValueError('Only MONOCHROME1/2 grayscale DICOMs are supported')
        try:
            pixels = dataset.pixel_array
        except Exception as error:
            raise RuntimeError('DICOM pixel decoding failed ({}). Compressed DICOMs may require a compatible pydicom decoder plugin, e.g. pylibjpeg with pylibjpeg-openjpeg.'.format(type(error).__name__)) from error
        image = preprocess_pixels(pixels, dataset, **options)
        _save_png(destination, image, overwrite=existing == 'overwrite')
        return 'written', ''
    except Exception as error:
        return 'failed', '{}: {}'.format(type(error).__name__, error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--csv', type=Path, help='Optional file selection; no labels or fold columns needed')
    parser.add_argument('--path-column', default='image_path')
    parser.add_argument('--pattern', nargs='+', help='Folder scan patterns (default: *.dcm *.dicom, case-insensitive)')
    parser.add_argument('--height', type=int, default=1520)
    parser.add_argument('--width', type=int, default=912)
    parser.add_argument('--border', type=int, default=5)
    parser.add_argument('--intensity', choices=('minmax', 'window'), default='minmax')
    parser.add_argument('--fallback', choices=('minmax', 'percentile'), default='percentile', help='Fallback when window metadata is absent')
    parser.add_argument('--p-lo', type=float, default=0.01)
    parser.add_argument('--p-hi', type=float, default=0.99)
    parser.add_argument('--no-breast-crop', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--existing', choices=('error', 'skip', 'overwrite'), default='error')
    parser.add_argument('--report', type=Path, help='New CSV path; default: a timestamped report under output root')
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.workers) < 1 or args.border < 0:
        parser.error('width, height and workers must be positive; border must be nonnegative')
    if not 0 <= args.p_lo < args.p_hi <= 1:
        parser.error('Require 0 <= p-lo < p-hi <= 1')
    input_root, output_root = args.input_root.resolve(), args.output_root.resolve()
    try:
        jobs = plan_inputs(input_root, output_root, args.csv, args.path_column, args.pattern)
        report = (args.report or output_root / ('preprocessing_report_' + datetime.now().strftime('%Y%m%dT%H%M%S_%f') + '.csv')).resolve()
        if report.exists() or any(report == p for pair in jobs for p in pair):
            raise ValueError('Report must be a new file, distinct from all inputs and PNG outputs')
        if _inside(report, input_root) and not _inside(report, output_root):
            raise ValueError('Report must not be written into the source image tree')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    options = dict(width=args.width, height=args.height, border=args.border,
                   intensity=args.intensity, fallback=args.fallback, p_lo=args.p_lo,
                   p_hi=args.p_hi, breast_crop=not args.no_breast_crop)
    os.umask(0o077)
    report.parent.mkdir(parents=True, exist_ok=True)
    counts = dict(written=0, skipped=0, failed=0)
    work = [(source, destination, options, args.existing) for source, destination in jobs]
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        with report.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['input_path', 'output_path', 'status', 'error'])
            results = executor.map(_convert_job, work, chunksize=1) if executor else map(_convert_job, work)
            for number, ((source, destination), (status, error)) in enumerate(zip(jobs, results), start=1):
                counts[status] += 1
                writer.writerow([source.relative_to(input_root).as_posix(), destination.relative_to(output_root).as_posix(), status, error])
                handle.flush()
                if number % 100 == 0 or number == len(jobs):
                    print('Processed {}/{}: {}'.format(number, len(jobs), counts), flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True)
    print('Report: {}'.format(report), flush=True)
    return 1 if counts['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
