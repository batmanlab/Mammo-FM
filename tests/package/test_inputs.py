"""Synthetic-only input/preprocessing tests; no clinical images or identifiers."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, PngImagePlugin
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from mammofm._inputs import (
    DEFAULT_MEAN, DEFAULT_STD, inference_spec, inference_tensor,
    convert_dicom_to_png, load_image, normalize_dataset,
)
from mammofm._preprocessing import preprocess_pixels


class ImageInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(42)
        self.pixels = rng.integers(0, 4096, size=(40, 30), dtype=np.uint16)

    def tearDown(self):
        self.temporary.cleanup()

    def write_dicom(self, name='image.dcm', photo='MONOCHROME2', window=True, frames=1):
        path = self.root / name
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()
        data = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
        data.is_little_endian = True
        data.is_implicit_VR = False
        data.SOPClassUID = meta.MediaStorageSOPClassUID
        data.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        data.PatientName = 'SYNTHETIC^TEST'
        data.PatientID = 'SYNTHETIC-IDENTIFIER'
        data.Rows, data.Columns = self.pixels.shape
        data.PhotometricInterpretation = photo
        data.SamplesPerPixel = 1
        data.BitsAllocated = 16
        data.BitsStored = 12
        data.HighBit = 11
        data.PixelRepresentation = 0
        data.NumberOfFrames = frames
        data.PixelData = self.pixels.tobytes()
        if window:
            data.WindowCenter = 1800
            data.WindowWidth = 2200
            data.RescaleSlope = 1.2
            data.RescaleIntercept = -50
        data.save_as(str(path), write_like_original=False)
        return path, data

    def write_png(self, name='image.png', pixels=None):
        path = self.root / name
        Image.fromarray(self.pixels.astype(np.uint8) if pixels is None else pixels).save(path)
        return path

    def test_dataset_aliases_and_unknown_names(self):
        for name in ('NL-Breast', 'nl_breast', 'nlbreast', 'nlbs', ' NL Breast '):
            self.assertEqual(normalize_dataset(name), 'nlbreast')
        self.assertEqual(normalize_dataset('VinDr-Mammo'), 'vindr')
        self.assertEqual(normalize_dataset('Other-study'), 'otherstudy')
        self.assertEqual(normalize_dataset(None), 'general')
        with self.assertRaises(TypeError):
            normalize_dataset(3)

    def test_all_dicom_routes(self):
        path, dicom = self.write_dicom()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        for dataset in ('rsna', 'vindr', 'cmmd', 'lumina', 'another_study', None, 'NL-Breast'):
            with self.subTest(dataset=dataset):
                image, metadata = load_image(path, dataset)
                mode = 'window' if dataset == 'NL-Breast' else 'minmax'
                expected = preprocess_pixels(self.pixels, dicom, intensity=mode)
                self.assertEqual(image.mode, 'RGB')
                self.assertEqual(image.size, (912, 1520))
                np.testing.assert_array_equal(np.asarray(image)[:, :, 0], expected)
                self.assertEqual(metadata['intensity'], mode)
                self.assertNotIn('SYNTHETIC', json.dumps(metadata))
                self.assertEqual(image.info, {})
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_file_dicom_converter_writes_metadata_free_png_once(self):
        source, _ = self.write_dicom()
        destination = self.root / 'converted.png'
        details = convert_dicom_to_png(source, destination)
        self.assertEqual(destination.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        self.assertEqual(details['input_format'], 'dicom')
        self.assertNotIn('SYNTHETIC', json.dumps(details))
        converted, png_details = load_image(destination)
        direct, _ = load_image(source)
        np.testing.assert_array_equal(np.asarray(converted), np.asarray(direct))
        self.assertEqual(converted.info, {})
        self.assertEqual(png_details['input_format'], 'png')
        with self.assertRaises(FileExistsError):
            convert_dicom_to_png(source, destination)
        with self.assertRaisesRegex(ValueError, 'DICOM source'):
            convert_dicom_to_png(self.write_png(), self.root / 'not-dicom.png')
        with self.assertRaisesRegex(ValueError, r'\.png'):
            convert_dicom_to_png(source, self.root / 'wrong.jpg')

    def test_window_percentile_fallback(self):
        path, dicom = self.write_dicom(window=False)
        image, metadata = load_image(path, 'nl_breast')
        np.testing.assert_array_equal(np.asarray(image)[:, :, 0], preprocess_pixels(self.pixels, dicom, intensity='window'))
        self.assertEqual(metadata['window_source'], 'percentile_1_99')

    def test_window_metadata_selected(self):
        path, _ = self.write_dicom()
        _, metadata = load_image(path, 'nlbreast')
        self.assertEqual(metadata['window_source'], 'dicom')

    def test_monochrome1_route(self):
        path, dicom = self.write_dicom(photo='MONOCHROME1')
        for dataset, mode in [('rsna', 'minmax'), ('nlbreast', 'window')]:
            image, _ = load_image(path, dataset)
            np.testing.assert_array_equal(np.asarray(image)[:, :, 0], preprocess_pixels(self.pixels, dicom, intensity=mode))

    def test_extensionless_dicom(self):
        path, _ = self.write_dicom(name='extensionless')
        self.assertTrue(load_image(path, 'rsna')[1]['dicom_conversion'])

    def test_multiframe_and_color_rejected(self):
        path, _ = self.write_dicom(frames=2)
        with self.assertRaisesRegex(ValueError, 'single-frame'):
            load_image(path, 'rsna')
        path, _ = self.write_dicom(name='color.dcm', photo='RGB')
        with self.assertRaisesRegex(ValueError, 'MONOCHROME'):
            load_image(path, 'rsna')

    def test_decode_error_advises_both_codecs(self):
        path, _ = self.write_dicom()
        class Undecodable:
            PhotometricInterpretation = 'MONOCHROME2'

            @property
            def pixel_array(self):
                raise RuntimeError('decoder missing')

        with patch.object(pydicom, 'dcmread', return_value=Undecodable()):
            with self.assertRaisesRegex(RuntimeError, 'pylibjpeg-libjpeg.*pylibjpeg-openjpeg'):
                load_image(path)

    def test_png_bypasses_intensity_crop_resize(self):
        pixels = (self.pixels % 93 + 45).astype(np.uint8)
        path = self.write_png(pixels=pixels)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        for dataset in ('rsna', 'nlbreast', 'cmmd', None):
            image, metadata = load_image(path, dataset)
            self.assertEqual(image.size, (30, 40))
            np.testing.assert_array_equal(np.asarray(image)[:, :, 0], pixels)
            self.assertFalse(metadata['dicom_conversion'])
            self.assertFalse(metadata['breast_crop'])
            self.assertEqual(metadata['intensity'], 'already_preprocessed')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_png_text_metadata_removed(self):
        path = self.root / 'with_text.png'
        info = PngImagePlugin.PngInfo()
        info.add_text('PatientName', 'SYNTHETIC^TEST')
        Image.fromarray(self.pixels.astype(np.uint8)).save(path, pnginfo=info)
        image, metadata = load_image(path)
        self.assertEqual(image.info, {})
        self.assertNotIn('SYNTHETIC', json.dumps(metadata))

    def test_png_rgb_preserved(self):
        rgb = np.repeat(self.pixels[:, :, None].astype(np.uint8), 3, axis=2)
        rgb[:, :, 1] //= 2
        image, _ = load_image(self.write_png(pixels=rgb))
        np.testing.assert_array_equal(np.asarray(image), rgb)

    def test_png_16bit_alpha_and_jpeg_rejected(self):
        with self.assertRaisesRegex(ValueError, '16-bit'):
            load_image(self.write_png(name='16bit.png', pixels=self.pixels))
        with self.assertRaisesRegex(ValueError, 'alpha'):
            load_image(self.write_png(name='alpha.png', pixels=np.zeros((40, 30, 4), dtype=np.uint8)))
        path = self.root / 'image.jpg'
        Image.fromarray(self.pixels.astype(np.uint8)).save(path)
        with self.assertRaisesRegex(ValueError, 'DICOM'):
            load_image(path)

    def test_invalid_input_types_and_missing_path(self):
        with self.assertRaises(TypeError):
            load_image(self.pixels)
        with self.assertRaises(FileNotFoundError):
            load_image(self.root / 'absent.png')

    def test_inference_shapes_and_independent_normalization(self):
        image, _ = load_image(self.write_png())
        for branch, dataset, height, width in (
            ('clip', 'vindr', 1520, 912),
            ('classifier', 'nlbreast', 1520, 912),
            ('classifier', 'rsna', 1520, 912),
            ('classifier', 'cmmd', 1520, 912),
            ('clip', 'cmmd', 1520, 912),
            ('classifier', 'lumina', 1520, 912),
            ('clip', 'lumina', 1520, 912),
            ('clip', None, 1520, 912),
            ('features', 'unknown', 1520, 912),
        ):
            with self.subTest(branch=branch, dataset=dataset):
                tensor = inference_tensor(image, branch, dataset)
                self.assertEqual(tuple(tensor.shape), (1, 3, height, width))
                expected = cv2.resize(np.asarray(image), (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                expected -= expected.min()
                expected /= expected.max()
                expected = (expected - DEFAULT_MEAN) / DEFAULT_STD
                np.testing.assert_array_equal(tensor.numpy()[0], expected.transpose(2, 0, 1))

    def test_features_branch_retains_identity_and_portrait_shape(self):
        for dataset in ('rsna', 'NL-Breast', 'vindr', 'unknown', None):
            spec = inference_spec('features', dataset)
            self.assertEqual(spec['branch'], 'features')
            self.assertEqual((spec['height'], spec['width']), (1520, 912))

    def test_constant_images_are_finite(self):
        image, _ = load_image(self.write_png(pixels=np.full((15, 15), 50, dtype=np.uint8)))
        tensor = inference_tensor(image)
        self.assertTrue(np.isfinite(tensor.numpy()).all())
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), -DEFAULT_MEAN / DEFAULT_STD, places=6)

    def test_inference_arguments_validated(self):
        image, _ = load_image(self.write_png())
        for std in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                inference_tensor(image, std=std)
        with self.assertRaises(ValueError):
            inference_spec('unsupported')
        with self.assertRaises(TypeError):
            inference_tensor(image.convert('L'))


class PurePreprocessingTests(unittest.TestCase):
    def test_minmax_independent_formula(self):
        pixels = np.arange(225, dtype=np.uint16).reshape(15, 15)
        metadata = SimpleNamespace(PhotometricInterpretation='MONOCHROME2')
        output = preprocess_pixels(pixels, metadata, width=5, height=5, breast_crop=False)
        expected = pixels[5:-5, 5:-5].astype(np.float64)
        expected = ((expected - expected.min()) / (expected.max() - expected.min()) * 255).astype(np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_invalid_pixel_values_rejected(self):
        metadata = SimpleNamespace(PhotometricInterpretation='MONOCHROME2')
        for pixels in (np.full((20, 20), np.nan), np.zeros((20, 20), dtype=complex), np.zeros((5, 5))):
            with self.assertRaises(ValueError):
                preprocess_pixels(pixels, metadata)


if __name__ == '__main__':
    unittest.main()
