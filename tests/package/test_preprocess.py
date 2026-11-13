"""Synthetic-only checks for standalone, CPU-only DICOM preprocessing."""

import hashlib
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.errors import InvalidDicomError
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

import mammofm as mf
from mammofm._preprocessing import preprocess_pixels


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pixels = np.random.default_rng(42).integers(0, 4096, size=(40, 30), dtype=np.uint16)

    def write_dicom(self, name='synthetic.dcm', photo='MONOCHROME2', *,
                    pixels=None, frames=1, samples=1):
        pixels = self.pixels if pixels is None else pixels
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
        data.PatientName = 'SYNTHETIC^PREPROCESS'
        data.PatientID = 'SYNTHETIC-PREPROCESS-IDENTIFIER'
        data.Rows, data.Columns = pixels.shape
        data.PhotometricInterpretation = photo
        data.SamplesPerPixel = samples
        data.BitsAllocated = 16
        data.BitsStored = 12
        data.HighBit = 11
        data.PixelRepresentation = 0
        data.NumberOfFrames = frames
        data.WindowCenter = 1800
        data.WindowWidth = 2200
        data.RescaleSlope = 1.2
        data.RescaleIntercept = -50
        if samples > 1:
            data.PlanarConfiguration = 0
            pixels = np.repeat(pixels[:, :, None], samples, axis=2)
        if frames > 1:
            pixels = np.repeat(pixels[None, ...], frames, axis=0)
        data.PixelData = pixels.tobytes()
        data.save_as(str(path), write_like_original=False)
        return path, data

    def assert_image_contract(self, image):
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, 'L')
        self.assertEqual(image.size, (912, 1520))
        self.assertEqual(np.asarray(image).dtype, np.dtype(np.uint8))
        self.assertEqual(image.info, {})
        self.assertFalse(image.getexif())
        self.assertFalse(getattr(image, 'filename', ''))
        self.assertFalse(hasattr(image, 'PatientID'))

    def test_public_export_and_exact_single_argument_signature(self):
        self.assertIn('preprocess', mf.__all__)
        signature = inspect.signature(mf.preprocess)
        self.assertEqual(tuple(signature.parameters), ('img',))
        with self.assertRaises(TypeError):
            mf.preprocess('unused', dataset='rsna')
        with self.assertRaises(TypeError):
            mf.preprocess('unused', device='cuda')

    def test_monochrome_routes_match_unified_pixels_exactly(self):
        for photo in ('MONOCHROME1', 'MONOCHROME2'):
            path, metadata = self.write_dicom(photo + '.dcm', photo)
            expected = preprocess_pixels(self.pixels, metadata)
            for source in (path, str(path)):
                with self.subTest(photo=photo, input_type=type(source).__name__):
                    image = mf.preprocess(source)
                    self.assert_image_contract(image)
                    np.testing.assert_array_equal(np.asarray(image), expected)

    def test_unified_default_is_not_dataset_windowing(self):
        path, metadata = self.write_dicom()
        image = mf.preprocess(path)
        np.testing.assert_array_equal(np.asarray(image), preprocess_pixels(self.pixels, metadata))
        self.assertFalse(np.array_equal(
            np.asarray(image), preprocess_pixels(self.pixels, metadata, intensity='window')))

    def test_load_image_receives_no_dataset_override(self):
        from mammofm._inputs import load_image
        path, _ = self.write_dicom()
        with patch('mammofm._inputs.load_image', wraps=load_image) as loader:
            image = mf.preprocess(path)
        loader.assert_called_once_with(path)
        self.assert_image_contract(image)

    def test_constant_images_are_valid_finite_grayscale(self):
        pixels = np.full(self.pixels.shape, 1200, dtype=np.uint16)
        for photo in ('MONOCHROME1', 'MONOCHROME2'):
            with self.subTest(photo=photo):
                path, metadata = self.write_dicom(photo + '.dcm', photo, pixels=pixels)
                image = mf.preprocess(path)
                self.assert_image_contract(image)
                np.testing.assert_array_equal(np.asarray(image), preprocess_pixels(pixels, metadata))
                self.assertTrue(np.isfinite(np.asarray(image)).all())
                self.assertFalse(np.asarray(image).any())

    def test_extensionless_dicom_is_supported(self):
        path, metadata = self.write_dicom('extensionless')
        image = mf.preprocess(path)
        self.assert_image_contract(image)
        np.testing.assert_array_equal(np.asarray(image), preprocess_pixels(self.pixels, metadata))

    def test_preprocessing_has_no_file_model_or_download_side_effects(self):
        path, _ = self.write_dicom()
        before_hash = hashlib.sha256(path.read_bytes()).digest()
        before_files = set(self.root.iterdir())
        unexpected = AssertionError('Preprocessing must not save files or construct/load models')
        with patch('PIL.Image.Image.save', side_effect=unexpected), \
                patch('tempfile.TemporaryDirectory', side_effect=unexpected), \
                patch('tempfile.NamedTemporaryFile', side_effect=unexpected), \
                patch('mammofm.api._client', side_effect=unexpected), \
                patch('mammofm.api.MammoFM', side_effect=unexpected), \
                patch('mammofm._assets.AssetStore.resolve', side_effect=unexpected):
            image = mf.preprocess(path)
        self.assert_image_contract(image)
        self.assertEqual(set(self.root.iterdir()), before_files)
        self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before_hash)

    def test_caller_can_save_metadata_free_png_without_pixel_changes(self):
        path, _ = self.write_dicom()
        image = mf.preprocess(path)
        destination = self.root / 'caller_saved.png'
        self.assertFalse(destination.exists())
        image.save(destination)
        with Image.open(destination) as saved:
            self.assertEqual(saved.format, 'PNG')
            self.assertEqual(saved.mode, 'L')
            self.assertEqual(saved.size, image.size)
            self.assertEqual(saved.info, {})
            np.testing.assert_array_equal(np.asarray(saved), np.asarray(image))

    def test_png_is_rejected_even_when_named_like_dicom(self):
        for name in ('already_preprocessed.png', 'disguised.dcm'):
            path = self.root / name
            Image.fromarray(self.pixels.astype(np.uint8)).save(path, format='PNG')
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'DICOM'):
                mf.preprocess(path)

    def test_unsupported_missing_and_invalid_inputs_are_rejected(self):
        for source in (None, 7, self.pixels, {'lmlo': 'unused.dcm'}, Image.new('L', (20, 20))):
            with self.subTest(input_type=type(source).__name__), self.assertRaises(TypeError):
                mf.preprocess(source)
        for source in (self.root / 'missing.dcm', self.root):
            with self.subTest(source_type='missing' if source != self.root else 'directory'):
                with self.assertRaises(FileNotFoundError):
                    mf.preprocess(source)
        jpeg = self.root / 'unsupported.jpg'
        Image.fromarray(self.pixels.astype(np.uint8)).save(jpeg)
        with self.assertRaises(ValueError):
            mf.preprocess(jpeg)
        malformed = self.root / 'malformed.dcm'
        malformed.write_bytes(b'synthetic invalid DICOM fixture')
        with self.assertRaises(InvalidDicomError):
            mf.preprocess(malformed)

    def test_color_and_multiframe_dicoms_are_rejected(self):
        for name, settings in (
            ('color.dcm', {'photo': 'RGB', 'samples': 3}),
            ('multiframe.dcm', {'frames': 2}),
            ('palette.dcm', {'photo': 'PALETTE COLOR'}),
        ):
            path, _ = self.write_dicom(name, **settings)
            with self.subTest(name=name), self.assertRaises(ValueError):
                mf.preprocess(path)

    def test_decoder_failure_preserves_actionable_error(self):
        path, _ = self.write_dicom()
        error = RuntimeError('synthetic decoder unavailable')
        class Undecodable:
            PhotometricInterpretation = 'MONOCHROME2'
            @property
            def pixel_array(self):
                raise error
        with patch('pydicom.dcmread', return_value=Undecodable()):
            with self.assertRaisesRegex(RuntimeError, r'pixel decoding failed.*mammofm\[dicom\]') as caught:
                mf.preprocess(path)
        self.assertIs(caught.exception.__cause__, error)

    def test_import_and_execution_do_not_import_model_or_gpu_libraries(self):
        path, _ = self.write_dicom()
        code = '''
import importlib.abc
from pathlib import Path
import sys
from unittest.mock import patch

heavy = {'torch', 'transformers', 'huggingface_hub', 'peft', 'accelerate', 'cupy'}
backends = {'mammofm._model', 'mammofm._risk', 'mammofm._reports', 'mammofm._report_model'}
class NoHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in heavy or fullname in backends:
            raise AssertionError('Unexpected model or GPU import: ' + fullname)
sys.meta_path.insert(0, NoHeavyImports())
import mammofm as mf
assert not (heavy & sys.modules.keys())
unexpected = AssertionError('Unexpected model construction or asset resolution')
with patch('mammofm.api._client', side_effect=unexpected), \\
        patch('mammofm.api.MammoFM', side_effect=unexpected), \\
        patch('mammofm._assets.AssetStore.resolve', side_effect=unexpected):
    image = mf.preprocess(Path(sys.argv[1]))
assert image.mode == 'L' and image.size == (912, 1520)
assert not image.info
assert not (heavy & sys.modules.keys())
assert not (backends & sys.modules.keys())
'''
        subprocess.run([sys.executable, '-c', code, str(path)], check=True,
                       capture_output=True, text=True, timeout=60)


if __name__ == '__main__':
    unittest.main()
