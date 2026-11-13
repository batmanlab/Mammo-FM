"""Standalone prognosis contract tests using synthetic inputs and tiny models only."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import torch

import mammofm as mf
from mammofm import api, exam
from mammofm._registry import BU_RISK_ENCODER, ENCODER, PROGNOSIS_ENCODER, RISK_CHECKPOINTS


class PrognosisTests(unittest.TestCase):
    def setUp(self):
        self.client = mf.MammoFM(device='cpu', local_files_only=True)
        self.model = object()
        self.paths = {'rcc': 'right_cc.png', 'LCC': 'left_cc.png',
                      'RMLO': 'right_mlo.png', 'LmLo': 'left_mlo.png'}
        self.images = torch.arange(48, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
        self.addCleanup(mf.clear_cache)

    @contextmanager
    def prepared(self, paths):
        yield self.images, {}, paths

    def test_public_export_is_lazy(self):
        code = (
            "import sys; import mammofm as mf; "
            "assert callable(mf.prognosis); assert 'prognosis' in mf.__all__; "
            "assert not ({'torch', 'transformers', 'huggingface_hub'} & sys.modules.keys())"
        )
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

    def test_five_json_probabilities_are_direct_uncalibrated_sigmoid_outputs(self):
        probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        with patch.object(exam, '_prepared_views', side_effect=self.prepared) as prepared, \
                patch.object(self.client, '_get_risk_model', return_value=self.model), \
                patch('mammofm._risk.predict_risks', return_value=probabilities) as predict:
            actual = self.client.prognosis(self.paths)
        self.assertEqual(tuple(actual), tuple(f'{year}_year' for year in range(1, 6)))
        for year, expected in enumerate(probabilities[0].tolist(), start=1):
            self.assertEqual(type(actual[f'{year}_year']), float)
            self.assertAlmostEqual(actual[f'{year}_year'], expected, places=7)
        self.assertEqual(json.loads(json.dumps(actual, allow_nan=False)), actual)
        self.assertEqual(tuple(prepared.call_args.args[0]), exam.VIEW_ORDER)
        self.assertEqual(tuple(prepared.call_args.args[0].values()),
                         ('left_mlo.png', 'left_cc.png', 'right_mlo.png', 'right_cc.png'))
        self.assertIs(predict.call_args.args[0], self.model)
        self.assertIs(predict.call_args.args[1], self.images)

    def test_invalid_view_inputs_fail_before_models_or_images(self):
        valid = {view: 'unused.png' for view in exam.VIEW_ORDER}
        invalid = ['single.png', Path('single.dcm'), ['same.png'] * 4, {},
                   {'lcc': 'only.png'}, {**valid, 'lcc': 'duplicate.png'},
                   {**valid, 'unknown': 'extra.png'}, {**valid, 3: 'bad.png'}]
        with patch.object(exam, '_prepared_views') as prepared, \
                patch.object(self.client.store, 'resolve') as resolve, \
                patch.object(self.client, '_get_risk_model') as model:
            for value in invalid:
                with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                    self.client.prognosis(value)
        prepared.assert_not_called()
        resolve.assert_not_called()
        model.assert_not_called()

    def test_missing_image_fails_before_asset_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = {view: Path(temporary) / f'{view}.png' for view in exam.VIEW_ORDER}
            with patch.object(self.client.store, 'resolve') as resolve:
                with self.assertRaises(FileNotFoundError):
                    self.client.prognosis(paths)
            resolve.assert_not_called()

    def test_only_requested_bu_model_and_batmanlab_encoder_are_resolved_and_cached(self):
        self.assertEqual(PROGNOSIS_ENCODER['filename'], 'Mammo-FM_BatmanlabTrained_CLIP.tar')
        self.assertEqual(ENCODER['filename'], PROGNOSIS_ENCODER['filename'])
        self.assertEqual(PROGNOSIS_ENCODER['sha256'], BU_RISK_ENCODER['sha256'])
        self.assertEqual(PROGNOSIS_ENCODER['size'], BU_RISK_ENCODER['size'])
        self.assertRegex(PROGNOSIS_ENCODER['revision'], r'^[0-9a-f]{40}$')
        expected = [RISK_CHECKPOINTS['bu']['s1s2'], PROGNOSIS_ENCODER]
        with patch.object(self.client.store, 'resolve', side_effect=lambda asset: Path(asset['filename'])) as resolve, \
                patch('mammofm._risk.load_risk_model', return_value=self.model) as loader:
            self.assertIsNone(self.client._risk_model)
            self.assertIs(self.client._get_risk_model(), self.model)
            self.assertIs(self.client._get_risk_model(), self.model)
            self.assertEqual([call.args[0] for call in resolve.call_args_list], expected)
            loader.assert_called_once_with(
                Path(expected[0]['filename']), device=torch.device('cpu'),
                encoder_checkpoint=Path(expected[1]['filename']))
            self.client.clear_cache()
            self.assertIsNone(self.client._risk_model)
            self.assertIs(self.client._get_risk_model(), self.model)
            self.assertEqual(loader.call_count, 2)

    def test_load_failure_is_not_cached_or_returned_as_null(self):
        with patch.object(self.client.store, 'resolve', return_value=Path('unused')), \
                patch('mammofm._risk.load_risk_model', side_effect=mf.CheckpointError('invalid weights')):
            with self.assertRaisesRegex(mf.CheckpointError, 'invalid weights'):
                self.client._get_risk_model()
        self.assertIsNone(self.client._risk_model)

    def test_nonfinite_or_wrong_length_probability_outputs_raise(self):
        cases = [(torch.tensor([[0., 0., 0., 0., float('nan')]]), RuntimeError),
                 (torch.zeros(1, 4), RuntimeError)]
        with patch.object(exam, '_prepared_views', side_effect=self.prepared):
            for values, error in cases:
                with self.subTest(shape=tuple(values.shape)), \
                        patch.object(self.client, '_get_risk_model', return_value=self.model), \
                        patch('mammofm._risk.predict_risks', return_value=values):
                    with self.assertRaises(error):
                        self.client.prognosis(self.paths)

    def test_predictor_failure_propagates(self):
        with patch.object(exam, '_prepared_views', side_effect=self.prepared), \
                patch.object(self.client, '_get_risk_model', return_value=self.model), \
                patch('mammofm._risk.predict_risks', side_effect=RuntimeError('inference failed')):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                self.client.prognosis(self.paths)

    def test_wrapper_forwards_device_offline_and_cache_options(self):
        expected = {'1_year': 0.1}
        client = Mock()
        client.prognosis.return_value = expected
        with patch.object(api, '_client', return_value=client) as factory:
            actual = mf.prognosis(self.paths, device='cuda:1', cache_dir='cache',
                                  model_dir='models', local_files_only=True)
        self.assertIs(actual, expected)
        factory.assert_called_once_with('cuda:1', 'cache', 'models', True)
        client.prognosis.assert_called_once_with(self.paths)

    def test_real_predictor_uses_model_device_dtype_and_training_sequence(self):
        class TinyRiskModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
                self.received = None

            def forward(self, images, time, view, side):
                self.received = (images, time, view, side, torch.is_grad_enabled())
                return images.mean(dim=(2, 3, 4)).sum(dim=1, keepdim=True).expand(-1, 5) * 0

        from mammofm._risk import SIDE_SEQUENCE, TIME_SEQUENCE, VIEW_SEQUENCE
        model = TinyRiskModel()
        with patch.object(exam, '_prepared_views', side_effect=self.prepared), \
                patch.object(self.client, '_get_risk_model', return_value=model):
            result = self.client.prognosis(self.paths)
        images, time, view, side, gradients_enabled = model.received
        self.assertEqual(images.dtype, torch.float64)
        self.assertEqual(images.device, model.anchor.device)
        self.assertFalse(gradients_enabled)
        for actual, expected in ((time, TIME_SEQUENCE), (view, VIEW_SEQUENCE), (side, SIDE_SEQUENCE)):
            self.assertEqual(actual.tolist(), [list(expected)])
            self.assertEqual(actual.device, model.anchor.device)
        self.assertEqual(result, {f'{year}_year': 0.5 for year in range(1, 6)})

    def test_mixed_png_dicom_uses_unified_converter_and_removes_temporary_png(self):
        from mammofm import _inputs
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for index, view in enumerate(exam.VIEW_ORDER):
                path = root / f'{view}.png'
                Image.fromarray(np.arange(120, dtype=np.uint8).reshape(12, 10) + index).save(path)
                paths[view] = path
            source = root / 'synthetic.dcm'
            metadata = FileMetaDataset()
            metadata.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
            metadata.MediaStorageSOPInstanceUID = generate_uid()
            metadata.TransferSyntaxUID = ExplicitVRLittleEndian
            metadata.ImplementationClassUID = generate_uid()
            dicom = FileDataset(str(source), {}, file_meta=metadata, preamble=b'\0' * 128)
            dicom.Rows, dicom.Columns = 40, 30
            dicom.PhotometricInterpretation = 'MONOCHROME2'
            dicom.SamplesPerPixel = 1
            dicom.BitsAllocated = dicom.BitsStored = 16
            dicom.HighBit = 15
            dicom.PixelRepresentation = 0
            dicom.PixelData = np.arange(1200, dtype=np.uint16).reshape(40, 30).tobytes()
            pydicom.dcmwrite(source, dicom, write_like_original=False)
            paths['LMLO'] = source
            hashes = {view: hashlib.sha256(path.read_bytes()).hexdigest() for view, path in paths.items()}
            image_sizes = []

            def transform(image, **kwargs):
                image_sizes.append((image.size, kwargs))
                return torch.full((1, 3, 2, 2), float(len(image_sizes)))

            with patch('mammofm._inputs.convert_dicom_to_png', wraps=_inputs.convert_dicom_to_png) as convert, \
                    patch('mammofm._inputs.inference_tensor', side_effect=transform), \
                    patch.object(self.client, '_get_risk_model', return_value=self.model), \
                    patch('mammofm._risk.predict_risks', return_value=torch.full((1, 5), 0.5)) as predict:
                output = self.client.prognosis({key.lower(): value for key, value in reversed(paths.items())})
            self.assertEqual(len(output), 5)
            convert.assert_called_once()
            self.assertEqual(convert.call_args.args[0], source)
            self.assertEqual(convert.call_args.kwargs, {'dataset': 'general'})
            converted = Path(convert.call_args.args[1])
            self.assertEqual(converted.name, 'LMLO.png')
            self.assertFalse(converted.parent.exists())
            self.assertEqual([size for size, _ in image_sizes], [(912, 1520), (10, 12), (10, 12), (10, 12)])
            self.assertTrue(all(options == {'branch': 'features', 'dataset': 'general'}
                                for _, options in image_sizes))
            images = predict.call_args.args[1]
            self.assertEqual(images[0, :, 0, 0, 0].tolist(), [1., 2., 3., 4.])
            self.assertEqual(hashes, {view: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for view, path in paths.items()})


if __name__ == '__main__':
    unittest.main()
