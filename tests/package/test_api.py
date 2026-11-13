"""Portable API contract checks using synthetic tensors, never clinical data."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
from PIL import Image
import torch

import mammofm as mf
from mammofm import api
from mammofm._registry import CLASSIFIERS, ENCODER, TEXT_ASSETS


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = mf.MammoFM(device='cpu')
        self.load = patch('mammofm._inputs.load_image', return_value=(
            Image.new('RGB', (12, 20)), {'input_format': 'png'})).start()
        self.tensor = patch('mammofm._inputs.inference_tensor', return_value=torch.zeros(1, 3, 20, 12)).start()
        self.addCleanup(patch.stopall)
        self.addCleanup(mf.clear_cache)

    def fake_models(self):
        clip = Mock()
        clip.encode_image_normalized.return_value = torch.tensor([[1., 0.]])
        self.client._get_clip = Mock(return_value=clip)
        self.client._prototype = Mock(return_value=torch.tensor([[0., 1.], [1., 0.]]))
        classifier = Mock()
        classifier.probabilities.return_value = torch.tensor([.25])
        self.density_classifier = Mock()
        self.density_classifier.probabilities.return_value = torch.tensor([[.1, .2, .3, .4]])
        self.client._get_classifier = Mock(side_effect=lambda dataset, task, method:
            self.density_classifier if task == 'density' else classifier)
        return clip, classifier

    def test_import_is_lazy(self):
        code = "import sys; import mammofm; assert not ({'torch','transformers','huggingface_hub'} & sys.modules.keys())"
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

    def test_availability_exact_supported_scope(self):
        result = mf.availability()
        self.assertEqual(set(result), {'rsna', 'vindr', 'cmmd', 'nlbreast', 'lumina'})
        entries = [item for dataset in result.values() for method in dataset.values() for item in method.values()]
        self.assertEqual(sum(item['available'] for item in entries), 33)
        self.assertEqual(len(entries), 36)
        self.assertEqual(len(CLASSIFIERS), 24)
        self.load.assert_not_called()

    def test_alias_and_unknown_datasets(self):
        self.assertEqual(mf.availability('NL-Breast'), mf.availability('nlbreast'))
        for dataset in ('unknown', 'cmmd_typo', None):
            with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                self.client.diagnosis('unused', dataset)
        self.load.assert_not_called()

    def test_lumina_cancer_all_methods(self):
        self.fake_models()
        result = self.client.diagnosis('unused', 'LUMINA', strict=True)
        self.assertEqual(result['dataset'], 'lumina')
        self.assertEqual(result['unavailable'], {})
        self.load.assert_called_once_with('unused', 'lumina')
        for method in api.METHODS:
            self.assertEqual(set(result[method]), {'cancer'})
            self.assertIsInstance(result[method]['cancer'], float)
        self.assertEqual([call.args for call in self.client._get_classifier.call_args_list],
                         [('lumina', 'cancer', method) for method in api.METHODS[1:]])
        for method in api.METHODS[1:]:
            asset = CLASSIFIERS[('lumina', 'cancer', method)]
            self.assertEqual(len(asset['revision']), 40)
            self.assertEqual(len(asset['sha256']), 64)
            self.assertEqual(result['metadata']['checkpoints'][method]['cancer']['revision'],
                             asset['revision'])
        json.dumps(result, allow_nan=False)

    def test_strict_missing_rejects_before_inference(self):
        self.fake_models()
        with self.assertRaises(mf.MissingModelError):
            self.client.diagnosis('unused', 'rsna', strict=True)
        self.load.assert_not_called()
        self.client._get_clip.assert_not_called()

    def test_density_zero_shot_requires_validated_release_not_just_prompts(self):
        with patch('mammofm.api._prompts', return_value={'density': {}}):
            for dataset in ('rsna', 'vindr'):
                entry = mf.availability(dataset)[dataset]['zero_shot']['density']
                self.assertFalse(entry['available'])
                self.assertIn('No validated packaged zero-shot density endpoint', entry['reason'])
        self.load.assert_not_called()

    def test_rsna_invasive_is_available_only_as_conditional_classifier(self):
        self.fake_models()
        with self.assertWarns(UserWarning):
            result = self.client.diagnosis('unused', 'rsna')
        self.assertIsNone(result['zero_shot']['invasive_cancer'])
        self.assertIsNone(result['zero_shot']['density'])
        self.assertEqual(set(result['unavailable']), {'zero_shot'})
        self.assertEqual(set(result['unavailable']['zero_shot']), {'invasive_cancer', 'density'})
        for method in api.METHODS:
            self.assertIsInstance(result[method]['cancer'], float)
        for method in api.METHODS[1:]:
            self.assertIsInstance(result[method]['invasive_cancer'], float)
            self.assertEqual(list(result[method]['density']), list('ABCD'))
            self.assertAlmostEqual(sum(result[method]['density'].values()), 1.)
        definition = result['metadata']['task_definitions']['invasive_cancer'].lower()
        self.assertIn('cancer-positive', definition)
        self.assertIn('not', definition)
        self.assertIn('population', definition)
        self.assertEqual([call.args[0] for call in self.client._prototype.call_args_list], ['cancer'])
        json.dumps(result, allow_nan=False)

    def test_vindr_schema_scoring_and_selection(self):
        self.fake_models()
        with self.assertWarns(UserWarning):
            result = self.client.diagnosis('unused', 'vindr')
        self.assertEqual(set(result['unavailable']), {'zero_shot'})
        self.assertEqual(set(result['unavailable']['zero_shot']), {'density'})
        for method in api.METHODS:
            self.assertEqual(set(result[method]), {'abnormality', 'mass', 'suspicious_calcification', 'density'})
        self.assertAlmostEqual(result['zero_shot']['mass'], float(torch.tensor([0., 1.]).softmax(0)[1]))
        calls = [call.args for call in self.client._get_classifier.call_args_list]
        self.assertEqual(calls, [( 'vindr', task, method) for method in api.METHODS[1:]
                                for task in ('abnormality', 'mass', 'suspicious_calcification', 'density')])
        self.assertIn('BI-RADS', result['metadata']['abnormality_definition'])
        self.assertFalse(result['metadata']['probabilities_calibrated'])
        json.dumps(result, allow_nan=False)

    def test_cmmd_all_binary_methods_and_canonical_calcification_prompts(self):
        self.fake_models()
        result = self.client.diagnosis('unused', 'CMMD', strict=True)
        self.assertEqual(result['dataset'], 'cmmd')
        self.assertEqual(result['unavailable'], {})
        self.load.assert_called_once_with('unused', 'cmmd')
        for method in api.METHODS:
            self.assertEqual(set(result[method]), {'cancer', 'mass', 'calcification'})
            self.assertTrue(all(isinstance(value, float) for value in result[method].values()))
        self.assertEqual(api._prompt_task('calcification'), 'suspicious_calcification')
        prototype = torch.ones(2, 4)
        self.client._prototypes['suspicious_calcification'] = prototype
        self.assertIs(api.MammoFM._prototype(self.client, 'calcification'), prototype)
        self.assertEqual(set(self.client._prototypes), {'suspicious_calcification'})
        json.dumps(result, allow_nan=False)

    def test_density_probability_contract_and_metadata(self):
        self.fake_models()
        result = self.client.diagnosis('unused', 'rsna', methods='linear_probe', strict=True)
        self.assertEqual(result['unavailable'], {})
        self.assertEqual(list(result['linear_probe']['density']), list('ABCD'))
        for label, value in zip('ABCD', (.1, .2, .3, .4)):
            self.assertIsInstance(result['linear_probe']['density'][label], float)
            self.assertAlmostEqual(result['linear_probe']['density'][label], value)
        self.assertEqual(result['metadata']['density_class_order'], list('ABCD'))
        self.assertIn('softmax', result['metadata']['density_scoring'])
        self.client._get_clip.assert_not_called()
        self.client._prototype.assert_not_called()
        json.dumps(result, allow_nan=False)

    def test_density_rejects_wrong_shape_nonfinite_range_and_nonunit_sum(self):
        self.fake_models()
        bad_values = (
            [.1, .2, .3, .4], [[.1], [.2], [.3], [.4]], [[1.]],
            [[.1, .2, .3, .4, .5]], [[.1, .2, .3, .4], [.1, .2, .3, .4]],
            [[float('nan'), .2, .3, .4]], [[float('inf'), .2, .3, .4]],
            [[-.1, .2, .3, .6]], [[1.1, 0., 0., 0.]], [[.1, .1, .1, .1]],
        )
        for values in bad_values:
            self.density_classifier.probabilities.return_value = torch.tensor(values)
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                self.client.diagnosis('unused', 'rsna', methods='linear_probe', strict=True)

    def test_selected_method_skips_clip(self):
        self.fake_models()
        result = self.client.diagnosis('unused', 'nlbreast', methods='full_finetune')
        self.assertNotIn('zero_shot', result)
        self.assertNotIn('linear_probe', result)
        self.client._get_clip.assert_not_called()
        self.client._get_classifier.assert_called_once_with('nlbreast', 'cancer', 'full_finetune')

    def test_invalid_methods(self):
        for methods in ([], ['x'], ['zero_shot', 'zero_shot']):
            with self.subTest(methods=methods), self.assertRaises(ValueError):
                self.client.diagnosis('unused', 'vindr', methods=methods)
        self.load.assert_not_called()

    def test_failures_not_hidden_as_null(self):
        self.fake_models()
        self.client._get_classifier.side_effect = mf.CheckpointError('test failure')
        with self.assertRaises(mf.CheckpointError):
            self.client.diagnosis('unused', 'nlbreast', methods='linear_probe')

    def test_bad_probability_rejected(self):
        _, classifier = self.fake_models()
        for values in ([float('nan')], [float('inf')], [-.01], [1.01], [.1, .2]):
            classifier.probabilities.return_value = torch.tensor(values)
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                self.client.diagnosis('unused', 'nlbreast', methods='linear_probe')

    def test_features_are_complete_and_unprojected(self):
        spatial = torch.arange(48, dtype=torch.float32).reshape(1, 4, 3, 4)
        model = Mock()
        model.features.return_value = {'before_pooling': spatial, 'after_pooling': spatial.mean((2, 3))}
        self.client._get_encoder = Mock(return_value=model)
        result = self.client.features('unused', 'new-dataset')
        np.testing.assert_array_equal(result['before_pooling'], spatial.numpy())
        np.testing.assert_array_equal(result['after_pooling'], spatial.mean((2, 3)).numpy())
        self.assertEqual(result['metadata']['dataset'], 'newdataset')
        self.assertEqual(result['metadata']['inference']['branch'], 'features')
        self.assertEqual(result['metadata']['checkpoint']['filename'], ENCODER['filename'])

    def test_nonfinite_features_rejected(self):
        model = Mock()
        model.features.return_value = {'before_pooling': torch.tensor([float('nan')]), 'after_pooling': torch.ones(1)}
        self.client._get_encoder = Mock(return_value=model)
        with self.assertRaises(RuntimeError):
            self.client.features('unused')

    def test_prototype_mean_then_normalize_and_cache(self):
        model = Mock(text_max_length=2048)
        # Negative mean [0.5,0.5], positive mean [1,0]; test class order.
        model.encode_text_normalized.side_effect = [torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([[1., 0.]])]
        self.client._get_clip = Mock(return_value=model)
        self.client._tokenizer = Mock(return_value={'input_ids': torch.ones(1, 2, dtype=torch.long)})
        with patch('mammofm.api._prompts', return_value={'test': {'negative': ['a', 'b'], 'positive': ['c']}}):
            first = self.client._prototype('test')
            second = self.client._prototype('test')
        torch.testing.assert_close(first, torch.tensor([[2**-.5, 2**-.5], [1., 0.]]))
        self.assertIs(first, second)
        self.assertEqual(model.encode_text_normalized.call_count, 2)
        self.assertEqual(self.client._tokenizer.call_args.kwargs['max_length'], 2048)

    def test_clip_colocates_partial_local_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for index, asset in enumerate(TEXT_ASSETS.values()):
                directory = root / str(index)
                directory.mkdir()
                path = directory / Path(asset['filename']).name
                path.write_text('{}')
                paths[asset['filename']] = path
            self.client.store.resolve = Mock(side_effect=lambda asset: paths.get(asset['filename'], root / 'weights.tar'))
            def verify_directory(directory, **kwargs):
                self.assertEqual({p.name for p in Path(directory).iterdir()}, set(TEXT_ASSETS))
                self.assertTrue(kwargs['local_files_only'])
                self.assertFalse(kwargs['trust_remote_code'])
                return Mock()
            with patch('transformers.AutoTokenizer.from_pretrained', side_effect=verify_directory), \
                    patch('mammofm._model.load_clip', return_value=Mock()) as loader:
                self.client._get_clip()
            self.assertFalse(loader.call_args.kwargs['text_config_dir'].exists())

    def test_classifier_cache_is_bounded_and_task_specific(self):
        self.client.store.resolve = Mock(side_effect=lambda asset: Path(asset['filename']))
        with patch('mammofm._model.load_classifier', side_effect=lambda path, **kwargs: Mock(path=path)) as loader:
            one = self.client._get_classifier('vindr', 'mass', 'linear_probe')
            self.assertIs(one, self.client._get_classifier('vindr', 'mass', 'linear_probe'))
            self.client._get_classifier('vindr', 'mass', 'full_finetune')
            self.client._get_classifier('nlbreast', 'cancer', 'linear_probe')
        self.assertEqual(loader.call_count, 3)
        self.assertEqual(len(self.client._classifiers), 2)
        self.assertNotIn(('vindr', 'mass', 'linear_probe'), self.client._classifiers)

    def test_classifier_loader_receives_task_specific_output_count(self):
        self.client.store.resolve = Mock(side_effect=lambda asset: Path(asset['filename']))
        with patch('mammofm._model.load_classifier', return_value=Mock()) as loader:
            self.client._get_classifier('rsna', 'density', 'linear_probe')
            self.assertEqual(loader.call_args.kwargs['num_classes'], 4)
            self.client._get_classifier('rsna', 'invasive_cancer', 'linear_probe')
            self.assertEqual(loader.call_args.kwargs['num_classes'], 1)

    def test_bad_registered_output_count_is_rejected_before_download(self):
        key = ('rsna', 'density', 'linear_probe')
        self.client.store.resolve = Mock()
        with patch.dict(CLASSIFIERS, {key: {**CLASSIFIERS[key], 'num_classes': 1}}):
            with self.assertRaisesRegex(ValueError, 'output count'):
                self.client._get_classifier(*key)
        self.client.store.resolve.assert_not_called()

    def test_prompt_batch_size_validation(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mf.MammoFM(prompt_batch_size=value)

    def test_clear_releases_references(self):
        self.client._encoder = self.client._clip = self.client._tokenizer = Mock()
        self.client._prototypes['x'] = Mock()
        self.client.clear_cache()
        self.assertIsNone(self.client._encoder)
        self.assertIsNone(self.client._clip)
        self.assertEqual(self.client._prototypes, {})


if __name__ == '__main__':
    unittest.main()
