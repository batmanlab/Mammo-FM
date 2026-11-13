import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

import mammofm as mf
from mammofm import api
from mammofm._registry import ENCODER


class CustomPromptTests(unittest.TestCase):
    def setUp(self):
        self.client = mf.MammoFM(device='cpu', local_files_only=True, prompt_batch_size=2)
        self.model = Mock(text_max_length=64)
        self.client._get_clip = Mock(return_value=self.model)
        self.client._tokenizer = Mock(side_effect=lambda batch, **kwargs: {
            'input_ids': torch.ones(len(batch), 3, dtype=torch.long),
            'attention_mask': torch.ones(len(batch), 3, dtype=torch.long),
        })
        self.addCleanup(mf.clear_cache)

    def test_public_exports_stay_lazy(self):
        code = (
            "import sys; import mammofm as mf; "
            "assert all(callable(getattr(mf, n)) and n in mf.__all__ "
            "for n in ['text_embeddings','image_embeddings','zero_shot']); "
            "assert not ({'torch','transformers','huggingface_hub','numpy'} & sys.modules.keys())"
        )
        subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)

    def test_text_embeddings_batch_order_dtype_and_normalization(self):
        prompts = ['first prompt', 'second prompt', 'third prompt']
        self.model.encode_text_normalized.side_effect = [
            torch.tensor([[3., 4.], [0., 2.]], dtype=torch.float64, requires_grad=True),
            torch.tensor([[5., 0.]], dtype=torch.float64, requires_grad=True),
        ]
        actual = self.client.text_embeddings(prompts)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, [[.6, .8], [0., 1.], [1., 0.]])
        self.assertEqual([call.args[0] for call in self.client._tokenizer.call_args_list],
                         [prompts[:2], prompts[2:]])
        self.assertEqual(prompts, ['first prompt', 'second prompt', 'third prompt'])
        for call in self.client._tokenizer.call_args_list:
            self.assertEqual(call.kwargs, {'padding': 'longest', 'truncation': True,
                                          'return_tensors': 'pt', 'max_length': 64})
        for call in self.model.encode_text_normalized.call_args_list:
            self.assertTrue(all(value.device.type == 'cpu' for value in call.args[0].values()))

    def test_text_encoding_disables_gradients(self):
        def encode(tokens):
            self.assertFalse(torch.is_grad_enabled())
            return torch.ones(tokens['input_ids'].shape[0], 2)
        self.model.encode_text_normalized.side_effect = encode
        for prompts in ('one prompt', ('another prompt',)):
            self.assertEqual(self.client.text_embeddings(prompts).shape, (1, 2))

    def test_invalid_prompts_fail_before_model_loading(self):
        for prompts in ([], (), '', '   ', ['ok', ''], ['ok', 5], None,
                        b'bytes', {'positive': 'text'}, (item for item in ['text'])):
            with self.subTest(prompts=type(prompts)), self.assertRaises((TypeError, ValueError)):
                self.client.text_embeddings(prompts)
        self.client._get_clip.assert_not_called()
        self.client._tokenizer.assert_not_called()

    def test_invalid_model_embeddings_raise(self):
        for value in (torch.zeros(1, 2), torch.tensor([[float('nan'), 1.]]),
                      torch.tensor([[float('inf'), 1.]]), torch.ones(2, 2),
                      torch.ones(2), torch.ones(1, 0), torch.ones(1, 2, 1),
                      torch.ones(1, 2, dtype=torch.bool),
                      torch.ones(1, 2, dtype=torch.complex64), [[1., 2.]]):
            self.model.encode_text_normalized.return_value = value
            with self.subTest(value=type(value)), self.assertRaises(RuntimeError):
                self.client.text_embeddings(['text'])

    def test_inconsistent_text_batch_dimensions_raise(self):
        self.model.encode_text_normalized.side_effect = [torch.ones(2, 2), torch.ones(1, 3)]
        with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
            self.client.text_embeddings(['a', 'b', 'c'])

    def test_image_embedding_uses_clip_preprocessing_and_projection(self):
        image = Image.new('RGB', (12, 20))
        tensor = torch.zeros(1, 3, 20, 12)
        def encode(values):
            self.assertFalse(torch.is_grad_enabled())
            self.assertIs(values, tensor)
            return torch.tensor([[3., 4.]], requires_grad=True)
        self.model.encode_image_normalized.side_effect = encode
        with patch('mammofm._inputs.load_image', return_value=(image, {})) as load, \
                patch('mammofm._inputs.inference_tensor', return_value=tensor) as transform:
            actual = self.client.image_embeddings('example.dcm', 'NL-Breast')
        load.assert_called_once_with('example.dcm', 'nlbreast')
        transform.assert_called_once_with(image, branch='clip', dataset='nlbreast')
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, [[.6, .8]])
        self.model.features.assert_not_called()

    def test_image_embedding_unknown_dataset_uses_existing_route(self):
        self.model.encode_image_normalized.return_value = torch.ones(1, 2)
        image = Image.new('RGB', (12, 20))
        with patch('mammofm._inputs.load_image', return_value=(image, {})) as load, \
                patch('mammofm._inputs.inference_tensor', return_value=torch.zeros(1, 3, 20, 12)):
            self.client.image_embeddings('example.png', 'my_dataset')
            self.client.image_embeddings('example.png')
        self.assertEqual([call.args[1] for call in load.call_args_list], ['mydataset', 'general'])

    def test_bad_image_fails_before_model_loading(self):
        with patch('mammofm._inputs.load_image', side_effect=FileNotFoundError('missing')):
            with self.assertRaises(FileNotFoundError):
                self.client.image_embeddings('missing.png')
        self.client._get_clip.assert_not_called()

    def test_embedding_wrappers_forward_runtime_options(self):
        client = Mock()
        options = dict(device='cuda:1', cache_dir='cache', model_dir='models', local_files_only=True)
        with patch.object(api, '_client', return_value=client) as factory:
            self.assertIs(mf.text_embeddings(['prompt'], **options), client.text_embeddings.return_value)
            factory.assert_called_once_with('cuda:1', 'cache', 'models', True)
            client.text_embeddings.assert_called_once_with(['prompt'])
            factory.reset_mock()
            self.assertIs(mf.image_embeddings('image.png', 'rsna', **options), client.image_embeddings.return_value)
            factory.assert_called_once_with('cuda:1', 'cache', 'models', True)
            client.image_embeddings.assert_called_once_with('image.png', 'rsna')

    def test_end_to_end_custom_prompts_and_client_scorer(self):
        self.model.encode_text_normalized.side_effect = [torch.tensor([[0., 4.]]), torch.tensor([[4., 0.]])]
        positive = self.client.text_embeddings(['positive finding'])
        negative = self.client.text_embeddings(['no finding'])
        self.model.encode_image_normalized.return_value = torch.tensor([[0., 5.]])
        with patch('mammofm._inputs.load_image', return_value=(Image.new('RGB', (12, 20)), {})), \
                patch('mammofm._inputs.inference_tensor', return_value=torch.zeros(1, 3, 20, 12)):
            image = self.client.image_embeddings('image.png', 'rsna')
        result = mf.zero_shot(image, positive, negative)
        self.assertEqual(result, self.client.zero_shot(image, positive, negative))
        self.assertEqual(result['score'], [1.])
        self.assertNotIn('probability', result)
        self.assertEqual(result['positive_cosine'], [1.])
        self.assertEqual(result['negative_cosine'], [0.])
        self.assertEqual(ENCODER['filename'], 'Mammo-FM_BatmanlabTrained_CLIP.tar')


if __name__ == '__main__':
    unittest.main()
