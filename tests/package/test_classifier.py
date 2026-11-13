"""Synthetic binary/density heads: safe shape inference and probability semantics."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from mammofm import _model


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        self.encoder_patch = patch('mammofm._model._image_encoder', side_effect=lambda: nn.Linear(2, 2))
        self.encoder_patch.start()
        self.addCleanup(self.encoder_patch.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'synthetic_classifier.pth'

    def checkpoint(self, num_classes=1):
        model = _model.Classifier(num_classes)
        state = dict(model.state_dict())
        torch.save({'model': state}, self.path)
        return state

    def test_binary_default_retains_single_sigmoid_output(self):
        model = _model.Classifier()
        self.assertEqual(model.num_classes, 1)
        self.assertEqual(tuple(model.classifier.classification_head.weight.shape), (1, 2048))
        model.features = Mock(return_value={'after_pooling': torch.ones(2, 2048)})
        logits = model(torch.empty(0))
        probabilities = model.probabilities(torch.empty(0))
        self.assertEqual(tuple(probabilities.shape), (2,))
        torch.testing.assert_close(probabilities, logits.sigmoid().squeeze(-1))

    def test_density_uses_softmax_in_abcd_order_not_independent_sigmoids(self):
        model = _model.Classifier(4)
        with torch.no_grad():
            model.classifier.classification_head.weight.zero_()
            model.classifier.classification_head.bias.copy_(torch.tensor([-2., 0., 1., 3.]))
        model.features = Mock(return_value={'after_pooling': torch.ones(2, 2048)})
        probabilities = model.probabilities(torch.empty(0))
        expected = torch.tensor([[-2., 0., 1., 3.]]).softmax(dim=1).expand(2, -1)
        self.assertEqual(tuple(probabilities.shape), (2, 4))
        torch.testing.assert_close(probabilities, expected)
        torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))
        self.assertFalse(probabilities.requires_grad)
        self.assertFalse(torch.allclose(probabilities[0], torch.tensor([-2., 0., 1., 3.]).sigmoid()))

    def test_classifier_rejects_unsupported_output_count(self):
        for count in (0, 2, 3, 5, True, 4.0, '4'):
            with self.subTest(count=count), self.assertRaises(ValueError):
                _model.Classifier(count)

    def test_safe_loader_infers_both_head_sizes_and_keeps_eval_device(self):
        for count in (1, 4):
            with self.subTest(count=count):
                state = self.checkpoint(count)
                loaded = _model.load_classifier(self.path, device='cpu')
                self.assertEqual(loaded.num_classes, count)
                self.assertFalse(loaded.training)
                self.assertEqual(next(loaded.parameters()).device.type, 'cpu')
                for name, value in loaded.state_dict().items():
                    torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_safe_loader_checks_task_expected_output_count(self):
        self.checkpoint(4)
        self.assertEqual(_model.load_classifier(self.path, num_classes=4).num_classes, 4)
        with self.assertRaisesRegex(ValueError, '4 classifier outputs; expected 1'):
            _model.load_classifier(self.path, num_classes=1)
        self.checkpoint(1)
        with self.assertRaisesRegex(ValueError, '1 classifier outputs; expected 4'):
            _model.load_classifier(self.path, num_classes=4)
        for count in (True, 2, 4.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                _model.load_classifier(self.path, num_classes=count)

    def test_malformed_or_missing_head_is_rejected_before_model_construction(self):
        original = self.checkpoint(4)
        weight_key = 'classifier.classification_head.weight'
        bias_key = 'classifier.classification_head.bias'
        invalid_states = []
        for shape in ((2, 2048), (4, 2047), (2048,), (0, 2048)):
            invalid_states.append({**original, weight_key: torch.zeros(shape)})
        invalid_states.extend([
            {**original, bias_key: torch.zeros(1)},
            {**original, bias_key: torch.zeros(4, 1)},
            {key: value for key, value in original.items() if key != weight_key},
            {key: value for key, value in original.items() if key != bias_key},
        ])
        with patch('mammofm._model.Classifier') as constructor:
            for index, state in enumerate(invalid_states):
                torch.save({'model': state}, self.path)
                with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'classifier head'):
                    _model.load_classifier(self.path)
            constructor.assert_not_called()

    def test_missing_encoder_tensor_still_fails_strict_loading(self):
        state = self.checkpoint(4)
        del state['image_encoder.weight']
        torch.save({'model': state}, self.path)
        with self.assertRaisesRegex(ValueError, 'architecture mismatch'):
            _model.load_classifier(self.path)


if __name__ == '__main__':
    unittest.main()
