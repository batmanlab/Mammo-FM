import unittest
from types import SimpleNamespace
import torch
from mammo_sae.core import shapley, shapley_all_background


class IdentitySAE:
    model = SimpleNamespace(n_latents=2)
    def encode(self, features, level=-1):
        return features
    def decode(self, values):
        return values


class FullBackgroundTests(unittest.TestCase):
    def test_additive_contribution(self):
        result = shapley_all_background(IdentitySAE(), torch.tensor([[2., 3.]]),
            lambda x: x.sum(), [0, 1], samples=20)
        self.assertEqual(float(result[0]), 2.)
        self.assertEqual(float(result[1]), 3.)

    def test_unselected_neuron_is_not_held_active(self):
        sae = IdentitySAE(); x = torch.ones(1, 2)
        predictor = lambda values: values.prod()
        selected = shapley(sae, x, predictor, [0], samples=100)
        full = shapley_all_background(sae, x, predictor, [0], samples=100, seed=42)
        self.assertEqual(float(selected[0]), 1.)
        self.assertAlmostEqual(float(full[0]), .5, delta=.15)

    def test_invalid_neurons_rejected(self):
        with self.assertRaises(ValueError):
            shapley_all_background(IdentitySAE(), torch.ones(1,2), torch.sum, [2])


if __name__ == '__main__':
    unittest.main()
