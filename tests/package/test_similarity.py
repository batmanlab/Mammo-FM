import json
import math
import subprocess
import sys
import unittest

import numpy as np
import torch

from mammofm._similarity import zero_shot


class SimilarityTests(unittest.TestCase):
    def test_exact_cosines_and_raw_difference(self):
        result = zero_shot([3, 4], [1, 0], [0, 2])
        self.assertEqual(set(result), {
            "positive_cosine", "negative_cosine", "score", "metadata",
        })
        self.assertAlmostEqual(result["positive_cosine"][0], 0.6)
        self.assertAlmostEqual(result["negative_cosine"][0], 0.8)
        self.assertAlmostEqual(result["score"][0], -0.2)
        self.assertEqual(result["metadata"], {
            "scoring": "positive_cosine - negative_cosine",
        })

    def test_each_prompt_is_normalized_before_ensemble_mean(self):
        result = zero_shot([1, 0], [[100, 0], [0, 1]], [[-1, 0], [0, -50]])
        self.assertAlmostEqual(result["positive_cosine"][0], 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(result["negative_cosine"][0], -1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(result["score"][0], math.sqrt(2.0))

    def test_multiple_images_and_opposed_and_identical_prototypes(self):
        result = zero_shot([[1, 0], [-2, 0], [0, 3]], [1, 0], [-1, 0])
        self.assertEqual(result["positive_cosine"], [1.0, -1.0, 0.0])
        self.assertEqual(result["negative_cosine"], [-1.0, 1.0, 0.0])
        self.assertEqual(result["score"], [2.0, -2.0, 0.0])
        identical = zero_shot([[1, 0], [0, 1]], [[2, 0]], [4, 0])
        self.assertEqual(identical["score"], [0.0, 0.0])

    def test_swapping_positive_and_negative_negates_score(self):
        images = [[1, 2, 3], [-4, 2, 1]]
        positive = [[3, 1, 5], [1, 0, -1]]
        negative = [[-1, -2, 1]]
        first = zero_shot(images, positive, negative)
        second = zero_shot(images, negative, positive)
        self.assertEqual(first["positive_cosine"], second["negative_cosine"])
        self.assertEqual(first["negative_cosine"], second["positive_cosine"])
        np.testing.assert_allclose(first["score"], -np.asarray(second["score"]), rtol=0, atol=0)

    def test_python_floats_and_json_serialization(self):
        result = zero_shot(np.array([[1, 2], [3, 4]], dtype=np.float32), [1, 0], [0, 1])
        for key in ("positive_cosine", "negative_cosine", "score"):
            self.assertIsInstance(result[key], list)
            self.assertTrue(all(type(value) is float for value in result[key]))
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_numpy_inputs_are_not_mutated_including_readonly_views(self):
        images = np.array([[1.0, 2.0], [4.0, 3.0]])[:, ::-1]
        positive = np.array([[3.0, 1.0], [1.0, 5.0]])
        negative = np.array([1.0, -2.0])
        originals = [value.copy() for value in (images, positive, negative)]
        for value in (images, positive, negative):
            value.setflags(write=False)
        zero_shot(images, positive, negative)
        for value, original in zip((images, positive, negative), originals):
            np.testing.assert_array_equal(value, original)

    def test_large_and_tiny_finite_inputs_normalize_stably(self):
        maximum = np.finfo(np.float64).max
        tiny = np.nextafter(0.0, 1.0)
        with np.errstate(all="raise"):
            result = zero_shot([[maximum, maximum], [tiny, tiny]], [maximum, 0], [0, -maximum])
        expected = 1.0 / math.sqrt(2.0)
        np.testing.assert_allclose(result["positive_cosine"], [expected, expected], rtol=0, atol=1e-15)
        np.testing.assert_allclose(result["negative_cosine"], [-expected, -expected], rtol=0, atol=1e-15)
        json.dumps(result, allow_nan=False)

    def test_extended_precision_finite_values_remain_supported(self):
        maximum = np.finfo(np.longdouble).max
        result = zero_shot(np.array([maximum, maximum], dtype=np.longdouble), [1, 0], [0, 1])
        self.assertAlmostEqual(result["positive_cosine"][0], 1.0 / math.sqrt(2.0))
        self.assertEqual(result["score"], [0.0])
        json.dumps(result, allow_nan=False)

    def test_extended_precision_score_exactly_matches_returned_cosine_difference(self):
        result = zero_shot(
            np.array([[1, 2, 3], [3, 1, 2]], dtype=np.longdouble),
            np.array([7, 2, 1], dtype=np.longdouble),
            np.array([-2, 8, 3], dtype=np.longdouble),
        )
        for positive, negative, score in zip(
                result["positive_cosine"], result["negative_cosine"], result["score"]):
            self.assertEqual(score, positive - negative)

    def test_empty_ragged_wrong_rank_and_nonfinite_inputs_are_rejected(self):
        invalid = (
            [], [[]], np.empty((0, 2)), np.empty((2, 0)),
            [[1, 2], [3]], 3.0, np.ones((1, 1, 2)),
            [1.0, float("nan")], [float("inf"), 0], [0, -float("inf")],
        )
        for value in invalid:
            for position in range(3):
                args = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
                args[position] = value
                with self.subTest(value=repr(value), position=position), self.assertRaises(ValueError):
                    zero_shot(*args)

    def test_boolean_complex_and_nonnumeric_inputs_are_rejected(self):
        invalid = (
            [True, False], [True, 1], [[1, 0], [False, 1]],
            [np.array([True, False]), np.array([1, 0])],
            np.array([True, False]), np.array([1, 2], dtype=object),
            [1 + 0j, 2], ["1", "2"], None, {"x": 1},
            torch.tensor([True, False]), torch.tensor([1 + 0j, 2 + 0j]),
        )
        for value in invalid:
            for position in range(3):
                args = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
                args[position] = value
                with self.subTest(value=repr(value), position=position), self.assertRaises(TypeError):
                    zero_shot(*args)

    def test_mismatched_dimensions_are_rejected(self):
        for args in (([1, 0], [1, 0, 0], [0, 1]), ([1, 0], [1, 0], [0, 1, 0])):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "same dimension"):
                zero_shot(*args)

    def test_zero_rows_and_zero_class_means_are_rejected(self):
        for position in range(3):
            args = [[1, 0], [1, 0], [0, 1]]
            args[position] = [[1, 0], [0, 0]]
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, "zero vector"):
                zero_shot(*args)
        for position in (1, 2):
            args = [[1, 0], [1, 0], [0, 1]]
            args[position] = [[1, 0], [-3, 0]]
            with self.subTest(mean_position=position), self.assertRaisesRegex(ValueError, "mean.*zero vector"):
                zero_shot(*args)

    def test_torch_inputs_detach_and_do_not_mutate_or_populate_gradients(self):
        images = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        positive = torch.tensor([2.0, 1.0], requires_grad=True)
        negative = torch.tensor([-1.0, 0.5], requires_grad=True)
        before = [value.detach().clone() for value in (images, positive, negative)]
        result = zero_shot(images, positive, negative)
        expected = zero_shot(*(value.numpy() for value in before))
        self.assertEqual(result, expected)
        for value, saved in zip((images, positive, negative), before):
            self.assertTrue(torch.equal(value, saved))
            self.assertIsNone(value.grad)
            self.assertTrue(value.requires_grad)
        if torch.cuda.is_available():
            with self.subTest(device="cuda"):
                cuda_images = images.detach().to("cuda").requires_grad_()
                cuda_positive = positive.detach().to("cuda")
                cuda_negative = negative.detach().to("cuda")
                self.assertEqual(zero_shot(cuda_images, cuda_positive, cuda_negative), expected)
                self.assertIsNone(cuda_images.grad)
                self.assertEqual(cuda_images.device.type, "cuda")

    def test_torch_integer_bfloat16_and_noncontiguous_inputs(self):
        images = torch.tensor([[1, 2], [3, 4]]).t()
        positive = torch.tensor([1, 0], dtype=torch.bfloat16)
        negative = torch.tensor([0, 1], dtype=torch.int32)
        expected = zero_shot([[1, 3], [2, 4]], [1, 0], [0, 1])
        self.assertEqual(zero_shot(images, positive, negative), expected)

    def test_import_is_lazy_and_list_scoring_does_not_load_models(self):
        code = """
import sys
import mammofm
import mammofm._similarity
assert not ({'numpy', 'torch', 'transformers', 'huggingface_hub'} & sys.modules.keys())
result = mammofm._similarity.zero_shot([1, 0], [1, 0], [0, 1])
assert not ({'torch', 'transformers', 'huggingface_hub', 'mammofm._model'} & sys.modules.keys())
assert result['positive_cosine'] == [1.0]
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
