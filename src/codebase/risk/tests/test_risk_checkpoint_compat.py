import unittest

import numpy as np
import torch

from mayo_mammofm_risk.model import compatible_model_state
from mayo_mammofm_risk.run import metric_rows


class TestHistoricalCheckpointCompatibility(unittest.TestCase):
    def test_duplicate_historical_encoder_is_removed(self):
        tensor = torch.tensor([1.0, 2.0])
        state = {
            "image_encoder.layer.weight": tensor,
            "aggregator.image_encoder.layer.weight": tensor.clone(),
            "aggregator.fc.weight": torch.ones(1, 2),
        }
        normalized = compatible_model_state(state)
        self.assertEqual(
            set(normalized),
            {"image_encoder.layer.weight", "aggregator.fc.weight"},
        )

    def test_conflicting_duplicate_is_rejected(self):
        state = {
            "image_encoder.layer.weight": torch.tensor([1.0]),
            "aggregator.image_encoder.layer.weight": torch.tensor([2.0]),
        }
        with self.assertRaisesRegex(ValueError, "conflicting encoder tensor"):
            compatible_model_state(state)

    def test_risk_evaluation_reports_auroc_only(self):
        predictions = np.asarray([[0.1] * 5, [0.9] * 5])
        labels = np.asarray([[0.0] * 5, [1.0] * 5])
        masks = np.ones((2, 5), dtype=bool)
        rows = metric_rows(predictions, labels, masks)
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(set(row), {"year", "n", "positives", "auroc"})


if __name__ == "__main__":
    unittest.main()
