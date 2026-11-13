"""CPU checks for scientific invariants and the portable command workflow."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mammo_sae.__main__ import main
from mammo_sae.analysis import spatial_map, sentence_ablation, render_sentences
from mammo_sae.adapters import FourViewPredictor
from mammo_sae.core import SpatialSAE, intervene, shapley
from mammo_sae.data import grouped_indices, extract, read_features
from mammo_sae.impl.sae import MatryoshkaAutoencoder
from mammo_sae.training import statistics


CONFIG = dict(n_inputs=3, n_latents=3, activation="TopKReLU", tied=False,
              normalize=False, latent_soft_cap=0.0, nesting_list=[1, 3],
              relative_importance=[1.0, 1.0])


def identity_sae():
    model = MatryoshkaAutoencoder(**CONFIG)
    with torch.no_grad():
        model.encoder.copy_(torch.eye(3))
        model.decoder.copy_(torch.eye(3))
        model.pre_bias.zero_()
        model.latent_bias.zero_()
    return SpatialSAE(model, torch.tensor([2., 3., 4.]), 2.0).eval()


class SAETest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_preprocessing_and_ablation_baselines(self):
        sae = identity_sae()
        x = torch.tensor([[3., 5., 7.], [4., 6., 8.]])
        torch.testing.assert_close(sae(x), x)
        result = intervene(sae, x, lambda y: y.sum(0), [], level=0)
        torch.testing.assert_close(result["ablation_delta"], torch.zeros(3))
        self.assertGreater(result["reconstruction_delta"].abs().sum().item(), 0)
        result = intervene(sae, x, lambda y: y.sum(0), [0])
        torch.testing.assert_close(result["ablation_delta"], torch.tensor([-3., 0., 0.]))

    def test_single_level_matches_original_encoder(self):
        sae = identity_sae()
        x = torch.randn(7, 3)
        expected, _, _ = sae.model.encode((x-sae.mean)*sae.scale)
        for level in range(len(expected)):
            torch.testing.assert_close(sae.encode(x, level), expected[level])

    def test_shapley_efficiency_and_seed(self):
        sae = identity_sae()
        x = torch.tensor([[3., 5., 7.]])
        predict = lambda y: y.sum(0).square()
        a = shapley(sae, x, predict, [0, 1], samples=5, seed=7)
        b = shapley(sae, x, predict, [0, 1], samples=5, seed=7)
        expected = -intervene(sae, x, predict, [0, 1])["ablation_delta"]
        torch.testing.assert_close(a[0]+a[1], expected)
        torch.testing.assert_close(a[0], b[0])

    def test_patient_disjoint_split(self):
        frame = pd.DataFrame({"patient_id": ["a", "a", "b", "c", "c"],
                              "exam_id": ["a1", "a2", "b1", "c1", "c2"]})
        train, val = grouped_indices(frame)
        self.assertFalse(set(frame.iloc[train].patient_id) & set(frame.iloc[val].patient_id))
        self.assertEqual(len(train)+len(val), len(frame))

    def test_spatial_coordinates_not_row_order(self):
        frame = pd.DataFrame(dict(patch_h=[1,0,1,0], patch_w=[1,0,0,1],
                                  grid_h=[2]*4, grid_w=[2]*4))
        np.testing.assert_array_equal(spatial_map([4,1,3,2], frame), [[1,2],[3,4]])
        with self.assertRaises(ValueError):
            spatial_map([4,1,3], frame.iloc[:3])

    def test_view_order_and_missing_view(self):
        frame = pd.DataFrame(dict(patient_id=["p"]*4, exam_id=["e"]*4,
                                  image_id=["a","b","c","d"], view=["R_CC","L_CC","R_MLO","L_MLO"]))
        adapter = FourViewPredictor(lambda x: x[:, :, 0], ["L_CC","L_MLO","R_CC","R_MLO"])
        torch.testing.assert_close(adapter.bind(frame)(torch.tensor([[3.],[1.],[4.],[2.]])), torch.tensor([1.,2.,3.,4.]))
        with self.assertRaises(ValueError):
            adapter.bind(frame.iloc[:3])

    def test_cleaned_legacy_adapter_matches_model(self):
        from mayo_mammofm_risk.model import FourViewTransformer
        from mammo_sae.adapters import legacy_risk
        model = FourViewTransformer(input_dim=3).eval()
        features = torch.randn(1, 4, 3)
        with torch.no_grad():
            expected = model(features, torch.zeros(1,4,dtype=torch.long),
                             torch.tensor([[1,1,0,0]]), torch.tensor([[0,1,0,1]])).sigmoid()[0]
        frame = pd.DataFrame(dict(patient_id=['p']*4,exam_id=['e']*4,image_id=list('abcd'),
                                  view=['L_CC','R_CC','L_MLO','R_MLO']))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.pt'
            torch.save(dict(model={'aggregator.'+k:v for k,v in model.state_dict().items()}),path)
            with torch.no_grad():
                actual=legacy_risk(path).bind(frame)(features[0])
            torch.testing.assert_close(actual, expected)

    def test_text_null_ablation_and_escaping(self):
        class Text:
            def encode_text(self, sentences):
                return torch.tensor([[1., 0., 0.] for _ in sentences])
            def project_image(self, x):
                return x
        rows = sentence_ablation(identity_sae(), torch.tensor([[3.,5.,7.]]), ["<script>"], Text(), [])
        self.assertEqual(rows[0]["similarity_drop"], 0.0)
        self.assertNotIn("<script>", render_sentences(rows))

    def test_rejects_incompatible_checkpoint_and_nonfinite_features(self):
        sae = identity_sae()
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            sae.encode(torch.tensor([[float("nan"), 1., 2.]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"sae.pt"
            torch.save(dict(model=sae.model.state_dict(), model_config=CONFIG,
                            mean_center=sae.mean, scaling_factor=sae.scale), path)
            changed = dict(CONFIG, latent_soft_cap=1.)
            with self.assertRaisesRegex(ValueError, "differs"):
                SpatialSAE.load(path, changed)

    def test_cli_refuses_to_overwrite_outputs(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"existing.json"
            path.write_text("keep this")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["evaluate", "--features", "unused.npy", "--checkpoint", "unused.pt",
                      "--output", str(path)])
            self.assertEqual(path.read_text(), "keep this")

    def test_risk_commands_preserve_keys_and_shapley_efficiency(self):
        from unittest.mock import patch
        sae = identity_sae()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(dict(model=sae.model.state_dict(), model_config=CONFIG,
                            mean_center=sae.mean, scaling_factor=sae.scale), root/"sae.pt")
            np.save(root/"features.npy", np.array([[3.,5.,7.]]*4, dtype=np.float32))
            views = ["L_CC", "L_MLO", "R_CC", "R_MLO"]
            pd.DataFrame(dict(patient_id=["001"]*4, exam_id=["002"]*4,
                              image_id=list("abcd"), view=views)).to_csv(root/"metadata.csv", index=False)
            (root/"adapter.json").write_text("{}")
            adapter = FourViewPredictor(lambda x: x.sum(dim=1), views)
            args = ["--features",str(root/"features.npy"),"--metadata",str(root/"metadata.csv"),
                    "--checkpoint",str(root/"sae.pt"),"--factory","example:factory",
                    "--adapter-config",str(root/"adapter.json"),"--patient-id","001",
                    "--exam-id","002","--neurons","0","1"]
            with patch("mammo_sae.data.factory", return_value=adapter):
                main(["intervene"]+args+["--output",str(root/"intervene.json")])
                main(["shapley"]+args+["--samples","4","--output",str(root/"shapley.json")])
            intervention = json.loads((root/"intervene.json").read_text())
            attribution = json.loads((root/"shapley.json").read_text())
            self.assertEqual(intervention["patient_id"], "001")
            self.assertEqual(attribution["exam_id"], "002")
            np.testing.assert_allclose(np.sum(list(attribution["values"].values()), axis=0),
                                       -np.array(intervention["values"]["ablation_delta"]))

    def test_cli_train_load_and_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.arange(90, dtype=np.float32).reshape(30,3)/20 + 1
            np.save(root/'train.npy', values[:24])
            np.save(root/'val.npy', values[24:])
            (root/'model.json').write_text(json.dumps(CONFIG))
            main(['train','--train',str(root/'train.npy'),'--val',str(root/'val.npy'),
                  '--model-config',str(root/'model.json'),'--output',str(root/'fit'),
                  '--epochs','2','--batch-size','8'])
            sae = SpatialSAE.load(root/'fit/best.pt')
            _, scale, _ = statistics(values[:24])
            self.assertAlmostEqual(sae.scale.item(), scale, places=6)
            main(['evaluate','--features',str(root/'val.npy'),'--checkpoint',str(root/'fit/best.pt'),
                  '--output',str(root/'metrics.json')])
            self.assertTrue(np.isfinite(json.loads((root/'metrics.json').read_text())['mse']).all())
            self.assertTrue((root/'metrics.json.command.json').exists())

    def test_extract_split_summary_heatmap_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = pd.DataFrame(dict(patient_id=['p1','p2'], exam_id=['e1','e2'],
                                       image_id=['i1','i2'], view=['L_CC','L_CC']))
            images.to_csv(root/'images.csv', index=False)
            extract(root/'images.csv', lambda row: torch.arange(12).reshape(3,2,2).float()+5, root/'features')
            values, frame = read_features(root/'features/features.npy', root/'features/metadata.csv')
            np.testing.assert_array_equal(values[0], [5,9,13])
            self.assertEqual(len(frame), 8)
            main(['split','--features',str(root/'features/features.npy'),
                  '--metadata',str(root/'features/metadata.csv'),'--output',str(root/'split')])
            sae = identity_sae()
            torch.save(dict(model=sae.model.state_dict(), model_config=CONFIG,
                            mean_center=sae.mean, scaling_factor=sae.scale), root/'sae.pt')
            shared = ['--features',str(root/'features/features.npy'),
                      '--metadata',str(root/'features/metadata.csv'),'--checkpoint',str(root/'sae.pt')]
            main(['neurons']+shared+['--top-k','2','--output',str(root/'stats.csv')])
            main(['heatmaps']+shared+['--neurons','0','--output',str(root/'maps')])
            self.assertEqual(len(pd.read_csv(root/'maps/manifest.csv')), 2)
            np.testing.assert_array_equal(np.load(root/'maps/image_000000_neuron_0.npy'), [[6,8],[10,12]])


if __name__ == '__main__':
    unittest.main()
