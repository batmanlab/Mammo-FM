"""Synthetic CLI and four-view exam contract tests; no clinical data or downloads."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from mammofm import cli, exam


class RiskOutputTests(unittest.TestCase):
    def test_mayo_risk_json_is_compact_and_has_exact_five_probabilities(self):
        probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = exam._risk_output('mayo', probabilities)
        self.assertEqual(result, {
            '1yr': 0.1,
            '2yr': 0.2,
            '3yr': 0.3,
            '4yr': 0.4,
            '5yr': 0.5,
            'comment': (
                'Image encoder is Mammo-FM multi-institution checkpoint. '
                'Risk heads are finetuned with Mayo risks. '
                'Risk scores are not calibrated. These are probabilities.'
            ),
        })

    def test_bu_risk_json_has_requested_comment_and_nothing_else(self):
        result = exam._risk_output('bu', [0.1] * 5)
        self.assertEqual(tuple(result), ('1yr', '2yr', '3yr', '4yr', '5yr', 'comment'))
        self.assertEqual(
            result['comment'],
            'Image encoder is Mammo-FM multi-institution checkpoint. '
            'Risk heads are finetuned with BMC risks. '
            'Risk scores are not calibrated. These are probabilities.')
        for excluded in ('checkpoint', 'metadata', 'calibration', 'risk_percent'):
            self.assertNotIn(excluded, result)

    def test_risk_progress_uses_mayo_and_bmc_labels_with_protocol(self):
        messages = []
        with patch.object(
                exam, '_run_one_risk', side_effect=({'1yr': 0.1}, {'1yr': 0.2})):
            mayo, bmc = exam._run_risks(
                images=Mock(), store=Mock(), device=torch.device('cpu'),
                bu_model='s1s2', progress=messages.append)
        self.assertEqual((mayo, bmc), ({'1yr': 0.1}, {'1yr': 0.2}))
        self.assertEqual(messages, [
            'Estimating Mayo-trained risk '
            '(Stage 1 MIRAI KD then Stage 2 BU outcomes)',
            'Estimating BMC-trained risk '
            '(Stage 1 MIRAI KD then Stage 2 BU outcomes)',
        ])


class PreparedViewTests(unittest.TestCase):
    def test_mixed_inputs_convert_only_dicom_and_preserve_canonical_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for view in exam.VIEW_ORDER:
                suffix = '.dcm' if view in {'LMLO', 'RCC'} else '.png'
                path = root / (view + suffix)
                if suffix == '.png':
                    Image.new('L', (2, 2), color=10).save(path)
                else:
                    path.write_bytes(b'DICM-test-placeholder')
                paths[view] = path

            loads = []
            conversions = []
            def convert(source, destination, dataset=None):
                conversions.append((Path(source).name, Path(destination).name, dataset))
                Image.new('L', (2, 2), color=20).save(destination)
                return {'input_format': 'dicom', 'output_shape': [1520, 912, 3]}
            def load(path, dataset=None):
                loads.append((Path(path).name, dataset))
                return Image.open(path).convert('RGB'), {'input_format': 'png'}
            sentinels = iter(range(4))
            def transform(image, **kwargs):
                return torch.full((1, 3, 2, 2), float(next(sentinels)))

            with patch('mammofm._inputs.convert_dicom_to_png', side_effect=convert), \
                    patch('mammofm._inputs.load_image', side_effect=load), \
                    patch('mammofm._inputs.inference_tensor', side_effect=transform):
                with exam._prepared_views(paths) as (images, metadata, prepared):
                    self.assertEqual(images[0, :, 0, 0, 0].tolist(), [0., 1., 2., 3.])
                    self.assertEqual(tuple(prepared), exam.VIEW_ORDER)
                    self.assertTrue(prepared['LMLO'].is_file())
                    self.assertEqual(prepared['LCC'], paths['LCC'])
                    converted_parent = prepared['LMLO'].parent
                    self.assertEqual(metadata['LMLO']['input_format'], 'dicom')
            self.assertFalse(converted_parent.exists())
            self.assertEqual(conversions, [
                ('LMLO.dcm', 'LMLO.png', 'general'),
                ('RCC.dcm', 'RCC.png', 'general'),
            ])
            self.assertEqual([name for name, _ in loads],
                             ['LMLO.png', 'LCC.png', 'RMLO.png', 'RCC.png'])


class ExamAndCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = {view: self.root / f'{view}.png' for view in exam.VIEW_ORDER}
        for path in self.paths.values():
            path.write_bytes(b'placeholder')

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    @contextmanager
    def prepared(_paths):
        metadata = {view: {'input_format': 'png'} for view in exam.VIEW_ORDER}
        yield torch.zeros(1, 4, 3, 2, 2), metadata, _paths

    def test_default_run_writes_three_objects_only_after_all_branches_succeed(self):
        mayo = {'1yr': 0.1, 'comment': 'Mayo'}
        bu = {'1yr': 0.2, 'comment': 'BMC'}
        report = {'stage': 3, 'grounded_report': 'Findings: final\nImpression: final'}
        output = self.root / 'out'
        with patch.object(exam, '_prepared_views', side_effect=self.prepared), \
                patch.object(exam, '_device', return_value=torch.device('cpu')), \
                patch.object(exam, '_run_risks', return_value=(mayo, bu)) as risks, \
                patch.object(exam, '_run_report', return_value=report) as reports:
            result = exam.run_exam(self.paths, output)
        self.assertEqual(result, {'mayo_risk': mayo, 'bu_risk': bu, 'report': report})
        self.assertEqual({path.name for path in output.iterdir()},
                         {'risk_mayo.json', 'risk_bu.json', 'report.json'})
        self.assertEqual(json.loads((output / 'risk_mayo.json').read_text()), mayo)
        self.assertEqual(json.loads((output / 'risk_bu.json').read_text()), bu)
        self.assertEqual(json.loads((output / 'report.json').read_text()), report)
        risks.assert_called_once()
        reports.assert_called_once()
        self.assertEqual(tuple(reports.call_args.args[0]), exam.VIEW_ORDER)

    def test_branch_failure_publishes_no_json(self):
        output = self.root / 'failed'
        with patch.object(exam, '_prepared_views', side_effect=self.prepared), \
                patch.object(exam, '_device', return_value=torch.device('cpu')), \
                patch.object(exam, '_run_risks', return_value=({}, {})), \
                patch.object(exam, '_run_report', side_effect=RuntimeError('report failed')):
            with self.assertRaisesRegex(RuntimeError, 'report failed'):
                exam.run_exam(self.paths, output)
        self.assertEqual(list(output.glob('*.json')), [])

    def test_existing_outputs_are_not_silently_overwritten(self):
        output = self.root / 'existing'
        output.mkdir()
        existing = output / 'report.json'
        existing.write_text('keep me')
        with patch.object(exam, '_prepared_views') as prepared:
            with self.assertRaises(FileExistsError):
                exam.run_exam(self.paths, output)
        prepared.assert_not_called()
        self.assertEqual(existing.read_text(), 'keep me')

    def test_cli_stdout_contains_final_outputs_but_never_preliminary_or_paths(self):
        final = {
            'report': {'stage': 3, 'grounded_report': 'FINAL REPORT'},
            'mayo_risk': {'5yr': 0.1, 'comment': 'Mayo'},
            'bu_risk': {'5yr': 0.2, 'comment': 'BMC'},
        }
        argv = ['exam']
        for view in ('lmlo', 'lcc', 'rmlo', 'rcc'):
            argv += [f'--{view}', f'/PRIVATE/{view}.dcm']
        argv += ['--output-dir', '/PRIVATE/output', '--quiet']
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(cli, 'run_exam', return_value=final) as runner, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(argv)
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), final)
        self.assertEqual(stderr.getvalue(), '')
        self.assertNotIn('SECRET_PRELIMINARY', stdout.getvalue())
        self.assertNotIn('/PRIVATE/', stdout.getvalue())
        self.assertEqual(tuple(runner.call_args.args[0]), exam.VIEW_ORDER)
        self.assertEqual(runner.call_args.kwargs['bu_model'], 's1s2')

    def test_cli_can_select_s2_only_bu_head(self):
        argv = ['exam']
        for view in ('lmlo', 'lcc', 'rmlo', 'rcc'):
            argv += [f'--{view}', f'/PRIVATE/{view}.dcm']
        argv += [
            '--output-dir', '/PRIVATE/output', '--only', 'risk',
            '--bu-model', 's2-only', '--quiet',
        ]
        with patch.object(cli, 'run_exam', return_value={}) as runner, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(argv), 0)
        self.assertEqual(runner.call_args.kwargs['bu_model'], 's2_only')

    def test_report_wrapper_discards_grounding_scores_and_internal_text(self):
        metadata = {
            'visual_encoder': {'filename': 'encoder'},
            'stage2': {'adapter': {'filename': 'stage2'}},
            'base_language_model': {'repo_id': 'llama'},
            'checkpoints': {'mass': {'filename': 'mass'}},
            'generation': {'seed': 0},
        }
        generated = {
            'grounded_report': 'Findings: final\nImpression: final',
            'findings': {
                'views': {'lmlo': {'mass': {'probability': 0.9}}},
                'metadata': metadata,
            },
        }
        fake_client = Mock()
        fake_client.report_gen.return_value = generated
        store = Mock(cache_dir=None, model_dir=None, local_files_only=True)
        with patch('mammofm.api.MammoFM', return_value=fake_client):
            result = exam._run_report(
                self.paths, store, torch.device('cpu'),
                {view: {'input_format': 'png'} for view in exam.VIEW_ORDER},
                base_model_path=None, seed=0, preliminary_max_new_tokens=512,
                grounding_max_new_tokens=320, progress=lambda _message: None)
        serialized = json.dumps(result, allow_nan=False)
        self.assertIn('Findings: final', serialized)
        self.assertNotIn('probability', serialized)
        self.assertNotIn('threshold', serialized)
        self.assertNotIn('cutoff', serialized)
        self.assertNotIn('SECRET_PRELIMINARY', serialized)
        self.assertFalse(result['metadata']['preliminary_report_exposed'])
        self.assertFalse(result['metadata']['grounding_predictions_exposed'])


if __name__ == '__main__':
    unittest.main()
