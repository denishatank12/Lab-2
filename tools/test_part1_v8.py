"""Delivery tests for the source-linked revision, not pretrained GPU scoring."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class DeliveryTests(unittest.TestCase):
    def test_prior_v6_and_v7_notebooks_are_unchanged(self):
        expected = {
            'Lab2_Part1_Kaggle_Final.ipynb': '73cab0044dde8763b7f8e95511288d888da21962d55a405606769b1959be1c01',
            'Lab2_Part1_Kaggle_v7.ipynb': '10de8b19fe4a29bc6cf69d4ad01cf6eb72a8ce8512d9c4e6e7e97c3bba9f6802'}
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / 'part1_rag/notebooks' / name).read_bytes()).hexdigest(), digest)

    def test_artifact_embeds_exact_new_policy_and_tests_without_old_run_dependency(self):
        path = ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_v8.ipynb'
        notebook = nbformat.read(path, 4)
        nbformat.validate(notebook)
        self.assertLess(path.stat().st_size, 950_000)
        codes = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
        for source in codes:
            compile(source, 'v8-outer-cell', 'exec')
        self.assertNotIn('LAB2_PART1_SAVED_ROOT', '\n'.join(codes))
        self.assertIn('require_private_artifacts=False', codes[0])
        self.assertIn('os.environ["LAB2_RAG_EVIDENCE_FIRST"] = "1"', codes[0])
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(codes[0]).body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in {'BUNDLE_BASE64', 'BUNDLE_SHA256'}}
        raw = base64.b64decode(values['BUNDLE_BASE64'])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), values['BUNDLE_SHA256'])
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            manifest = json.loads(bundle.read('STANDALONE_MANIFEST.json'))
            for name, expected in manifest['files'].items():
                content = bundle.read(name)
                self.assertEqual(hashlib.sha256(content).hexdigest(), expected, name)
                if name.endswith('.py'):
                    compile(content, name, 'exec')
            for name in ['part1_rag/src/evidence_answer.py', 'tools/test_evidence_answer.py',
                         'tools/test_part1_pipeline.py', 'tools/test_constrained_decoding.py']:
                self.assertEqual(bundle.read(name), (ROOT / name).read_bytes())
            self.assertEqual(len(manifest['resources']), 3)
            self.assertFalse(any(n.startswith('part2_diffusion/') for n in bundle.namelist()))
            child = nbformat.reads(bundle.read('part1_rag/notebooks/Part1_Kaggle_Experiment.ipynb').decode(), 4)
            joined = '\n'.join(c.source for c in child.cells if c.cell_type == 'code')
            for cell in child.cells:
                if cell.cell_type == 'code':
                    compile(cell.source, 'v8-child-cell', 'exec')
            self.assertIn('independent-evidence-first-7b-v8', joined)
            self.assertIn('test_evidence_answer.py', joined)
            self.assertIn('validate_submission.py', joined)
            self.assertLess(joined.index('publish_submission(ROOT)'), joined.index('part1_rag/src/evaluate_rag.py'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
