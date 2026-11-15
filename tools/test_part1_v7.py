"""Local delivery audit of the independent v7 artifact, not GPU quality evidence."""
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
    def test_v6_artifact_is_preserved_byte_for_byte(self):
        old = ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb'
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),
                         '73cab0044dde8763b7f8e95511288d888da21962d55a405606769b1959be1c01')

    def test_new_notebook_is_small_independent_and_hash_checked(self):
        path = ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_v7.ipynb'
        notebook = nbformat.read(path, 4)
        nbformat.validate(notebook)
        self.assertLess(path.stat().st_size, 950_000)
        codes = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
        for source in codes:
            compile(source, 'v7-delivery-cell', 'exec')
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
            self.assertEqual(len(manifest['resources']), 3)
            self.assertFalse(any(name.startswith('part2_diffusion/') for name in bundle.namelist()))
            self.assertIn('part1_rag/src/evidence_answer.py', bundle.namelist())
            self.assertEqual(bundle.read('part1_rag/src/evidence_answer.py'),
                             (ROOT / 'part1_rag/src/evidence_answer.py').read_bytes())
            self.assertEqual(bundle.read('part1_rag/eval/golden_candidate.json'),
                             (ROOT / 'part1_rag/eval/golden_candidate.json').read_bytes())
            child = nbformat.reads(bundle.read('part1_rag/notebooks/Part1_Kaggle_Experiment.ipynb').decode(), 4)
            joined = '\n'.join(c.source for c in child.cells if c.cell_type == 'code')
            self.assertIn('LAB2_RAG_EVIDENCE_FIRST', joined)
            self.assertIn('test_evidence_answer.py', joined)
            self.assertIn('process_env["LAB2_RAG_EVIDENCE_FIRST"] = "0"', joined)
            self.assertIn('publish_submission(ROOT)', joined)
            self.assertIn('validate_submission.py', joined)
            self.assertLess(joined.index('publish_submission(ROOT)'), joined.index('part1_rag/src/evaluate_rag.py'))
            for cell in child.cells:
                if cell.cell_type == 'code':
                    compile(cell.source, 'v7-child-cell', 'exec')


if __name__ == '__main__':
    unittest.main(verbosity=2)
