"""CPU fixture checks for recovery safety, not a real GPU completion claim."""
from contextlib import redirect_stdout
import io
import json
import os
import base64
import ast
from pathlib import Path
import sys
import tempfile
import subprocess
import types
import unittest
import zipfile
from unittest.mock import Mock, patch

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'part1_rag/src')]
import recover_part1_evaluation as recovery
from run_checkpoint import atomic_json


class RecoveryTests(unittest.TestCase):
    def test_real_original_v5_checkpoints_resume_without_regenerating_saved_answer(self):
        """Use the actual failed-version source, but explicitly fictional models/data."""
        content = subprocess.check_output(['git','show',
            'd0f5ff83f70588b38c31061d151eedbe4f9e7597:part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb'], cwd=ROOT)
        notebook = nbformat.reads(content.decode(), 4)
        source = next(c.source for c in notebook.cells if c.cell_type == 'code')
        values = {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(source).body
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
            and node.targets[0].id == 'BUNDLE_BASE64'}
        with tempfile.TemporaryDirectory() as folder:
            saved = Path(folder) / 'original_v5_fixture'
            saved.mkdir()
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(values['BUNDLE_BASE64']))) as bundle:
                bundle.extractall(saved)
            program = r'''
import importlib.util, json, os, sys
from pathlib import Path
root, recovery_file = map(Path, sys.argv[1:])
sys.path[:0] = [str(root/'tools'),str(root/'part1_rag/src')]
os.environ.update(LAB2_RAG_PROFILE='7b-nf4', LAB2_RAG_NATURAL_JSON='1',
                  LAB2_RAG_EVIDENCE_REVIEW='1', LAB2_RAG_REVIEW_REQUIRED='0')
import multimodal
from run_checkpoint import atomic_json, experiment_sources, runtime_identity, file_hash
class Model:
    broken = True
    def json(self,*args,**kwargs):
        return {'judgments':{}} if self.broken else {'judgments':{
            'DOC1:p1':{'supported':False,'reason':'Fictional unsupported test evidence.'}}}
    def unload(self): pass
class Index:
    def __init__(self,*args):
        self.ingestion={'source_sha256':'FICTIONAL FIXTURE ONLY'}
        self.db=type('DB',(),{'close':lambda self:None})()
class Chat:
    calls=[]
    def __init__(self,*args,model=None,dense_only=False,**kwargs):
        self.model=model or Model();self.history=[];self.session_facts=[];self.last_hits=[]
        self.dense=dense_only
    def restore(self,history,facts,hits):
        self.history,self.session_facts,self.last_hits=history,facts,hits
    def answer(self,question):
        self.calls.append((self.dense,question))
        answer=dict(answer='Fictional fixture only.',status='answered',citations=['DOC1:p1'],figure_references=[])
        self.last_hits=[dict(citation='DOC1:p1',doc_id='DOC1',page=1,text='Fixture.',image_path='')]
        self.history.append(dict(user=question,assistant=answer))
        return answer
multimodal.MultimodalIndex, multimodal.Conversation = Index, Chat
import evaluate_rag
output=root/'part1_rag/outputs/v2';output.mkdir(parents=True)
golden=root/'part1_rag/eval/golden_candidate.json'
atomic_json(golden,[dict(id=f'T{i}',question=f'Fictional question {i}',conversation_id=f'C{i}',
 expected_status='answered',expected_numbers=[],required_terms=[],sources=['DOC1:p1'],team_verified=False)
 for i in range(25)])
try: evaluate_rag.evaluate(root/'part1_rag/data',output/'evaluation',golden)
except ValueError as error: assert 'Citation judge' in str(error)
else: raise AssertionError('The original v5 judge must fail in this fixture.')
assert len(Chat.calls)==1
(output/'submission.csv').write_text('FICTIONAL COMPETITION CSV FIXTURE\n')
(output/'questions_used.csv').write_text('FICTIONAL QUESTIONS\n')
atomic_json(output/'submission.provenance.json',dict(
 submission_sha256=file_hash(output/'submission.csv'), questions_sha256=file_hash(output/'questions_used.csv'),
 sources=experiment_sources('part1_rag'), runtime=runtime_identity()))
atomic_json(root/'runtime_checks/part1_completion.json',dict(seconds=5828.516117332,error='FICTIONAL FAILURE'))
Model.broken=False
spec=importlib.util.spec_from_file_location('fixture_recovery',recovery_file)
recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(recovery)
recovery.recover(root)
assert len(Chat.calls)==50, 'A saved answer was regenerated instead of being restored.'
assert Chat.calls.count((True,'Fictional question 0'))==1
record=json.loads((root/'runtime_checks/part1_evaluation_recovery.json').read_text())
assert record['evaluation_completed'] and record['original_files_and_csv_unchanged']
assert record['within_90_minutes'] is False
recovery.recover(root)
assert len(Chat.calls)==50
print('PASS: actual v5 checkpoint recovery; all data/models in this test are fictional CPU fixtures.')
'''
            result = subprocess.run([sys.executable,'-c',program,str(saved),
                str(ROOT/'tools/recover_part1_evaluation.py')], cwd=saved,
                capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, (result.stdout+result.stderr)[-5000:])
            self.assertIn('PASS: actual v5 checkpoint recovery', result.stdout)

    def fixture(self, folder):
        root = Path(folder)
        output = root / 'part1_rag/outputs/v2'
        output.mkdir(parents=True)
        (output / 'submission.csv').write_text('SYNTHETIC CSV FIXTURE, NOT SUBMITTABLE\n')
        (output / 'questions_used.csv').write_text('SYNTHETIC QUESTIONS\n')
        original = root / 'part1_rag/src/evaluate_rag.py'
        original.parent.mkdir(parents=True)
        original.write_text('# Fictional original source fixture\n')
        receipt = dict(submission_sha256=recovery.sha(output / 'submission.csv'),
            questions_sha256=recovery.sha(output / 'questions_used.csv'),
            runtime={'fixture':'same runtime'},
            sources={'part1_rag/src/evaluate_rag.py':recovery.sha(original),
                     'rag_runtime_options':{'LAB2_RAG_NATURAL_JSON':'1'}})
        atomic_json(output / 'submission.provenance.json', receipt)
        atomic_json(root / 'runtime_checks/part1_completion.json',
                    dict(seconds=5828.516117332, error='FICTIONAL OLD ERROR', computational_checks_passed=False))
        return root, output, receipt

    def exercise(self, root, output, receipt, evaluator):
        judge = types.SimpleNamespace(PROTOCOL='exact-source-map-v2', judge_citations=Mock())
        with patch('run_checkpoint.experiment_sources', return_value=receipt['sources']), \
             patch('run_checkpoint.runtime_identity', return_value=receipt['runtime']), \
             patch.object(recovery, 'load_module', side_effect=[judge, evaluator]), \
             patch('part1_kaggle.export', return_value=root / 'fictional.zip'), \
             patch('kaggle_runtime.publish_zip'), patch.dict(os.environ), redirect_stdout(io.StringIO()):
            recovery.recover(root)

    def test_original_sources_and_csv_are_hash_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root, output, receipt = self.fixture(folder)
            self.assertTrue(recovery.validate_saved_files(root, receipt))
            (output / 'submission.csv').write_text('CHANGED')
            with self.assertRaisesRegex(ValueError, 'CSV changed'):
                recovery.validate_saved_files(root, receipt)

    def test_changed_source_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root, output, receipt = self.fixture(folder)
            original = root / 'part1_rag/src/evaluate_rag.py'
            original.write_text('CHANGED SOURCE')
            with self.assertRaisesRegex(ValueError, 'Original source changed'):
                recovery.validate_saved_files(root, receipt)
            receipt['sources'] = {'../outside':'not-a-source-hash'}
            with self.assertRaisesRegex(ValueError, 'Original source changed'):
                recovery.validate_saved_files(root, receipt)

    def test_recovery_keeps_csv_old_failure_and_timing_limit_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root, output, receipt = self.fixture(folder)
            old_record = (root / 'runtime_checks/part1_completion.json').read_bytes()
            def evaluate(*args):
                atomic_json(output / 'evaluation/evaluation.json', dict(results=[
                    {'system':'dense_baseline'}, {'system':'hybrid_multimodal'}]))
            evaluator = types.SimpleNamespace(evaluate=Mock(side_effect=evaluate))
            self.exercise(root, output, receipt, evaluator)
            record = json.loads((root / 'runtime_checks/part1_evaluation_recovery.json').read_text())
            self.assertTrue(record['evaluation_completed'])
            self.assertFalse(record['competition_answers_regenerated'])
            self.assertTrue(record['original_files_and_csv_unchanged'])
            self.assertFalse(record['within_90_minutes'])
            self.assertGreaterEqual(record['combined_attempt_seconds'], 5828.516117332)
            self.assertEqual((root / 'runtime_checks/part1_completion.json').read_bytes(), old_record)
            self.exercise(root, output, receipt, evaluator)
            self.assertEqual(evaluator.evaluate.call_count, 1, 'Completed recovery must not repeat inference.')

    def test_recovery_failure_is_not_hidden_or_marked_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root, output, receipt = self.fixture(folder)
            evaluator = types.SimpleNamespace(evaluate=Mock(side_effect=RuntimeError('Fictional GPU failure')))
            with self.assertRaisesRegex(RuntimeError, 'GPU failure'):
                self.exercise(root, output, receipt, evaluator)
            record = json.loads((root / 'runtime_checks/part1_evaluation_recovery.json').read_text())
            self.assertFalse(record['evaluation_completed'])
            self.assertTrue(record['original_files_and_csv_unchanged'])

    def test_different_runtime_cannot_silently_regenerate_saved_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            root, output, receipt = self.fixture(folder)
            with patch('run_checkpoint.experiment_sources', return_value=receipt['sources']), \
                 patch('run_checkpoint.runtime_identity', return_value={'fixture':'different runtime'}), \
                 patch.object(recovery, 'load_module') as loader, patch.dict(os.environ):
                with self.assertRaisesRegex(ValueError, 'not the original runtime'):
                    recovery.recover(root)
            loader.assert_not_called()

    def test_recovery_notebook_embeds_exact_patch_and_guards_missing_results(self):
        notebook = nbformat.read(ROOT / 'part1_rag/notebooks/Lab2_Part1_Evaluation_Recovery.ipynb', 4)
        codes = [c.source for c in notebook.cells if c.cell_type == 'code']
        for code in codes: compile(code, 'recovery-cell', 'exec')
        with tempfile.TemporaryDirectory() as folder:
            root, output, receipt = self.fixture(folder)
            atomic_json(root / '.lab2_runtime_cuda.json', dict(python=sys.executable))
            env = {}
            with patch.dict(os.environ, LAB2_PART1_SAVED_ROOT=str(root)), redirect_stdout(io.StringIO()):
                exec(codes[0], env)
                exec(codes[2], env)
            for name in ['citation_judge.py', 'recover_part1_evaluation.py']:
                self.assertEqual((env['PATCH_DIR'] / name).read_bytes(), (ROOT / 'tools' / name).read_bytes())
            self.assertEqual((output / 'submission.csv').read_bytes(), b'SYNTHETIC CSV FIXTURE, NOT SUBMITTABLE\n')
            self.assertNotIn('generate_submission.py', codes[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
