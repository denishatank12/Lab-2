"""Interruption/resume regressions. All fixture answers are synthetic, never lab results."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'part1_rag/src')]
from run_checkpoint import ItemCheckpoint,atomic_json
from vlm_runtime import placement_plan


class CheckpointTests(unittest.TestCase):
    def test_changed_protocol_cannot_reuse_a_result(self):
        with tempfile.TemporaryDirectory() as folder:
            old=ItemCheckpoint(folder,{'test':'old'})
            old.put('item',{'answer':'synthetic fixture only'})
            self.assertIsNone(ItemCheckpoint(folder,{'test':'changed'}).get('item'))
            self.assertIsNotNone(old.get('item'))

    def test_missing_or_changed_image_invalidates_only_its_item(self):
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/'image.fixture'
            image.write_bytes(b'NOT A REAL MODEL IMAGE')
            cache=ItemCheckpoint(folder,{'test':'artifact'})
            cache.put('image',{'test_fixture':True},[image])
            cache.put('other',{'test_fixture':True})
            image.write_bytes(b'CHANGED FIXTURE')
            self.assertIsNone(cache.get('image'))
            self.assertIsNotNone(cache.get('other'))

    def test_corrupt_checkpoint_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            cache=ItemCheckpoint(folder,{'test':'corrupt'})
            cache.path('item').write_text('{truncated')
            self.assertIsNone(cache.get('item'))

    def test_interrupted_atomic_replace_preserves_old_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.json'
            atomic_json(path,{'original':True})
            with patch('run_checkpoint.os.replace',side_effect=OSError('SIMULATED INTERRUPTION')):
                with self.assertRaises(OSError):atomic_json(path,{'replacement':True})
            self.assertEqual(json.loads(path.read_text()),{'original':True})
            self.assertEqual(list(Path(folder).glob('*.tmp')),[])

    def test_other_hardware_does_not_borrow_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('run_checkpoint.runtime_identity',return_value={'gpu':'fixture-A'}):
                cache=ItemCheckpoint(folder,{'test':'hardware'})
                cache.put('item',{'seconds':1,'test_fixture':True})
            with patch('run_checkpoint.runtime_identity',return_value={'gpu':'fixture-B'}):
                self.assertIsNone(ItemCheckpoint(folder,{'test':'hardware'}).get('item'))


class ModelPlanTests(unittest.TestCase):
    def test_4050_plan_reserves_activation_memory(self):
        plan=placement_plan(6*2**30)
        self.assertEqual(plan['device_map'],'auto')
        self.assertLessEqual(plan['max_memory'][0],3.25*2**30)
        self.assertTrue(plan['offload_buffers'])

    def test_large_gpu_uses_sequential_full_model_placement(self):
        self.assertEqual(placement_plan(16*2**30)['device_map'],{'':'cuda:0'})

    def test_cpu_cannot_certify_actual_gpu_preflight(self):
        from overnight_preflight import preflight
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'actually run'):
                preflight('part3_agents')


class SubmissionResumeTest(unittest.TestCase):
    def test_golden_evaluation_reuses_saved_turns_in_both_systems(self):
        import evaluate_rag
        class Judge:
            calls=[]
            def json(self,*args,**kwargs):
                self.calls.append(kwargs)
                return {'citations':[{'source':[],'supported':True}]} if len(self.calls)==1 else {
                    'citations':[{'source':'DOC1:p1','supported':True}]}
        judge=Judge()
        checked=evaluate_rag.judge_citations(judge,{'citations':['DOC1:p1']},[{'image_path':'SYNTHETIC-IMAGE-FIXTURE'}])
        self.assertTrue(checked[0]['supported'])
        self.assertEqual(judge.calls[-1]['images'],['SYNTHETIC-IMAGE-FIXTURE'])
        self.assertEqual(len(judge.calls),2)
        class Model:
            def unload(self):pass
        class Index:
            def __init__(self,*args):
                self.ingestion={'source_sha256':'SYNTHETIC GOLDEN TEST DATA'}
                self.db=type('DB',(),{'close':lambda self:None})()
        class Chat:
            interrupted=True
            calls=[]
            def __init__(self,*args,model=None,**kwargs):
                self.model=model or Model();self.history=[];self.session_facts=[];self.last_hits=[]
            def restore(self,history,facts,hits):
                self.history,self.session_facts,self.last_hits=history,facts,hits
            def answer(self,question):
                self.calls.append(question)
                if question=='question 1' and self.interrupted:
                    raise RuntimeError('SIMULATED EVALUATION INTERRUPTION')
                answer=dict(answer='Synthetic test answer only.',status='answered',citations=[],figure_references=[])
                self.last_hits=[dict(citation='DOC1:p1',doc_id='DOC1',page=1)]
                self.history.append(dict(user=question,assistant=answer))
                return answer
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            golden=folder/'golden.json'
            golden.write_text(json.dumps([dict(id=f'T{i}',question=f'question {i}',conversation_id='C1' if i<2 else f'C{i}',
                expected_status='answered',expected_numbers=[],required_terms=[],sources=['DOC1:p1'],team_verified=False)
                for i in range(25)]))
            with patch.object(evaluate_rag,'MultimodalIndex',Index),patch.object(evaluate_rag,'Conversation',Chat):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError,'SIMULATED'):
                        evaluate_rag.evaluate(folder,folder/'output',golden)
                    Chat.interrupted=False
                    evaluate_rag.evaluate(folder,folder/'output',golden)
                    calls=len(Chat.calls)
                    evaluate_rag.evaluate(folder,folder/'output',golden)
            self.assertEqual(Chat.calls.count('question 0'),2)
            self.assertEqual(len(Chat.calls),calls)
            evaluation=json.loads((folder/'output/evaluation.json').read_text())
            self.assertFalse(evaluation['golden_all_team_verified'])

    def test_actual_entry_point_restores_conversation_after_interruption(self):
        import generate_submission
        import multimodal
        class Model:
            def unload(self):pass
        class Index:
            def __init__(self,*args):
                self.ingestion={'source_sha256':'SYNTHETIC TEST DATA'}
                self.db=type('DB',(),{'close':lambda self:None})()
        class Chat:
            interrupted=True
            calls=[]
            def __init__(self,*args,**kwargs):
                self.model=Model();self.history=[];self.session_facts=[];self.last_hits=[]
            def answer(self,question):
                self.calls.append((question,len(self.history)))
                if question=='second' and self.interrupted:
                    raise RuntimeError('SIMULATED INTERRUPTED GPU PROCESS')
                answer=dict(answer='Synthetic test response only.',status='answered',citations=[],figure_references=[])
                self.history.append(dict(user=question,assistant=answer))
                return answer
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            questions=folder/'Questions.csv'
            with questions.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['id','conversation_id','question_id','turn','question'])
                writer.writeheader()
                writer.writerows([dict(id='T1',conversation_id='C1',question_id='Q1',turn=1,question='first'),
                                 dict(id='T2',conversation_id='C1',question_id='Q2',turn=2,question='second')])
            args=['generate_submission.py','--data-dir',str(folder),'--questions-path',str(questions),
                  '--output',str(folder/'submission.csv')]
            with patch.object(multimodal,'MultimodalIndex',Index),patch.object(multimodal,'Conversation',Chat),patch.object(sys,'argv',args):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError,'SIMULATED'):
                        generate_submission.main()
                    Chat.interrupted=False
                    generate_submission.main()
            self.assertEqual([q for q,_ in Chat.calls].count('first'),1)
            self.assertEqual(Chat.calls[-1],('second',1))
            with (folder/'submission.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),2)

    def test_actual_scenario_cli_resumes_all_completed_smoke_items(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'scenarios.json'
            command=[sys.executable,str(ROOT/'part3_agents/src/run_scenarios.py'),'--output',str(path)]
            first=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
            original=path.read_text()
            second=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
            self.assertNotIn('RESUMED',first.stdout)
            self.assertEqual(second.stdout.count('RESUMED unchanged completed scenario'),15)
            self.assertEqual(path.read_text(),original)


class AutosaveTests(unittest.TestCase):
    def test_real_notebook_kernel_saves_completed_outputs(self):
        import nbformat
        from overnight_runner import execute_saved
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            source=folder/'fixture.ipynb';target=folder/'executed.ipynb'
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('TEST FIXTURE, NOT MODEL RESULTS')")]),source)
            with redirect_stdout(io.StringIO()):execute_saved(source,target,folder)
            saved=nbformat.read(target,as_version=4)
            self.assertTrue(saved.metadata['overnight_execution']['completed'])
            self.assertIn('TEST FIXTURE',saved.cells[0].outputs[0].text)
            self.assertFalse(nbformat.read(source,as_version=4).cells[0].outputs)

    def test_real_notebook_failure_is_saved_not_hidden(self):
        import nbformat
        from overnight_runner import execute_saved
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            source=folder/'fixture.ipynb';target=folder/'failed.ipynb'
            nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('BEFORE TEST FAILURE')"),
                nbformat.v4.new_code_cell("raise RuntimeError('SIMULATED TEST FAILURE')")]),source)
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(Exception):execute_saved(source,target,folder)
            saved=nbformat.read(target,as_version=4)
            self.assertFalse(saved.metadata['overnight_execution']['completed'])
            self.assertTrue(any(o.output_type=='error' for o in saved.cells[1].outputs))
            self.assertIn('BEFORE TEST FAILURE',saved.cells[0].outputs[0].text)


if __name__=='__main__':
    unittest.main(verbosity=2)
