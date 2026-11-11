"""Interruption/resume regressions. All fixture answers are synthetic, never lab results."""
from contextlib import redirect_stdout
import csv
import io
import json
import os
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
    def run_system_colab_extension_fixture(self, failing=False):
        """Inject the system IPython path Colab uses, without editing /etc."""
        import nbformat
        from jupyter_client import KernelManager
        from overnight_runner import execute_saved
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            system=folder/'system-ipython';system.mkdir()
            marker=folder/'HOST_CONFIG_WAS_LOADED'
            config=(f"from pathlib import Path\nPath({str(marker)!r}).write_text('HOST CONFIG EXECUTED')\n"
                "c = get_config()\n"
                "c.IPKernelApp.kernel_class = 'google.colab._kernel.Kernel'\n"
                "c.InteractiveShellApp.extensions = ['google.colab']\n"
                "c.InteractiveShellApp.extra_extensions = ['google.colab._import_hooks']\n"
                "c.InteractiveShellApp.matplotlib = 'inline'\n"
                "c.InteractiveShellApp.reraise_ipython_extension_failures = True\n")
            (system/'ipython_config.py').write_text(config)
            source=folder/'fixture.ipynb';target=folder/'executed.ipynb'
            cells=[nbformat.v4.new_code_cell(
                "import os, sys\nfrom IPython import get_ipython\n"
                "assert type(get_ipython().kernel).__module__ == 'ipykernel.ipkernel'\n"
                "assert not any(k.startswith('google.colab') for k in sys.modules)\n"
                "assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'\n"
                "assert os.environ['HF_TOKEN'] == 'SYNTHETIC_TOKEN_NOT_A_CREDENTIAL'\n"
                "import matplotlib.pyplot as plt\n"
                "plt.plot([0, 1], [0, 1]); plt.show(); plt.close('all')\n"
                "print('SYSTEM CONFIG ISOLATION FIXTURE')")]
            if failing:
                cells.append(nbformat.v4.new_code_cell("raise RuntimeError('SIMULATED CELL FAILURE')"))
            nbformat.write(nbformat.v4.new_notebook(cells=cells),source)
            original_format=KernelManager.format_kernel_cmd
            injection=("import IPython.core.application as _host_app; "
                f"_host_app.ENV_CONFIG_DIRS[:] = [{str(system)!r}]; "
                f"_host_app.SYSTEM_CONFIG_DIRS[:] = [{str(system)!r}]; ")
            def inject_system_path(manager,extra_arguments=None):
                command=original_format(manager,extra_arguments=extra_arguments)
                if '-c' in command:
                    index=command.index('-c')+1
                    command[index]=injection+'\n'+command[index]
                else:
                    index=command.index('-m')
                    self.assertEqual(command[index+1],'ipykernel_launcher')
                    command=command[:index]+['-c',injection+
                        "import runpy; runpy.run_module('ipykernel_launcher', run_name='__main__')"]+command[index+2:]
                return command
            environment=dict(MPLBACKEND='module://google.colab._matplotlib',
                CUDA_VISIBLE_DEVICES='0',HF_TOKEN='SYNTHETIC_TOKEN_NOT_A_CREDENTIAL')
            with patch.dict(os.environ,environment), patch.object(KernelManager,'format_kernel_cmd',inject_system_path):
                with redirect_stdout(io.StringIO()):
                    if failing:
                        with self.assertRaisesRegex(Exception,'SIMULATED CELL FAILURE'):
                            execute_saved(source,target,folder)
                    else:
                        execute_saved(source,target,folder)
                for key,value in environment.items():
                    self.assertEqual(os.environ[key],value)
            saved=nbformat.read(target,as_version=4)
            self.assertFalse(marker.exists(),'System configuration must not be read, not merely hidden.')
            self.assertEqual((system/'ipython_config.py').read_text(),config)
            self.assertEqual(saved.metadata['overnight_execution']['completed'],not failing)
            self.assertTrue(any('SYSTEM CONFIG ISOLATION FIXTURE' in o.get('text','') for o in saved.cells[0].outputs))
            self.assertTrue(any('image/png' in o.get('data',{}) for o in saved.cells[0].outputs))
            if failing:
                self.assertTrue(any(o.output_type=='error' for o in saved.cells[1].outputs))

    def test_system_colab_extensions_and_backend_cannot_break_child_kernel(self):
        self.run_system_colab_extension_fixture()

    def test_system_colab_extensions_do_not_hide_real_cell_failure(self):
        self.run_system_colab_extension_fixture(failing=True)

    def run_colab_config_fixture(self, failing=False):
        """Real child process, with the missing host-only Colab kernel configured."""
        import nbformat
        from overnight_runner import execute_saved
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            host=folder/'host-config';host.mkdir()
            profile=folder/'host-ipython/profile_default';profile.mkdir(parents=True)
            config="c = get_config()\nc.IPKernelApp.kernel_class = 'google.colab._kernel.Kernel'\n"
            (host/'jupyter_config.py').write_text(config)
            (profile/'ipython_kernel_config.py').write_text(config)
            source=folder/'fixture.ipynb';target=folder/'executed.ipynb'
            cells=[nbformat.v4.new_code_cell(
                "import os, sys\nfrom IPython import get_ipython\n"
                "assert type(get_ipython().kernel).__module__ == 'ipykernel.ipkernel'\n"
                "assert os.environ['CUDA_VISIBLE_DEVICES'] == '0'\n"
                "assert os.environ['HF_TOKEN'] == 'SYNTHETIC_TOKEN_NOT_A_CREDENTIAL'\n"
                "print('STANDARD KERNEL FIXTURE:', sys.executable)")]
            if failing:
                cells.append(nbformat.v4.new_code_cell("raise RuntimeError('SIMULATED CELL FAILURE')"))
            nbformat.write(nbformat.v4.new_notebook(cells=cells),source)
            environment=dict(JUPYTER_CONFIG_DIR=str(host),JUPYTER_CONFIG_PATH=str(host),
                JUPYTER_NO_CONFIG='',IPYTHONDIR=str(profile.parent),
                CUDA_VISIBLE_DEVICES='0',HF_TOKEN='SYNTHETIC_TOKEN_NOT_A_CREDENTIAL')
            with patch.dict(os.environ,environment):
                with redirect_stdout(io.StringIO()):
                    if failing:
                        with self.assertRaisesRegex(Exception,'SIMULATED CELL FAILURE'):
                            execute_saved(source,target,folder)
                    else:
                        execute_saved(source,target,folder)
                for key,value in environment.items():
                    self.assertEqual(os.environ[key],value)
            saved=nbformat.read(target,as_version=4)
            self.assertEqual(saved.metadata['overnight_execution']['completed'],not failing)
            self.assertIn('STANDARD KERNEL FIXTURE:',saved.cells[0].outputs[0].text)
            if failing:
                self.assertTrue(any(o.output_type=='error' for o in saved.cells[1].outputs))
            self.assertEqual((host/'jupyter_config.py').read_text(),config)
            self.assertEqual((profile/'ipython_kernel_config.py').read_text(),config)

    def test_colab_host_config_cannot_break_standard_child_kernel(self):
        self.run_colab_config_fixture()

    def test_colab_host_config_still_preserves_real_cell_errors(self):
        self.run_colab_config_fixture(failing=True)

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
