"""Temporary ZIP and browser-protocol fixtures only; no generated lab results."""
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from export_lab_results import export_all,redact_common_secrets
from download_results import register_zip_transfer,offer_download
from run_checkpoint import digest


class BundleTests(unittest.TestCase):
    def fixture(self,root):
        import nbformat
        part=root/'part1_rag'
        (part/'outputs').mkdir(parents=True)
        (part/'notebooks').mkdir()
        notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('SYNTHETIC TEST ONLY')")])
        nbformat.write(notebook,part/'notebooks/Part1.ipynb')
        (part/'outputs/result.json').write_text('{"test_fixture":true}')
        return notebook

    def test_all_parts_and_hashes_and_missing_items_are_honest(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            for part in ['part2_diffusion','part3_agents']:
                (root/part/'outputs').mkdir(parents=True)
                (root/part/'outputs/result.json').write_text('{"test_fixture":true}')
            with redirect_stdout(io.StringIO()):archive=export_all(root)
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                manifest=json.loads(z.read('BUNDLE_MANIFEST.json'))
                for row in manifest['files']:
                    data=z.read(row['path'])
                    self.assertEqual(hashlib.sha256(data).hexdigest(),row['sha256'])
                    self.assertEqual(len(data),row['bytes'])
                self.assertIn('CONTRIBUTIONS.md',manifest['missing_expected_files'])
                self.assertTrue(all(f'{p}/outputs/result.json' in z.namelist() for p in ['part1_rag','part2_diffusion','part3_agents']))

    def test_autosaved_outputs_selected_including_a_failed_run(self):
        import nbformat
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);n=self.fixture(root)
            n.cells[0].execution_count=1
            n.cells[0].outputs=[nbformat.v4.new_output('error',ename='RuntimeError',evalue='SIMULATED FAILURE',traceback=[])]
            n.metadata['overnight_execution']=dict(completed=False,smoke_only=True,source_cells_sha256=digest([c.source for c in n.cells]))
            target=root/'output/executed/part1_rag/Part1.ipynb'
            target.parent.mkdir(parents=True);nbformat.write(n,target)
            with redirect_stdout(io.StringIO()):archive=export_all(root)
            with zipfile.ZipFile(archive) as z:
                saved=json.loads(z.read('part1_rag/notebooks/Part1.ipynb'))
                self.assertEqual(saved['cells'][0]['outputs'][0]['evalue'],'SIMULATED FAILURE')
                self.assertFalse(saved['metadata']['overnight_execution']['completed'])

    def test_models_caches_raw_data_and_old_archives_excluded_but_adapter_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            for name in ['models/base.safetensors','run1/pytorch_lora_weights.safetensors',
                         'run1/checkpoint-20/training_state.pt','history.zip','.env','vector_db/database.json']:
                path=root/'part2_diffusion/outputs'/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'SYNTHETIC FILE FIXTURE')
            raw=root/'part2_diffusion/data/processed/train/original.png'
            raw.parent.mkdir(parents=True);raw.write_bytes(b'NOT A REAL IMAGE')
            with redirect_stdout(io.StringIO()):archive=export_all(root)
            with zipfile.ZipFile(archive) as z:
                names=z.namelist()
                self.assertIn('part2_diffusion/outputs/run1/pytorch_lora_weights.safetensors',names)
                self.assertFalse(any(x in name for name in names for x in ['base.safetensors','training_state.pt','history.zip','.env','vector_db','original.png']))

    def test_common_credentials_redacted_without_editing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            path=root/'part1_rag/outputs/log.txt'
            secret='hf_'+'A'*30
            original='test fixture only '+secret+' https://example.test/?token=SYNTHETIC_SECRET'
            path.write_text(original)
            with redirect_stdout(io.StringIO()):archive=export_all(root)
            with zipfile.ZipFile(archive) as z:
                data=z.read('part1_rag/outputs/log.txt').decode()
                self.assertNotIn(secret,data);self.assertNotIn('SYNTHETIC_SECRET',data)
            self.assertEqual(path.read_text(),original)

    def test_external_symlink_is_not_packaged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            outside=root/'not_requested.txt';outside.write_text('PRIVATE TEST FIXTURE')
            (root/'part1_rag/outputs/link.txt').symlink_to(outside)
            with redirect_stdout(io.StringIO()):archive=export_all(root)
            with zipfile.ZipFile(archive) as z:self.assertFalse(any('link.txt' in n for n in z.namelist()))


class DownloadTests(unittest.TestCase):
    def test_binary_comm_transfers_exact_bytes_and_closes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.zip';data=os.urandom(2*1024*1024+19);path.write_bytes(data)
            targets={}
            manager=SimpleNamespace(register_target=lambda name,handler:targets.update({name:handler}))
            shell=SimpleNamespace(kernel=SimpleNamespace(comm_manager=manager))
            name=register_zip_transfer(shell,path)
            class Comm:
                buffers=[];closed=False
                def on_msg(self,callback):self.read=callback
                def on_close(self,callback):self.close_callback=callback
                def send(self,message,buffers):self.buffers.extend(buffers)
                def close(self):self.closed=True;self.close_callback({})
            comm=Comm();targets[name](comm,{})
            while not comm.closed:comm.read({})
            self.assertEqual(b''.join(comm.buffers),data)
            self.assertTrue(all(len(b)<=1024*1024 for b in comm.buffers))

    def test_native_copy_uses_requested_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);downloads=root/'Downloads';downloads.mkdir()
            path=root/'fixture.zip';path.write_bytes(b'SYNTHETIC ZIP FIXTURE')
            with patch.dict(os.environ,{'LAB2_DOWNLOAD_DIR':str(downloads)}),patch('IPython.get_ipython',return_value=None),redirect_stdout(io.StringIO()):
                offer_download(path)
            self.assertEqual((downloads/path.name).read_bytes(),path.read_bytes())

    @unittest.skipUnless(shutil.which('node'),'Node needed for frontend fixture test')
    def test_frontend_javascript_accepts_complete_not_partial_transfer(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fixture.zip';path.write_bytes(b'XYZ')
            captured=[];targets={}
            manager=SimpleNamespace(register_target=lambda name,handler:targets.update({name:handler}))
            shell=SimpleNamespace(kernel=SimpleNamespace(comm_manager=manager))
            with patch.dict(os.environ,{'LAB2_DOWNLOAD_DIR':''}),patch('download_results.platform.system',return_value='Linux'),patch('IPython.get_ipython',return_value=shell),patch('IPython.display.display',side_effect=lambda item:captured.append(item)),redirect_stdout(io.StringIO()):
                offer_download(path,auto_start=True)
            script=captured[0].data
            harness='''const nodes=[];let saved=false;
global.document={body:{appendChild(n){nodes.push(n)}},createElement(tag){return {tag,style:{},children:[],append(...x){this.children.push(...x)},appendChild(x){this.children.push(x)},click(){if(this.tag==='a')saved=true}}}};
global.google={colab:{kernel:{accessAllowed:true,comms:{async open(){return {send(){},messages:(async function*(){yield {buffers:[new Uint8Array([88,89,90]).buffer]}})()}}}}}};
'''
            verify="\nsetTimeout(()=>{if(!saved)process.exit(7);},30);"
            subprocess.run(['node','-e',harness+script+verify],check=True,capture_output=True,text=True)
            incomplete=harness.replace('88,89,90','88,89')
            subprocess.run(['node','-e',incomplete+script+"\nsetTimeout(()=>{if(saved)process.exit(8)},30);"],check=True,capture_output=True,text=True)

    def test_download_cell_compiles_without_changing_prior_results(self):
        import nbformat
        from update_final_notebooks import add_download_section
        n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('ORIGINAL FIXTURE')",execution_count=1,
            outputs=[nbformat.v4.new_output('stream',name='stdout',text='ORIGINAL FIXTURE\n')])])
        original=dict(n.cells[0])
        add_download_section(n);add_download_section(n)
        self.assertEqual(dict(n.cells[0]),original)
        self.assertEqual(len(n.cells),3)
        compile(n.cells[-1].source,'download-cell','exec')


if __name__=='__main__':
    unittest.main(verbosity=2)
