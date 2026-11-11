"""Execute Parts 1 and 3 sequentially and save notebook outputs after every cell."""
import argparse
import atexit
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from run_checkpoint import atomic_json,digest


# IPython searches system-level /etc/ipython even with a fresh IPYTHONDIR and
# JUPYTER_NO_CONFIG. Do not execute those host files in the separate environment.
# Only configuration discovery is replaced; standard cell execution, extensions
# explicitly loaded by a cell, error propagation and inline plots still work.
KERNEL_BOOTSTRAP = '''from ipykernel.kernelapp import IPKernelApp
from ipykernel.ipkernel import IPythonKernel
from traitlets.config import Config

class LabKernelApp(IPKernelApp):
    def init_config_files(self):
        self.config_file_paths = [self.profile_dir.location]
        self.config_files = []

    def load_config_file(self, *args, **kwargs):
        # Configuration comes only from the explicit values below and CLI.
        # No user, virtualenv or system configuration Python is executed.
        return None

config = Config()
config.IPKernelApp.kernel_class = IPythonKernel
config.InteractiveShellApp.extensions = []
config.InteractiveShellApp.extra_extensions = []
config.InteractiveShellApp.exec_lines = []
config.InteractiveShellApp.exec_files = []
LabKernelApp.launch_instance(config=config)
'''


@contextmanager
def standard_kernel():
    """Isolate child kernels from Colab's host-only Jupyter/IPython settings."""
    with tempfile.TemporaryDirectory(prefix='lab2-child-kernel-') as profile:
        environment=os.environ.copy()
        environment['JUPYTER_NO_CONFIG']='1'
        environment['IPYTHONDIR']=profile
        environment['MPLBACKEND']='module://matplotlib_inline.backend_inline'
        environment['MPLCONFIGDIR']=str(Path(profile)/'matplotlib')
        for name in ['JUPYTER_CONFIG_PATH','JUPYTER_CONFIG_DIR','IPYTHON_PROFILE',
                     'IPYTHONSTARTUP','PYTHONSTARTUP','PYTHONPATH','PYTHONHOME','MATPLOTLIBRC']:
            environment.pop(name,None)
        manager=KernelManager(kernel_name='python3')
        # Reuse only Jupyter's discovery mechanism, never a host kernelspec's
        # Colab launcher, environment overrides or custom provisioner.
        manager.kernel_spec.argv=[sys.executable,'-I','-c',KERNEL_BOOTSTRAP,
            '-f','{connection_file}']
        manager.kernel_spec.env={}
        manager.kernel_spec.metadata={}
        try:
            yield manager,environment
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()


def execute_saved(source,target,working_dir=ROOT):
    source,target=Path(source),Path(target)
    notebook=nbformat.read(source,as_version=4)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            cell.outputs=[]
            cell.execution_count=None
    notebook.metadata['overnight_execution']=dict(source=str(source),completed=False,
        smoke_only=os.environ.get('LAB2_SMOKE_ONLY')=='1',
        source_cells_sha256=digest([c.source for c in notebook.cells]))
    target.parent.mkdir(parents=True,exist_ok=True)
    def save(**kwargs):
        # Write atomically, retaining the last readable notebook if interrupted.
        atomic_json(target,json.loads(nbformat.writes(notebook)))
    def progress(cell,cell_index,**kwargs):
        if cell.cell_type=='code':
            print(source.name,'cell',cell_index+1,'of',len(notebook.cells),flush=True)
    save()
    started=time.perf_counter()
    with standard_kernel() as (manager,environment):
        client=NotebookClient(notebook,km=manager,timeout=None,allow_errors=False,
            resources={'metadata':{'path':str(working_dir)}},on_cell_start=progress,on_cell_executed=save)
        try:
            client.execute(env=environment,cleanup_kc=True)
            notebook.metadata['overnight_execution'].update(completed=True,seconds=time.perf_counter()-started)
        finally:
            # nbclient can register this before startup fails. Do not leave a
            # second cleanup callback after our manager has already been closed.
            atexit.unregister(client._cleanup_kernel)
            if client.kc is not None:
                client.kc.stop_channels()
            save()
    print('Saved fully executed notebook:',target,flush=True)
    return target


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    if args.smoke_only:
        os.environ['LAB2_SMOKE_ONLY']='1'
        os.environ['LAB2_DEVICE']='cpu'
    else:
        os.environ.pop('LAB2_SMOKE_ONLY',None)
        os.environ['LAB2_DEVICE']='cuda'
        if not os.environ.get('HF_TOKEN') and not (ROOT/'part3_agents/outputs/hub_adapter/hub_provenance.json').is_file():
            raise RuntimeError('Supply the private adapter read token securely in the launcher before starting.')
    from export_lab_results import export
    outcomes=[]
    progress=ROOT/'runtime_checks/overnight_progress.json'
    atomic_json(progress,dict(status='running',completed_parts=[],smoke_only=args.smoke_only))
    try:
        for part,name in [('part1_rag','Part1_RAG_Run.ipynb'),('part3_agents','Part3_Agents_Run.ipynb')]:
            saved=execute_saved(ROOT/part/'notebooks'/name,ROOT/'output/executed'/part/name)
            archive=export(part)
            outcomes.append(dict(part=part,notebook=str(saved),archive=str(archive),smoke_only=args.smoke_only))
            atomic_json(progress,dict(status='running',completed_parts=outcomes,smoke_only=args.smoke_only))
    except Exception as exc:
        atomic_json(progress,dict(status='failed',completed_parts=outcomes,smoke_only=args.smoke_only,error=str(exc)))
        raise
    atomic_json(progress,dict(status='completed',completed_parts=outcomes,smoke_only=args.smoke_only))
    print('Both computational notebooks finished. Part 2 training was not repeated.',flush=True)
    print('Real ratings, reference review, team details, demo, permissions and external submissions are still separate.',flush=True)


if __name__=='__main__':
    main()
