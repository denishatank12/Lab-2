"""Execute revised CPU smoke paths while preserving every original Part 2 GPU output."""
from pathlib import Path
import datetime as dt
import json
import os
import sys
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]


def execute(notebook):
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    def progress(cell,cell_index,**kwargs):
        if cell.cell_type=='code':print('Executing cell',cell_index+1,flush=True)
    NotebookClient(notebook,km=manager,timeout=1800,allow_errors=False,
        resources={'metadata':{'path':str(ROOT)}},on_cell_start=progress).execute()
    return notebook


def main():
    os.environ['LAB2_SMOKE_ONLY']='1'
    os.environ['LAB2_DEVICE']='cpu'
    records=[]
    for part,filename in [('part1_rag','Part1_RAG_Run.ipynb'),('part3_agents','Part3_Agents_Run.ipynb')]:
        path=ROOT/part/'notebooks'/filename
        notebook=nbformat.read(path,as_version=4)
        started=time.perf_counter()
        execute(notebook)
        code=[c for c in notebook.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code)
        assert not [o for c in code for o in c.outputs if o.output_type=='error']
        nbformat.write(notebook,path)
        records.append(dict(part=part,mode='CPU smoke only; real VLM/generation/evaluation skipped',
            executed_code_cells=len(code),saved_errors=0,seconds=time.perf_counter()-started))
    path=ROOT/'part2_diffusion/notebooks/Part2_FineTune_Run.ipynb'
    original=nbformat.read(path,as_version=4)
    before=json.loads(nbformat.writes(original))['cells'][:-1]
    setup=f'''from pathlib import Path
import subprocess,sys,json,os
ROOT=Path({str(ROOT)!r})
PYTHON=Path(sys.executable)
print("Separate CPU completion refresh; original GPU training is not rerun.")
def run_script(relative,*arguments):
    subprocess.run([str(PYTHON),str(ROOT/relative),*map(str,arguments)],cwd=ROOT,check=True)
'''
    section=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(setup),original.cells[-1]])
    execute(section)
    original.cells[-1]=section.cells[-1]
    original.metadata['completion_refresh_scope']='Last cell executed separately on CPU; original GPU cells and results preserved.'
    nbformat.write(original,path)
    assert json.loads(nbformat.writes(original))['cells'][:-1]==before
    records.append(dict(part='part2_diffusion',mode='Original GPU results preserved; CPU refresh only',saved_errors=0))
    output=ROOT/'validation/revised_notebook_smoke.json'
    output.write_text(json.dumps(dict(time=dt.datetime.now().astimezone().isoformat(),notebooks=records),indent=2))
    print(output.read_text())


if __name__=='__main__':
    main()
