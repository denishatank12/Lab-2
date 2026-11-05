"""Execute every notebook cell; explicitly skip unavailable full GPU stages."""
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
records=[]
for part,filename in [('part1_rag','Part1_RAG_Run.ipynb'),
                      ('part2_diffusion','Part2_FineTune_Run.ipynb'),
                      ('part3_agents','Part3_Agents_Run.ipynb')]:
    # Part 1 is short enough to run with real dense embeddings on the CPU.
    os.environ['LAB2_SMOKE_ONLY']='0' if part=='part1_rag' else '1'
    os.environ['LAB2_DEVICE']='cpu'
    path=ROOT/part/'notebooks'/filename
    notebook=nbformat.read(path,as_version=4)
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    started=time.perf_counter()
    def progress(cell,cell_index,**kwargs):
        if cell.cell_type=='code': print(part,'cell',cell_index+1,flush=True)
    client=NotebookClient(notebook,km=manager,timeout=1800,allow_errors=False,
                          resources={'metadata':{'path':str(ROOT)}},on_cell_start=progress)
    try:
        client.execute()
    finally:
        nbformat.write(notebook,path)
    code=[c for c in notebook.cells if c.cell_type=='code']
    errors=[o for c in code for o in c.outputs if o.output_type=='error']
    assert not errors and all(c.execution_count is not None for c in code)
    records.append({'part':part,'mode':'full extractive baseline, CPU dense retrieval' if part=='part1_rag' else 'CPU smoke only',
                    'executed_code_cells':len(code),'error_outputs':len(errors),
                    'seconds':time.perf_counter()-started,'gpu_training_verified':False})
out=ROOT/'runtime_checks/notebook_validation.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({'time':dt.datetime.now().astimezone().isoformat(),'notebooks':records},indent=2))
print(out.read_text(),flush=True)
