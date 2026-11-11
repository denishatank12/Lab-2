"""Exercise the real isolated notebook kernel before running the longer checks."""
import json
from pathlib import Path
import tempfile

import nbformat

from overnight_runner import execute_saved
from run_checkpoint import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def preflight(root=ROOT):
    root=Path(root)
    target=root/'runtime_checks/standard_kernel_preflight.ipynb'
    source_code='''import json, sys
from IPython import get_ipython
import matplotlib
import matplotlib.pyplot as plt

kernel_class = type(get_ipython().kernel).__module__ + '.' + type(get_ipython().kernel).__name__
assert kernel_class == 'ipykernel.ipkernel.IPythonKernel'
assert not any(name.startswith('google.colab') for name in sys.modules)
fig, ax = plt.subplots(figsize=(2, 1))
ax.plot([0, 1], [0, 1])
ax.set_title('Kernel startup check, not lab results')
plt.show()
plt.close(fig)
print(json.dumps(dict(python=sys.executable, kernel_class=kernel_class, backend=matplotlib.get_backend())))
'''
    with tempfile.TemporaryDirectory(prefix='lab2-kernel-preflight-') as temporary:
        source=Path(temporary)/'preflight.ipynb'
        nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source_code)]),source)
        execute_saved(source,target,root)
    saved=nbformat.read(target,as_version=4)
    outputs=saved.cells[0].outputs
    if not saved.metadata['overnight_execution']['completed'] or any(o.output_type=='error' for o in outputs):
        raise RuntimeError('The standard child kernel did not complete. Keep the saved diagnostic notebook.')
    if not any('image/png' in o.get('data',{}) for o in outputs):
        raise RuntimeError('The standard kernel started but its inline PNG output was not saved.')
    details=next(json.loads(line) for o in outputs if o.output_type=='stream'
                 for line in o.text.splitlines() if line.startswith('{"python":'))
    record=dict(passed=True,**details,inline_plot_saved=True,
        scope='Real standard-kernel startup and inline image output, not full GPU inference or model-quality evidence.')
    atomic_json(root/'runtime_checks/standard_kernel_preflight.json',record)
    print(json.dumps(record,indent=2))
    print('STANDARD CHILD KERNEL STARTUP PASSED',flush=True)
    return record


if __name__=='__main__':
    preflight()
