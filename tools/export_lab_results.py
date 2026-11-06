"""Create a local portable result ZIP, excluding secrets, caches and optimizer pickles."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import zipfile
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]


def export(part):
    root=ROOT/part
    target=ROOT/'output/exports'
    target.mkdir(parents=True,exist_ok=True)
    archive=target/(part+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in (root/'outputs').rglob('*'):
            relative=path.relative_to(root)
            if not path.is_file() or any(p in ['previous_notebooks','import_history','generation_history','hub_export','vector_db','models']
                    or p.startswith('checkpoint-') for p in relative.parts):
                continue
            if path.suffix.lower() in ['.pt','.pth','.pkl','.pickle','.lock']:
                continue
            bundle.write(path,arcname=str(Path(part)/relative))
        for path in (root/'notebooks').glob('*.ipynb'):
            executed=ROOT/'output/executed'/part/path.name
            if executed.is_file():
                value=json.loads(executed.read_text())
                original=json.loads(path.read_text())
                fingerprint=hashlib.sha256(json.dumps([c['source'] if isinstance(c['source'],str) else ''.join(c['source']) for c in original['cells']],
                    sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                if (value.get('metadata',{}).get('overnight_execution',{}).get('completed') is True
                    and value['metadata']['overnight_execution']['source_cells_sha256']==fingerprint):
                    path=executed
            bundle.write(path,arcname=str(Path(part)/'notebooks'/path.name))
        for path in [ROOT/'docs/Final_Lab_Report_DRAFT.pdf',ROOT/'runtime_checks/final_submission_audit.json']:
            if path.is_file():
                bundle.write(path,arcname=str(path.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip():
            raise RuntimeError('Result archive integrity check failed.')
    print('Result ZIP created on the runtime filesystem:',archive)
    print('Size MiB:',round(archive.stat().st_size/2**20,2))
    print('Save the executed notebook first and rerun this export cell to include its latest outputs. '
          'A local/Docker Colab runtime stores this on the GPU machine, not automatically in browser Downloads.')
    return archive


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--part',required=True,choices=['part1_rag','part2_diffusion','part3_agents'])
    export(parser.parse_args().part)
