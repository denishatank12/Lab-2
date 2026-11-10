"""Create a local portable result ZIP, excluding secrets, caches and optimizer pickles."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import zipfile
import json
import hashlib
import os
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
PARTS=['part1_rag','part2_diffusion','part3_agents']
EXCLUDED_DIRS={'previous_notebooks','import_history','generation_history','hub_export','vector_db',
    'models','__pycache__','.cache','.git','.venv','qwen_weight_offload'}
TEXT_SUFFIXES={'.json','.jsonl','.yaml','.yml','.txt','.log','.md','.csv','.ipynb','.py'}


def selected_notebook(repository,part,path):
    executed=repository/'output/executed'/part/path.name
    if executed.is_file():
        value=json.loads(executed.read_text())
        original=json.loads(path.read_text())
        fingerprint=hashlib.sha256(json.dumps([
            c['source'] if isinstance(c['source'],str) else ''.join(c['source']) for c in original['cells']],
            sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if value.get('metadata',{}).get('overnight_execution',{}).get('source_cells_sha256')==fingerprint:
            return executed
    return path


def allowed_result(path,relative):
    if path.is_symlink() or not path.is_file():
        return False
    if any(p in EXCLUDED_DIRS or p.startswith('checkpoint-') for p in relative.parts):
        return False
    if path.suffix.lower() in {'.pt','.pth','.pkl','.pickle','.bin','.pem','.key','.p12','.pfx',
        '.lock','.zip','.download','.tmp','.db','.sqlite'}:
        return False
    if path.suffix.lower()=='.safetensors' and path.name not in {'pytorch_lora_weights.safetensors','adapter_model.safetensors'}:
        return False
    return not path.name.startswith('.')


def redact_common_secrets(data):
    text=data.decode('utf-8')
    text,n=re.subn(r'\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b',
        'REDACTED_CREDENTIAL',text)
    text,count=re.subn(r'(?i)([?&](?:token|access_token|api_key)=)[^&\s"\'<>]+',r'\1REDACTED_CREDENTIAL',text)
    n+=count
    text,count=re.subn(r'(?i)(["\'](?:api_key|hf_token|access_token|password|authorization)["\']\s*:\s*["\'])[^"\']*(["\'])',
        r'\1REDACTED_CREDENTIAL\2',text)
    n+=count
    return text.encode('utf-8'),n


def export_all(repository=ROOT):
    """One review bundle, not a declaration that the Canvas submission is complete."""
    repository=Path(repository).resolve()
    target=repository/'output/exports'
    target.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive=target/f'Lab2_All_Results_{stamp}.zip'
    temporary=archive.with_suffix('.partial.zip')
    members={}
    def add(path,name=None):
        if path.is_file() and not path.is_symlink():
            members[name or path.relative_to(repository).as_posix()]=path
    for name in ['README.md','RERUN_GUIDE.md','AI_USAGE.md','CONTRIBUTIONS.md','links.txt',
                 'requirements.txt','team_review.json','Lab2_Overnight_Run.ipynb','Lab2_RTX4090_Run_All.ipynb']:
        add(repository/name)
    for path in (repository/'tools').glob('*.py'):
        add(path)
    notebook_status={}
    for part in PARTS:
        root=repository/part
        for folder in ['src','eval','evidence']:
            for path in (root/folder).rglob('*'):
                if allowed_result(path,path.relative_to(root)):
                    add(path)
        for name in ['config.yaml','config.yml','submission_details.json','README.md','requirements.txt']:
            add(root/name)
        for path in (root/'outputs').rglob('*'):
            if allowed_result(path,path.relative_to(root)):
                add(path)
        for path in (root/'notebooks').glob('*.ipynb'):
            selected=selected_notebook(repository,part,path)
            add(selected,f'{part}/notebooks/{path.name}')
            try:
                value=json.loads(selected.read_text())
                notebook_status[part]=dict(selected_file=str(selected.relative_to(repository)),
                    execution=value.get('metadata',{}).get('overnight_execution'),
                    saved_errors=sum(o.get('output_type')=='error' for c in value['cells']
                        if c['cell_type']=='code' for o in c.get('outputs',[])))
            except (ValueError,KeyError):
                notebook_status[part]={'readable':False}
        for path in (root/'output/pdf').glob('*.pdf'):
            add(path)
    # Include split/prompt provenance, not the original training image dataset.
    for relative in ['data_summary.json','eval_prompts.json','train/metadata.jsonl','test/metadata.jsonl']:
        add(repository/'part2_diffusion/data/processed'/relative)
    for path in (repository/'runtime_checks').rglob('*'):
        if allowed_result(path,path.relative_to(repository/'runtime_checks')) and path.suffix.lower() in TEXT_SUFFIXES:
            add(path)
    for path in (repository/'validation').glob('*.json'):
        add(path)
    for name in ['Final_Lab_Report_DRAFT.pdf','Part2_Findings_DRAFT.pdf']:
        add(repository/'docs'/name)
    for path in (repository/'docs/rating_forms').glob('*'):
        add(path)
    expected=['part1_rag/outputs/v2/evaluation/evaluation.json',
        'part2_diffusion/outputs/run1/pytorch_lora_weights.safetensors',
        'part2_diffusion/outputs/run2/pytorch_lora_weights.safetensors',
        'part2_diffusion/outputs/generations/manifest.json',
        'part3_agents/outputs/v2/latest_run.json','CONTRIBUTIONS.md','links.txt']
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='All available Lab 2 evidence for review; missing items are not fabricated.',
        runtime_repository=str(repository),notebooks=notebook_status,
        missing_expected_files=[name for name in expected if not (repository/name).is_file()],
        exclusions=['Python environments and caches','pretrained base-model weights','optimizer pickle checkpoints',
            'original datasets and course PDFs','old result archives and history','hidden files and symbolic links'],files=[])
    required=sum(path.stat().st_size for path in members.values())
    if shutil.disk_usage(target).free<required+64*2**20:
        raise RuntimeError('Not enough free disk to safely create the full ZIP. Free space, then rerun only this export cell.')
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as bundle:
            for name,path in sorted(members.items()):
                redactions=0
                if path.suffix.lower() in TEXT_SUFFIXES:
                    try:
                        data,redactions=redact_common_secrets(path.read_bytes())
                    except UnicodeDecodeError:
                        data=path.read_bytes()
                    bundle.writestr(name,data)
                    size=len(data);sha=hashlib.sha256(data).hexdigest()
                else:
                    hasher=hashlib.sha256()
                    size=0
                    with path.open('rb') as source,bundle.open(name,'w',force_zip64=True) as destination:
                        for block in iter(lambda:source.read(2**20),b''):
                            destination.write(block)
                            hasher.update(block)
                            size+=len(block)
                    sha=hasher.hexdigest()
                manifest['files'].append(dict(path=name,bytes=size,sha256=sha,credential_redactions=redactions))
            bundle.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip() is not None:
                raise RuntimeError('ZIP integrity check failed. No complete download was offered.')
        os.replace(temporary,archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    checksum=hashlib.sha256()
    with archive.open('rb') as source:
        for block in iter(lambda:source.read(2**20),b''):
            checksum.update(block)
    record=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=checksum.hexdigest(),
        files=len(members),missing_expected_files=manifest['missing_expected_files'])
    from run_checkpoint import atomic_json
    atomic_json(target/'latest_all_results.json',record)
    print('ALL RESULTS ZIP:',archive)
    print('Size MiB:',round(record['bytes']/2**20,2),'Files:',len(members))
    print('SHA256:',record['sha256'])
    if manifest['missing_expected_files']:
        print('Missing on THIS runtime (see manifest):',', '.join(manifest['missing_expected_files']))
    print('No training or generation was repeated. Review the archive before sharing; common token patterns were redacted.')
    return archive

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
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--part',choices=PARTS)
    choice.add_argument('--all',action='store_true',help='Bundle all available parts, code, reports and logs for review.')
    args=parser.parse_args()
    export_all() if args.all else export(args.part)
