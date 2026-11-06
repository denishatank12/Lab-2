"""Opt-in private upload of final safe agent outputs, never input photos or secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def publish(folder,namespace,rights_reviewed=False):
    if not rights_reviewed:
        raise ValueError('Confirm permitted non-commercial educational sharing before upload.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,95}',namespace):
        raise ValueError('Use your verified Hub username or organization.')
    from huggingface_hub import HfApi
    api=HfApi()
    who=api.whoami()
    if namespace not in {who['name'],*[o['name'] for o in who.get('orgs',[])]}:
        raise ValueError('The namespace does not belong to this authenticated account.')
    folder=Path(folder)
    target=folder/'hub_export'
    target.mkdir(exist_ok=True)
    metadata=[]
    fingerprints={}
    for system,filename in [('baseline','scenario_traces_no_critique.json'),('critique','scenario_traces.json')]:
        source=folder/filename
        fingerprints[system]=hashlib.sha256(source.read_bytes()).hexdigest()
        rows=json.loads(source.read_text())
        for row in rows:
            path=Path(row['output_path']) if row.get('output_path') else None
            if not row.get('safe') or path is None or not path.is_file():
                continue
            name=f"{system}_{row['scenario_id']}_{path.name}"
            shutil.copy2(path,target/name)
            metadata.append(dict(file_name=name,system=system,scenario_id=row['scenario_id'],request=row['request'],
                **row.get('generation_metadata',{})))
    if not metadata:
        raise RuntimeError('There are no final safe generated images to publish.')
    (target/'metadata.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metadata))
    (target/'README.md').write_text('---\npretty_name: DATA 266 Lab 2 agent comparisons\n---\n'
        '# Part 3 baseline and critique outputs\n\nFinal safe outputs only. Original input images, unsafe withheld '
        'outputs, human identities and authentication tokens are not uploaded. Each row records its system, '
        'scenario, model revision, adapter, prompt, seed and sampling settings.\n\n'
        'Non-commercial course research only. Generated anime imagery may reflect copyrighted source '
        'characters; no additional source-image rights are granted. Automatic safety checks are imperfect.\n')
    repo_id=f'{namespace}/data266-lab2-p3-images'
    api.create_repo(repo_id,repo_type='dataset',private=True,exist_ok=True)
    commit=api.upload_folder(repo_id=repo_id,repo_type='dataset',folder_path=target,
        allow_patterns=[*[r['file_name'] for r in metadata],'README.md','metadata.jsonl'],
        commit_message='Save the paired image-agent results and their run settings')
    links=dict(images=f'https://huggingface.co/datasets/{repo_id}',commit=commit.oid,private=True,
               trace_sha256=fingerprints,uploaded_images=len(metadata),grader_access_verified=False)
    (folder/'hub_links.json').write_text(json.dumps(links,indent=2))
    print(json.dumps(links,indent=2))
    return links


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',required=True)
    parser.add_argument('--namespace',required=True)
    parser.add_argument('--rights-reviewed',action='store_true')
    args=parser.parse_args()
    publish(args.folder,args.namespace,args.rights_reviewed)
