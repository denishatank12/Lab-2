"""Stage real baseline images/records for a private Hub dataset, without GPU work."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

from final_submission_audit import ROOT


def prepare(archive):
    destination = ROOT / 'output/hub_staging/part3_baseline'
    files = {}
    with zipfile.ZipFile(archive) as bundle:
        status = json.loads(bundle.read('baseline_outputs/execution_status.json'))
        if not status.get('completed') or status.get('mock_only'):
            raise ValueError('Use the completed real baseline ZIP')
        for name in bundle.namelist():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe ZIP member')
            if not name.startswith('baseline_outputs/'):
                continue
            data = bundle.read(name)
            if relative.suffix in {'.json', '.csv', '.txt'} and re.search(rb'\bhf_[A-Za-z0-9]{20,}\b', data):
                raise ValueError('Credential-like data found; do not upload')
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
    manifest = dict(condition='Denisha baseline on Tesla T4', files=files,
                    source_archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                    partner_final_artifacts_included=False)
    (destination / 'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (destination / 'README.md').write_text('''---
task_categories:
  - image-to-text
  - image-to-image
tags:
  - academic-evaluation
---
# Team 09 Part 3 baseline artifacts

Denisha Ketan Tank and Shibin Biji Thomas Lnu, DATA 266 Lab 2.
This is Denisha's actual no-revision-loop baseline on one Tesla T4, seed 266.
It includes original COCO-based inputs, adversarial/invalid test fixtures,
generated outputs, 25 states, 25 traces, hardware/model records and a blank
human-rating form. Invalid image fixtures are intentional, not usable images.
Automatic proxies are not human judgments. The partner's final raw outputs
are not in this package and must be obtained separately for the complete
paired artifact release. No training weights or secret tokens are included.

Keep this dataset private until the team reviews sharing and grader access.
COCO photos retain their creator licenses; see the recorded input dataset
provenance and COCO terms. Generated Naruto-domain images do not grant rights
to copyrighted characters or training imagery. This is non-commercial course
research subject to input and model licenses. Do not use outputs as real-world
evidence. Model safety checks were retained; they are not guarantees.
''')
    record = dict(staged_folder=str(destination.relative_to(ROOT)), files=len(files),
                  source_archive_sha256=manifest['source_archive_sha256'], uploaded=False,
                  partner_final_artifacts_included=False,
                  pending='Account-authenticated upload, partner final artifact ZIP, rights review and grader access')
    (ROOT / 'submission/part3_hub_staging.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Staged original baseline artifacts:', destination)
    print('No training, inference or upload performed.')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--repo-id', default='denishatank/data266-lab2-p3-baseline')
    args = parser.parse_args()
    folder = prepare(args.archive)
    if args.upload:
        from huggingface_hub import HfApi, get_token
        if not get_token():
            raise RuntimeError('Sign in securely with the Hugging Face CLI or an HF_TOKEN environment secret. Never paste a token into source.')
        api = HfApi()
        api.create_repo(args.repo_id, repo_type='dataset', private=True, exist_ok=True)
        info = api.repo_info(args.repo_id, repo_type='dataset')
        if not info.private:
            raise RuntimeError('The destination is public. This uploader requires a private dataset for review.')
        commit = api.upload_folder(repo_id=args.repo_id, repo_type='dataset', folder_path=folder,
                                   commit_message='Add the measured Team 09 baseline artifacts')
        record = json.loads((ROOT / 'submission/part3_hub_staging.json').read_text())
        record.update(uploaded=True, dataset='https://huggingface.co/datasets/' + args.repo_id,
                      commit=commit.oid, grader_access_verified=False)
        (ROOT / 'submission/part3_hub_staging.json').write_text(json.dumps(record, indent=2) + '\n')
        print('Uploaded privately:', record['dataset'], commit.oid)
