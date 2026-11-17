"""Import the completed Part 3 baseline without changing any recorded output."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def import_results(archive):
    archive = Path(archive)
    target = ROOT / 'part3_agents/evidence/denisha_baseline'
    selected = ['PROVENANCE.json', 'runtime_environment.json', 'requirements.resolved.txt',
                'adapter_access.json', 'BUNDLE_MANIFEST.json', 'RUN_LOG_notebook.txt',
                'RUN_LOG_export.txt']
    records = {}
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip():
            raise ValueError('Damaged ZIP')
        for name in bundle.namelist():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive path')
        status = json.loads(bundle.read('baseline_outputs/execution_status.json'))
        metrics = json.loads(bundle.read('baseline_outputs/baseline_metrics.json'))
        if not status.get('completed') or status.get('mock_only') or not metrics.get('real_model_execution'):
            raise ValueError('This is not the completed real baseline')
        notebook = bundle.read('Part3_Baseline_Executed.ipynb')
        value = json.loads(notebook)
        code = [c for c in value['cells'] if c['cell_type'] == 'code']
        if not code or any(c.get('execution_count') is None for c in code):
            raise ValueError('Unexecuted notebook cells')
        if any(o.get('output_type') == 'error' for c in code for o in c.get('outputs', [])):
            raise ValueError('Saved notebook error')
        for name in bundle.namelist():
            if name in selected or (name.startswith('baseline_outputs/') and
                    PurePosixPath(name).suffix in {'.json', '.csv'}):
                destination = target / name
                data = bundle.read(name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                records[str(destination.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        destination = ROOT / 'part3_agents/notebooks/Part3_Baseline_Executed.ipynb'
        destination.write_bytes(notebook)
        records[str(destination.relative_to(ROOT))] = hashlib.sha256(notebook).hexdigest()
        # Images remain local for the report and a separate Hub package, not Git.
        assets = ROOT / 'output/review_assets/part3'
        for name in bundle.namelist():
            if name.startswith('baseline_outputs/') and PurePosixPath(name).suffix.lower() in {'.png', '.jpg'}:
                destination = assets / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(bundle.read(name))
    record = dict(archive=archive.name, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  executed_code_cells=len(code), saved_error_outputs=0, files=records,
                  original_outputs_unchanged=True, human_ratings_completed=False,
                  image_storage='Local report/Hub staging only; no separate image dataset in Git.')
    (target / 'IMPORT.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Imported genuine baseline:', archive.name, 'recorded files:', len(records))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    import_results(parser.parse_args().archive)
