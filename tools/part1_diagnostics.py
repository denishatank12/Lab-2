"""Read-only, credential-redacted diagnostics for a nested notebook failure."""
import json
from pathlib import Path
import re


def safe_text(text, limit=6000):
    text = re.sub(r'\x1b\[[0-9;]*m', '', str(text))
    text = re.sub(r'\bhf_[A-Za-z0-9]{10,}\b', '[TOKEN REDACTED]', text)
    return text[-limit:]


def failure_details(root):
    root = Path(root)
    sections = [f'Run folder: {root}']
    saved = root / 'output/executed/part1_rag/Part1_Kaggle_Experiment.ipynb'
    if saved.is_file():
        try:
            notebook = json.loads(saved.read_text(encoding='utf-8'))
            for i, cell in enumerate(notebook.get('cells', []), 1):
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'error':
                        sections.append(f'EXPERIMENT CELL {i}: ' + safe_text(
                            output.get('ename', '') + ': ' + output.get('evalue', '')))
                        sections.append(safe_text('\n'.join(output.get('traceback', []))))
        except (OSError, ValueError, TypeError) as exc:
            sections.append('Saved notebook could not be read: ' + safe_text(exc))
    logs = root / 'runtime_checks'
    candidates = sorted((p for p in logs.glob('*.log') if p.name != 'part1_kaggle.log'),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    # Include the latest failed inner stages, not just the outer wrapper error.
    for path in candidates[:4]:
        try:
            sections.append(path.name + ':\n' + safe_text(path.read_text(encoding='utf-8', errors='replace')))
        except OSError as exc:
            sections.append(path.name + ': ' + safe_text(exc))
    for name in ['part1_completion.json']:
        path = logs / name
        if path.is_file():
            sections.append(name + ':\n' + safe_text(path.read_text(encoding='utf-8', errors='replace'), 3500))
    if len(sections) == 1:
        sections.append('No saved inner error yet. Keep this session open and retain its results ZIP.')
    return '\n\n'.join(sections)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    print(failure_details(parser.parse_args().root))
