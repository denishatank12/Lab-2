"""Entire Part 1 control flow with fictional data/models, never quality evidence.

Unlike SMOKE_ONLY, this exercises the actual generation, answer validation,
citation-judge validation, evaluation, official CSV validator, publication and
export functions. No pretrained GPU inference is claimed by this fixture test.
"""
import csv
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'part1_rag/src')]
import generate_submission
import evaluate_rag
import multimodal
from part1_kaggle import publish_submission, export
from run_checkpoint import atomic_json, file_hash

HIT = dict(citation='DOC1:p1', doc_id='DOC1', page=1, figure_id='', image_path='',
           trust_level='high', text='A fictional metric used only in a software test.')


class FixtureModel:
    calls = []
    profile = {'fixture_only':True}

    def json(self, instruction, **kwargs):
        schema = kwargs.get('schema')
        self.calls.append(schema)
        if schema and 'judgments' in schema['properties']:
            return {'judgments': {'DOC1:p1': {'supported':False, 'reason':'Fictional classification fixture.'}}}
        # Every first call deliberately reproduces the no-figure-evidence bug.
        # The actual retry controller and unchanged validator must repair it.
        value = dict(answer='A fictional metric used only in a software test.', status='answered',
                     citations=['DOC1:p1'], figure_references=[] if schema else ['Invented fixture figure'])
        if os.environ.get('LAB2_RAG_EVIDENCE_FIRST') == '1':
            value['observations'] = [dict(source='DOC1:p1', finding='A fictional metric used only in a software test.')]
            value.pop('citations')
            value.pop('figure_references')
        return value

    def unload(self): pass


class FixtureIndex:
    searches = []

    def __init__(self, data, output):
        self.output = Path(output)
        self.ingestion = dict(source_sha256='FICTIONAL SOFTWARE TEST ONLY', documents={}, page_context={})
        self.records = [HIT]
        self.objects = []
        self.db = type('FixtureDatabase', (), {'close':lambda self:None})()

    def search(self, query, dense_only=False, **kwargs):
        self.searches.append(dense_only)
        return [dict(HIT)]


class PipelineTests(unittest.TestCase):
    def test_evidence_first_prediction_evaluation_publication_and_export(self):
        with patch.dict(os.environ, LAB2_RAG_EVIDENCE_FIRST='1'):
            self.test_fresh_prediction_evaluation_publication_and_export()

    def test_fresh_prediction_evaluation_publication_and_export(self):
        FixtureModel.calls, FixtureIndex.searches = [], []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'working/fresh-fixture'
            data, output = root / 'part1_rag/data', root / 'part1_rag/outputs/v2'
            data.mkdir(parents=True); output.mkdir(parents=True)
            questions = data / 'Questions.csv'
            with questions.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id','conversation_id','question_id','question'])
                writer.writeheader()
                writer.writerows(dict(id=str(i+1), conversation_id=f'C{i+1:02}', question_id='Q01',
                                      question=f'Fictional fixture query {i+1}?') for i in range(40))
            golden = root / 'part1_rag/eval/golden.json'
            atomic_json(golden, [dict(id=f'G{i}', conversation_id=f'GC{i}', question=f'Fictional golden query {i}?',
                expected_status='answered', expected_numbers=[], required_terms=['fictional'],
                sources=['DOC1:p1'], team_verified=False) for i in range(25)])
            shutil.copy2(questions, output / 'questions_used.csv')
            shutil.copy2(ROOT / 'part1_rag/data/validate_submission.py', data / 'validate_submission.py')
            atomic_json(output / 'run_inputs.json', dict(smoke_only=False, fixture_only=True,
                                                       questions_sha256=file_hash(questions)))
            argv = ['generate_submission.py', '--data-dir', str(data), '--questions-path', str(questions),
                    '--output', str(output / 'submission.csv'), '--require-dense']
            with patch.dict(os.environ, LAB2_RAG_PROFILE='7b-nf4', LAB2_RAG_NATURAL_JSON='1',
                            LAB2_RAG_EVIDENCE_REVIEW='0', LAB2_RAG_REVIEW_REQUIRED='0'), \
                 patch.object(multimodal, 'MultimodalIndex', FixtureIndex), \
                 patch.object(evaluate_rag, 'MultimodalIndex', FixtureIndex), \
                 patch('rag_model.RagVisionLanguageModel', FixtureModel), \
                 patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                generate_submission.main()
                published = publish_submission(root)
                evaluate_rag.evaluate(data, output / 'evaluation', golden)
                archive = export(root)
            with published.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 40)
            self.assertTrue(all(row['figure_references'] == 'NONE' for row in rows))
            metrics = json.loads((output / 'evaluation/evaluation.json').read_text())
            self.assertEqual({r['system'] for r in metrics['results']}, {'dense_baseline','hybrid_multimodal'})
            self.assertTrue(all(r['model_judged_citation_accuracy'] == 0 for r in metrics['results']),
                            'The fixture judge chose false; validation must not fabricate positive support.')
            self.assertTrue(all(r['human_groundedness'] is None for r in metrics['results']))
            self.assertFalse(metrics['golden_all_team_verified'])
            self.assertEqual(sum(FixtureIndex.searches), 25)
            self.assertEqual(len(FixtureIndex.searches), 90)
            judges = [s for s in FixtureModel.calls if s and 'judgments' in s['properties']]
            self.assertEqual(len(judges), 50)
            with zipfile.ZipFile(archive) as bundle:
                self.assertIsNone(bundle.testzip())
                self.assertIn('part1_rag/outputs/v2/submission.csv', bundle.namelist())
                self.assertIn('part1_rag/outputs/v2/evaluation/evaluation.json', bundle.namelist())


if __name__ == '__main__':
    unittest.main(verbosity=2)
