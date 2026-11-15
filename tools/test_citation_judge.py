"""Fictional regressions for the actual v5 citation-judge failure."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from citation_judge import judge_citations, validate_judgments

ANSWER = dict(answer='Fictional fixture answer.', citations=['DOC1:p1', 'DOC2:p2'])
GOOD = dict(judgments={'DOC1:p1':dict(supported=True, reason='Fictional support.'),
                       'DOC2:p2':dict(supported=False, reason='Fictional contradiction.')})


class Model:
    def __init__(self, responses): self.responses, self.calls = responses, []
    def json(self, instruction, **kwargs):
        self.calls.append(kwargs)
        value = self.responses[min(len(self.calls)-1, len(self.responses)-1)]
        if isinstance(value, Exception): raise value
        return copy.deepcopy(value)


class JudgeTests(unittest.TestCase):
    def test_natural_answer_flag_does_not_disable_judge_constraints(self):
        with patch.dict(os.environ, LAB2_RAG_NATURAL_JSON='1'):
            model = Model([GOOD])
            rows = judge_citations(model, ANSWER, [])
        expected = model.calls[0]['schema']['properties']['judgments']['properties']
        self.assertEqual(set(expected), set(ANSWER['citations']))
        self.assertEqual([r['supported'] for r in rows], [True, False])

    def test_reported_missing_judgments_retry_with_exact_keys(self):
        model = Model([{'citations':[]}, GOOD])
        rows = judge_citations(model, ANSWER, [{'image_path':'fixture.png'}])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]['schema'], model.calls[1]['schema'])
        self.assertEqual(model.calls[1]['images'], ['fixture.png'])
        self.assertEqual({r['source'] for r in rows}, set(ANSWER['citations']))

    def test_missing_extra_and_string_booleans_are_not_accepted(self):
        for values in [{}, {**GOOD['judgments'], 'DOC3:p3':{}},
                       {'DOC1:p1':dict(supported='true',reason='Fixture.'),
                        'DOC2:p2':GOOD['judgments']['DOC2:p2']}]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_judgments(dict(judgments=values), ANSWER['citations'])

    def test_bad_reasons_are_not_filled_in_by_python(self):
        for reason in [None, '', 'x'*101]:
            value = copy.deepcopy(GOOD)
            value['judgments']['DOC1:p1']['reason'] = reason
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                validate_judgments(value, ANSWER['citations'])

    def test_no_citations_spends_no_model_call(self):
        model = Model([])
        self.assertEqual(judge_citations(model, dict(citations=[]), []), [])
        self.assertEqual(model.calls, [])

    def test_persistent_invalid_output_is_not_a_fabricated_score(self):
        model = Model([dict(judgments={})])
        with self.assertRaisesRegex(ValueError, 'after 2 attempts'):
            judge_citations(model, ANSWER, [])
        self.assertEqual(len(model.calls), 2)

    def test_runtime_failure_is_not_caught_as_a_boolean_judgment(self):
        model = Model([RuntimeError('Fictional CUDA error')])
        with self.assertRaisesRegex(RuntimeError, 'CUDA error'):
            judge_citations(model, ANSWER, [])
        self.assertEqual(len(model.calls), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
