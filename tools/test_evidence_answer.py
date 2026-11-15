"""Fictional software regressions. These do not predict a leaderboard score."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'part1_rag/src')]
from evidence_answer import (evidence_schema, make_instruction, validate_evidence_answer,
                             table_lines, answer_from_evidence)
from evidence_review import evidence_packet
from multimodal import Conversation
from run_checkpoint import experiment_sources


HIT = dict(citation='DOC9:Table Z-42', doc_id='DOC9', page=9,
           figure_id='Table Z-42', image_path='', trust_level='high',
           caption='Fictional district rates, percent.', text='Not a competition source.',
           table_grid=[['District', 'Metric', '2041', '2042'],
                       ['Lark', 'Rate A', '5.3', '4.8'], ['', 'Rate B', '8.2', '8.0']])


def result(status='answered', source=HIT['citation'], figures=None):
    value = dict(observations=[dict(source=source, finding='Lark Rate B, 2042: 8.0 percent.')],
                 answer='Lark Rate B is 8.0% in 2042.', status=status)
    if figures is not None:
        value['figure_references'] = figures
    return value


class EvidencePolicyTests(unittest.TestCase):
    def test_schema_observes_sources_then_answers_then_classifies(self):
        schema = evidence_schema([HIT])
        self.assertEqual(list(schema['properties']),
                         ['observations', 'answer', 'status'])
        self.assertNotIn('citations', schema['properties'])
        self.assertNotIn('figure_references', schema['properties'])
        self.assertIn('session_override', schema['properties']['status']['enum'])
        self.assertEqual(schema['properties']['observations']['maxItems'], 4)

    def test_prompt_retains_user_context_but_never_prior_model_claims(self):
        packet = evidence_packet([HIT], {}, 'Compare the two rates.')
        history = [dict(user='What does Lark report in 2042?',
                        assistant=dict(answer='WRONG_ASSISTANT_NUMBER', status='answered'))]
        instruction = make_instruction('Now assume Rate A is 4.0%.', packet, [HIT],
            ['Assume Rate A is 4.5%.'], history, 'add', [])
        self.assertNotIn('WRONG_ASSISTANT_NUMBER', instruction)
        self.assertNotIn('"status": "answered"', instruction)
        self.assertIn('What does Lark report', instruction)
        self.assertLess(instruction.index('Assume Rate A is 4.5%'), instruction.rindex('Rate A is 4.0%'))
        self.assertIn('latest update wins', instruction)
        self.assertNotIn('golden_candidate', instruction)

    def test_table_repeats_no_numeric_matrix_and_preserves_entity_carry(self):
        formatted = table_lines(HIT['table_grid'])
        self.assertIn('District: Lark | Metric: Rate B | 2041: 8.2 | 2042: 8.0', formatted)
        self.assertEqual(formatted.count('8.2'), 1)
        self.assertIn('2041', formatted)

    def test_exact_object_citation_adds_only_its_metadata_label(self):
        other = dict(HIT, citation='DOC9:Table Z-43', figure_id='Table Z-43')
        value = validate_evidence_answer(result(), [HIT, other], Conversation.validate)
        self.assertEqual(value['figure_references'], ['Table Z-42'])
        self.assertEqual(value['answer'], result()['answer'])
        self.assertEqual(value['status'], 'answered')

    def test_unused_figure_and_invented_sources_are_rejected(self):
        other = dict(HIT, citation='DOC9:Table Z-43', figure_id='Table Z-43')
        with self.assertRaisesRegex(ValueError, 'derives citation/figure'):
            validate_evidence_answer(result(figures=['Table Z-43']), [HIT, other], Conversation.validate)
        with self.assertRaises(ValueError):
            validate_evidence_answer(result(source='DOC8:invented'), [HIT], Conversation.validate)

    def test_missing_observation_does_not_silently_pass(self):
        value = result()
        value['observations'] = []
        with self.assertRaisesRegex(ValueError, 'needs a citation'):
            validate_evidence_answer(value, [HIT], Conversation.validate)

    def test_empty_evidence_can_honestly_return_not_found(self):
        raw = dict(observations=[], answer='No supporting source was retrieved.',
                   status='not_found')
        self.assertEqual(validate_evidence_answer(raw, [], Conversation.validate)['status'], 'not_found')
        schema = evidence_schema([])
        self.assertEqual(schema['properties']['observations']['maxItems'], 0)

    def test_validator_never_invents_semantic_corrections_or_a_score(self):
        for status in ['answered', 'session_override', 'not_found', 'conflict', 'false_premise']:
            value = validate_evidence_answer(result(status), [HIT], Conversation.validate)
            self.assertEqual(value['status'], status)
            self.assertNotIn('score', value)

    def test_first_call_and_repair_both_use_source_bounded_observation_schema(self):
        class Model:
            def __init__(self): self.calls = []
            def json(self, instruction, **kwargs):
                self.calls.append(kwargs)
                return result(source='DOC8:invented') if len(self.calls) == 1 else result()
        model = Model()
        with redirect_stdout(io.StringIO()):
            value, audit = answer_from_evidence(model, 'Read Lark Rate B.',
                evidence_packet([HIT], {}, 'Read Lark Rate B.'), [HIT], [], [], 'none', [], Conversation.validate)
        self.assertEqual(value['figure_references'], ['Table Z-42'])
        self.assertIn('observations', model.calls[0]['schema']['properties'])
        self.assertEqual(model.calls[0]['schema'], model.calls[1]['schema'])
        self.assertIn('observations', model.calls[1]['schema']['properties'])
        self.assertIn('not independently certified', audit['meaning'])

    def test_conversation_keeps_model_observations_as_audit_and_assumptions_separate(self):
        class Index:
            ingestion = {}
            def search(self, *args, **kwargs): return [HIT]
        class Model:
            def json(self, instruction, **kwargs):
                self.instruction = instruction
                return result('session_override')
        model = Model()
        with patch.dict(os.environ, LAB2_RAG_EVIDENCE_FIRST='1', LAB2_RAG_EVIDENCE_REVIEW='0'):
            chat = Conversation(Index(), model)
            value = chat.answer('Assume Lark Rate B is 6.0% for this session.')
            self.assertEqual(value['status'], 'session_override')
            self.assertEqual(chat.last_audit['protocol'], 'evidence-first-v8')
            self.assertEqual(len(chat.session_facts), 1)
            chat.answer('What does that imply for the difference?')
            self.assertIn('Assume Lark Rate B is 6.0%', model.instruction)
            self.assertEqual(len(chat.session_facts), 1)
            chat.answer('Forget my assumptions.')
            self.assertEqual(chat.session_facts, [])

    def test_options_and_new_policy_are_checkpoint_bound(self):
        with patch.dict(os.environ, LAB2_RAG_EVIDENCE_FIRST='0'):
            before = experiment_sources('part1_rag')
        with patch.dict(os.environ, LAB2_RAG_EVIDENCE_FIRST='1'):
            after = experiment_sources('part1_rag')
        self.assertNotEqual(before, after)
        self.assertIn('part1_rag/src/evidence_answer.py', after)

    def test_reported_mismatch_is_impossible_in_new_three_field_contract(self):
        other = dict(HIT, citation='DOC9:Figure Z-43', figure_id='Figure Z-43')
        old_shape = dict(result(), citations=[HIT['citation']], figure_references=[other['figure_id']])
        with self.assertRaisesRegex(ValueError, 'Generate only'):
            validate_evidence_answer(old_shape, [HIT, other], Conversation.validate)
        # The same evidence selection under the new contract has exactly its
        # own object citation and label, not a silently dropped invalid list.
        value = validate_evidence_answer(result(), [HIT, other], Conversation.validate)
        self.assertEqual(value['citations'], [HIT['citation']])
        self.assertEqual(value['figure_references'], [HIT['figure_id']])

    def test_page_observation_never_invents_a_figure_and_shared_labels_deduplicate(self):
        page = dict(HIT, citation='DOC9:p10', page=10, figure_id='')
        value = validate_evidence_answer(result(source=page['citation']), [HIT, page], Conversation.validate)
        self.assertEqual(value['figure_references'], [])
        self.assertEqual(value['citations'], [page['citation']])
        raw = result()
        raw['observations'].append(dict(raw['observations'][0]))
        value = validate_evidence_answer(raw, [HIT], Conversation.validate)
        self.assertEqual(value['citations'], [HIT['citation']])
        self.assertEqual(value['figure_references'], [HIT['figure_id']])


if __name__ == '__main__':
    unittest.main(verbosity=2)
