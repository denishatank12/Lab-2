"""Fictional audit-control regressions, not evidence of a leaderboard score."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'part1_rag/src')]
from evidence_review import (evidence_packet, focused_text, needs_review,
                             session_memory_action, validate_review, quote_text)
from multimodal import Conversation, MultimodalIndex

HIT = dict(citation='DOC8:Table A-99', doc_id='DOC8', page=4, figure_id='Table A-99',
           text='Fictional growth is 7.4 percent in 2042.', trust_level='high', image_path='')
ANSWER = dict(answer='Fictional growth is 7.4 percent in 2042.', status='answered',
              citations=['DOC8:Table A-99'], figure_references=['Table A-99'])


def checked(answer=None):
    return dict(copy.deepcopy(answer or ANSWER), checks=[dict(source='DOC8:Table A-99',
        observation='The source gives 7.4 percent.', modality='text',
        quote='Fictional growth is 7.4 percent in 2042.')])


class PacketTests(unittest.TestCase):
    def test_duplicate_pages_are_sent_once_and_metadata_is_preserved(self):
        ingestion = dict(page_context={'DOC8:p4': 'Full fictional page and footnote.'},
                         documents={'DOC8': {'data_vintage': '2043'}})
        packet = evidence_packet([HIT, dict(HIT, citation='DOC8:p4', figure_id='')], ingestion, 'Growth?')
        self.assertEqual(len(packet['page_context']), 1)
        self.assertEqual(packet['evidence'][0]['document_metadata']['data_vintage'], '2043')

    def test_named_figure_does_not_receive_unrelated_page_plots(self):
        packet = evidence_packet([dict(HIT, figure_id='Figure A-99')],
            dict(page_context={'DOC8:p4': 'Unrelated adjacent chart.'}), 'Read Figure A-99.', {'figure a-99'})
        self.assertEqual(packet['page_context'], [])

    def test_late_relevant_lines_are_kept_without_reordering_or_inventing_values(self):
        lines = ['Header'] + ['Irrelevant prose line.'] * 80 + ['Fictional revised datum: 7.4 percent.', 'End note.']
        result = focused_text('\n'.join(lines), 'What is the revised fictional datum?', 500)
        self.assertIn('7.4 percent', result)
        self.assertLessEqual(len(result), 500)

    def test_large_packet_has_bounded_source_text_without_cutting_each_page_repeatedly(self):
        hits = [dict(HIT, page=i, citation=f'DOC8:p{i}', text=('Fictional data.\n' * 1000)) for i in range(1, 6)]
        ingestion = dict(page_context={f'DOC8:p{i}': 'Fictional source line.\n' * 1000 for i in range(1, 6)})
        packet = evidence_packet(hits, ingestion, 'Fictional data')
        total = sum(len(r['text']) for r in packet['evidence'] + packet['page_context'])
        self.assertLessEqual(total, 22000)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.packet = evidence_packet([HIT], {}, 'Growth?')

    def validate(self, raw, images=()):
        return validate_review(raw, [HIT], self.packet, images, Conversation.validate)

    def test_exact_source_observation_is_retained(self):
        self.assertEqual(self.validate(checked())['answer'], ANSWER)

    def test_invented_or_paraphrased_quote_is_not_accepted_as_source_text(self):
        raw = checked()
        raw['checks'][0]['quote'] = 'Growth was really 9.8 percent.'
        with self.assertRaisesRegex(ValueError, 'not present'):
            self.validate(raw)

    def test_pdf_typography_is_normalized_without_erasing_a_minus_sign_or_digit(self):
        self.assertEqual(quote_text('The rate was \u22122.4\npercent.'), quote_text('The rate was -2.4 percent.'))
        self.assertNotEqual(quote_text('The rate was -2.4 percent.'), quote_text('The rate was 2.4 percent.'))
        self.assertNotEqual(quote_text('The rate was 2.4 percent.'), quote_text('The rate was 2.5 percent.'))

    def test_image_observation_requires_that_exact_attached_source(self):
        raw = checked()
        raw['checks'][0]['modality'] = 'image'
        with self.assertRaisesRegex(ValueError, 'actually attached'):
            self.validate(raw)
        self.assertEqual(self.validate(raw, ['DOC8:Table A-99'])['answer'], ANSWER)

    def test_unsupported_answer_can_be_not_found_without_fabricated_checks(self):
        raw = dict(answer='The requested fact is absent.', status='not_found', citations=[], figure_references=[], checks=[])
        self.assertEqual(self.validate(raw)['answer']['status'], 'not_found')

    def test_answered_with_no_sources_is_rejected(self):
        raw = dict(answer='Invented fictional value.', status='answered', citations=[], figure_references=[], checks=[])
        with self.assertRaisesRegex(ValueError, 'supporting citation'):
            self.validate(raw)

    def test_object_citation_and_figure_reference_cannot_contradict_each_other(self):
        raw = checked()
        raw['figure_references'] = []
        with self.assertRaisesRegex(ValueError, 'also be listed'):
            self.validate(raw)

    def test_final_hybrid_result_is_reviewed_and_saved_but_dense_baseline_is_not(self):
        class Index:
            def search(self, *args, **kwargs): return [copy.deepcopy(HIT)]
        class Model:
            def __init__(self): self.calls = 0
            def json(self, instruction, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return dict(ANSWER, answer='Fictional growth is 9.8 percent in 2042.')
                return checked()
        with patch.dict(os.environ, LAB2_RAG_NATURAL_JSON='1', LAB2_RAG_EVIDENCE_REVIEW='1'):
            model = Model()
            chat = Conversation(Index(), model)
            self.assertEqual(chat.answer('What is growth in 2042?'), ANSWER)
            self.assertEqual(model.calls, 2)
            self.assertTrue(chat.last_audit['changed'])
            self.assertEqual(chat.history[-1]['assistant'], ANSWER)
            model = Model()
            chat = Conversation(Index(), model, dense_only=True)
            chat.answer('What is growth in 2042?')
            self.assertEqual(model.calls, 1)
            self.assertIsNone(chat.last_audit)

    def test_failed_audit_does_not_silently_return_the_draft(self):
        class Index:
            def search(self, *args, **kwargs): return [copy.deepcopy(HIT)]
        class Model:
            def json(self, *args, **kwargs): return copy.deepcopy(ANSWER)
        with patch.dict(os.environ, LAB2_RAG_EVIDENCE_REVIEW='1'), self.assertRaisesRegex(ValueError, 'after 2 attempts'):
            Conversation(Index(), Model()).answer('What is growth in 2042?')

    def test_simple_definition_does_not_spend_a_second_call(self):
        raw = dict(answer='A fictional definition.', status='answered', citations=['DOC8:p4'], figure_references=[])
        self.assertFalse(needs_review('Define the fictional term.', raw, [], []))
        self.assertTrue(needs_review('What is its revised value?', raw, [], []))


class MemoryTests(unittest.TestCase):
    def test_economic_change_is_not_a_user_assumption(self):
        self.assertEqual(session_memory_action('How did inflation change?'), 'none')
        self.assertEqual(session_memory_action('Do not assume any new values.'), 'none')
        self.assertEqual(session_memory_action('Assume the fictional rate is 7%.'), 'add')
        self.assertEqual(session_memory_action('Change our chat assumption to 8%.'), 'add')
        self.assertEqual(session_memory_action('Please forget my earlier assumption.'), 'clear')

    def test_forgetting_assumption_clears_only_that_conversation(self):
        class Index:
            def search(self, *args, **kwargs): return [copy.deepcopy(HIT)]
        class Model:
            def json(self, instruction, **kwargs):
                self.packet = json.loads(instruction.split('DATA:\n', 1)[1])
                return copy.deepcopy(ANSWER)
        first, second = Conversation(Index(), Model()), Conversation(Index(), Model())
        first.answer('Assume the fictional rate is 7%.')
        second.answer('Assume the fictional rate is 8%.')
        first.answer('Forget my earlier assumption.')
        self.assertEqual(first.session_facts, [])
        self.assertEqual(first.model.packet['session_facts'], [])
        self.assertEqual(len(second.session_facts), 1)

    def test_followup_reopens_cited_table_without_copying_a_user_assumption_into_the_kb(self):
        original = copy.deepcopy(HIT)
        class Index:
            objects = [original]
            records = [original]
            def __init__(self): self.calls = 0
            def search(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1: return [copy.deepcopy(original)]
                return [dict(HIT, citation='DOC9:p8', doc_id='DOC9', page=8,
                             figure_id='', text='An unrelated fictional source.')]
        class Model:
            def json(self, instruction, **kwargs):
                self.payload = json.loads(instruction.split('DATA:\n', 1)[1])
                return copy.deepcopy(ANSWER)
        index, model = Index(), Model()
        chat = Conversation(index, model)
        chat.answer('Assume the fictional growth rate in 2042 is 9.2%.')
        chat.answer('What is its difference from the document rate?')
        self.assertIn('DOC8:Table A-99', [r['citation'] for r in chat.last_hits])
        self.assertNotIn('9.2', original['text'])
        self.assertEqual(index.records, [original])
        self.assertLessEqual(len(chat.last_hits), 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
