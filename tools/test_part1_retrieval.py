"""Synthetic retrieval regressions. No competition answers or quality claims."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'part1_rag/src')]
from multimodal import (labels, retrieval_query, rerank_records, select_evidence, Conversation,
                        fuse_reranked, structured_table_evidence, content_query, record_key, mentioned_documents)
from rag_pipeline import _tokens
from generate_submission import conversation_turns


def hit(page, text, figure=''):
    return dict(citation=f'DOC1:p{page}', doc_id='DOC1', page=page, figure_id=figure,
                text=text, trust_level='high', image_path='')


class RetrievalTests(unittest.TestCase):
    def test_turn_order_uses_supplied_numbers_or_unchanged_csv_order(self):
        rows = [dict(id='b', turn='2'), dict(id='a', turn='1')]
        self.assertEqual([r['id'] for r in conversation_turns(rows)], ['a', 'b'])
        without = [dict(id='b'), dict(id='a')]
        self.assertEqual(conversation_turns(without), without)
        with self.assertRaises(ValueError):
            conversation_turns([dict(turn='1'), dict(turn='1')])

    def test_figure_typography_is_normalized_without_merging_corrections(self):
        self.assertEqual(labels('figure a–51, TABLE 2-7; Figure B-5 (corrected).'),
                         ['Figure A-51', 'Table 2-7', 'Figure B-5 (corrected)'])
        self.assertEqual(labels('Figure B-5 and Figure B-5 (corrected)'),
                         ['Figure B-5', 'Figure B-5 (corrected)'])

    def test_new_standalone_topic_does_not_inherit_old_chart_or_year(self):
        history = [dict(user='Read Figure A-1 in 1998.')]
        question = 'Read Figure A-2 in 2004.'
        self.assertEqual(retrieval_query(question, history), question)

    def test_followup_keeps_topic_but_current_chart_and_year_win(self):
        query = retrieval_query('And Figure A-2 in 2004?', [dict(user='Read Figure A-1 in 1998.')])
        self.assertIn('Previous conversation topic:', query)
        self.assertNotIn('Figure A-1', query)
        self.assertNotIn('1998', query)
        self.assertIn('Figure A-2', query)

    def test_pronoun_followup_and_session_memory_stay_separate(self):
        history = [dict(user='Assume the fictional country tax is 13%.')]
        self.assertIn('fictional country', retrieval_query('What is its implication?', history))
        self.assertEqual(retrieval_query('What is its implication?', []), 'What is its implication?')

    def test_an_assumption_update_retains_the_missing_original_entity(self):
        query = retrieval_query('Change our chat assumption to 14%. Give the unchanged document rate.',
            [dict(user='Assume the fictional country tax rate is 13%.')])
        self.assertIn('fictional country tax', query)

    def test_reranker_scores_pairs_and_preserves_source_records(self):
        records = [hit(1, 'A distractor.'), hit(2, 'Relevant fictional evidence.')]
        class Ranker:
            def predict(self, pairs, **kwargs):
                self.pairs = pairs
                return np.array([-1., 2.])
        ranker = Ranker()
        self.assertEqual(rerank_records('Fictional query', records, ranker), records[::-1])
        self.assertTrue(all(pair[0] == 'Fictional query' for pair in ranker.pairs))

    def test_reranker_cannot_silently_fail_open(self):
        class Ranker:
            def predict(self, pairs, **kwargs):
                return [float('nan')]
        with self.assertRaisesRegex(ValueError, 'invalid scores'):
            rerank_records('Fixture', [hit(1, 'Fixture')], Ranker())

    def test_named_objects_win_and_duplicate_page_context_is_removed(self):
        exact = hit(2, 'Fictional red blue green white', 'Figure A-99')
        duplicate = hit(2, 'Fictional red blue green white')
        other = hit(3, 'Other evidence from a different source page.')
        self.assertEqual(select_evidence([duplicate, other, exact], [exact], 2), [exact, other])

    def test_exact_chart_is_first_even_if_reranker_prefers_another(self):
        exact = hit(1, 'Named evidence.', 'Figure A-17')
        other = hit(2, 'Different evidence.', 'Figure A-18')
        self.assertEqual(select_evidence([other, exact], [exact], 2)[0], exact)

    def test_different_table_rows_with_the_same_long_caption_do_not_collide(self):
        caption = 'Fictional table caption ' * 8
        a, b = hit(1, caption + 'North value: 3.5'), hit(1, caption + 'South value: 9.2')
        self.assertNotEqual(record_key(a), record_key(b))
        self.assertEqual(len(fuse_reranked([a, b], [b, a], [a, b])), 2)

    def test_structured_table_replaces_its_flattened_page_not_another_page(self):
        flat = hit(4, 'Fictional output in 2042 was 7.4.')
        table = dict(hit(4, 'Fictional output table.', 'Table A-99'),
                     caption='Fictional output table', table_grid=[['Region', '2042'], ['North', '7.4']])
        other = hit(5, 'A prose definition unrelated to the table.')
        self.assertEqual(structured_table_evidence([flat, other], [table], 'North output 2042'), [table, other])

    def test_document_title_does_not_replace_the_query_topic(self):
        query = content_query('Compare North growth in Supplement Z and Briefing Y.', {
            'DOC8': {'title': 'Supplement Z: Fictional data'}, 'DOC9': {'title': 'Briefing Y: Fictional data'}})
        self.assertIn('North growth', query)
        self.assertNotIn('Supplement Z', query)
        self.assertNotIn('Briefing Y', query)

    def test_explicit_source_comparison_keeps_both_document_families(self):
        metadata = {'DOC8': {'title': 'Supplement Z: Fictional data'},
                    'DOC9': {'title': 'Briefing Y: Fictional data'}}
        self.assertEqual(mentioned_documents('Compare Supplement Z with Briefing Y.', metadata), ['DOC8', 'DOC9'])
        self.assertEqual(mentioned_documents('Read DOC8.', metadata), ['DOC8'])

    def test_numeric_constraints_are_preserved_and_initialisms_are_normalized(self):
        query = content_query('Assume U.S. growth was 6.2% in 2042.', {})
        self.assertIn('6.2', query)
        self.assertIn('2042', query)
        self.assertEqual(_tokens('U.S. GDP'), _tokens('US GDP'))

    def test_generic_source_name_is_used_only_when_unambiguous_in_the_manifest(self):
        metadata = {'DOC8': {'title': 'Supplement Z: Fictional data'},
                    'DOC9': {'title': 'Briefing Y: Fictional data'}}
        self.assertEqual(mentioned_documents('Read the trusted supplement.', metadata), ['DOC8'])
        metadata['DOC9'] = {'title': 'Supplement Y: Other data'}
        self.assertEqual(mentioned_documents('Read the supplement.', metadata), [])

    def test_conversation_searches_the_current_question_not_all_old_questions(self):
        class Index:
            def search(self, query, **kwargs):
                self.query = query
                return [hit(1, 'Fictional support.')]
        class Model:
            def json(self, instruction, **kwargs):
                payload = json.loads(instruction.split('DATA:\n')[1])
                return dict(status='answered', answer='Fictional support.',
                            citations=['DOC1:p1'], figure_references=[])
        index = Index()
        chat = Conversation(index, Model())
        chat.answer('Read Figure A-17.')
        chat.answer('Read Figure A-18.')
        self.assertEqual(index.query, 'Read Figure A-18.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
