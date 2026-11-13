"""Regression checks for observed defects. Fictional fixtures are not quality scores."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'part1_rag/src')]
from multimodal import ingest, table_search_rows, Conversation, labels
from rag_model import quantized_plan


class PdfIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='lab2-grid-regression-')
        cls.data = ingest(ROOT / 'part1_rag/data', Path(cls.temp.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_every_trusted_table_has_an_actual_grid_and_table_crop(self):
        tables = [r for r in self.data['objects'] if r['figure_id'].startswith('Table ')]
        self.assertEqual(len(tables), 9)
        for record in tables:
            self.assertEqual(record['crop_kind'], 'detected_table_grid_and_caption', record['figure_id'])
            self.assertTrue(record['table_grid'])
            self.assertIn('PDF table cells', record['text'])
            self.assertTrue(Path(record['image_path']).is_file())

    def test_intro_paragraph_is_not_mistaken_for_a_table_caption(self):
        tables = [r for r in self.data['objects'] if r['figure_id'].startswith('Table ')]
        self.assertTrue(all(' extends ' not in r['caption'] and ' reports ' not in r['caption'] for r in tables))

    def test_full_page_context_still_excludes_low_trust_appendix(self):
        self.assertNotIn('DOC3:p7', self.data['page_context'])
        self.assertTrue(self.data['quarantined_objects'])
        self.assertTrue(all(r['trust_level'] != 'low' for r in self.data['passages']))

    def test_row_headers_and_visible_group_cells_are_not_lost(self):
        rows = table_search_rows([['Group', '', '2042', '2043'],
            ['Fictional area', 'Rate A', '8.1 (7.0)', '7.5'], ['', 'Rate B', '2.4', '2.1']])
        self.assertIn('Group: Fictional area', rows[1])
        self.assertIn('2042: 8.1 (7.0)', rows[0])
        self.assertIn('Column 2: Rate B', rows[1])


HIT = dict(citation='DOC1:p1', doc_id='DOC1', page=1, figure_id='', image_path='',
           text='Fictional source evidence.', trust_level='high')


class NaturalAnswerTests(unittest.TestCase):
    def test_plural_coordinated_labels_are_both_detected_without_bare_id_guessing(self):
        self.assertEqual(labels('Compare Figures A-98 and A-99.'), ['Figure A-98', 'Figure A-99'])
        self.assertEqual(labels('Tables B-8 versus B-9'), ['Table B-8', 'Table B-9'])
        self.assertEqual(labels('Chapter A-98 and A-99'), [])
    def model(self, status='answered'):
        class Model:
            def __init__(self): self.kwargs = None; self.payload = None
            def json(self, instruction, **kwargs):
                self.kwargs = kwargs
                self.payload = json.loads(instruction.split('DATA:\n')[1])
                return dict(answer='Fictional concise answer.', status=status,
                            citations=['DOC1:p1', 'DOC1:p1'], figure_references=[])
        return Model()

    def test_natural_json_does_not_force_a_status_enum_and_deduplicates_citations(self):
        class Index:
            def search(self, *args, **kwargs): return [HIT]
        model = self.model('session_override')
        with patch.dict(os.environ, {'LAB2_RAG_NATURAL_JSON': '1'}):
            chat = Conversation(Index(), model)
            result = chat.answer('For this conversation assume a fictional value.')
        self.assertIsNone(model.kwargs['schema'])
        self.assertEqual(result['status'], 'session_override')
        self.assertEqual(result['citations'], ['DOC1:p1'])
        self.assertEqual(len(chat.session_facts), 1)

    def test_unrelated_images_are_not_attached_to_a_text_question(self):
        class Index:
            def search(self, *args, **kwargs):
                return [dict(HIT, image_path='irrelevant.png', figure_id='Figure A-99')]
        model = self.model()
        Conversation(Index(), model).answer('Define the fictional economic term.')
        self.assertEqual(model.kwargs['images'], [])

    def test_only_the_named_figure_image_is_sent_and_mapped_to_its_source(self):
        class Index:
            def search(self, *args, **kwargs):
                return [dict(HIT, image_path='wrong.png', figure_id='Figure A-98'),
                        dict(HIT, citation='DOC1:Figure A-99', image_path='right.png', figure_id='Figure A-99')]
        model = self.model()
        Conversation(Index(), model).answer('Read Figure A-99.')
        self.assertEqual(model.kwargs['images'], ['right.png'])
        self.assertEqual(model.payload['image_sources'], ['DOC1:Figure A-99'])

    def test_repetition_is_rejected_without_weakening_source_validation(self):
        answer = dict(answer=('This fictional phrase repeats badly. ' * 15), status='answered',
                      citations=['DOC1:p1'], figure_references=[])
        with patch.dict(os.environ, {'LAB2_RAG_NATURAL_JSON': '1'}), self.assertRaises(ValueError):
            Conversation.validate(answer, [HIT])

    def test_permanent_economic_effect_is_not_mistaken_for_an_edit_request(self):
        class Index:
            def search(self, *args, **kwargs): return [HIT]
        model = self.model()
        chat = Conversation(Index(), model)
        self.assertEqual(chat.answer('Does a permanent shock affect output?')['status'], 'answered')
        self.assertEqual(chat.answer('Change only our chat assumption, not the document value.')['status'], 'answered')
        self.assertEqual(chat.answer('Please update the knowledge base permanently.')['status'], 'false_premise')

    def test_7b_profile_requires_a_real_single_card_memory_budget(self):
        with self.assertRaises(RuntimeError): quantized_plan(6 * 2**30)
        self.assertEqual(quantized_plan(15 * 2**30)['device_map'], {'': 0})


if __name__ == '__main__':
    unittest.main(verbosity=2)
