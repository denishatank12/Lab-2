"""Format regression tests. Tiny random weights and fictional evidence are not quality results."""
import copy
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'part1_rag/src')]
from response_schemas import rag_schema, VISION_SCHEMA, CRITIQUE_SCHEMA, citation_judge_schema
from vlm_runtime import VisionLanguageModel, response_parser
from multimodal import Conversation

HITS = [dict(citation='DOC1:Figure 1-1', doc_id='DOC1', page=1, figure_id='Figure 1-1')]
ANSWER = dict(status='answered', answer='Fictional format test only.',
              citations=['DOC1:p1'], figure_references=['Figure 1-1'])


def accepts(schema, value):
    parser = response_parser(schema)
    for ch in json.dumps(value, ensure_ascii=False, separators=(',', ':')):
        if ch not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(ch)
    return parser.can_end()


class SchemaTests(unittest.TestCase):
    def test_free_text_shortcut_is_kept_for_large_qwen_vocabulary(self):
        parser = response_parser(rag_schema(HITS))
        for ch in '{"status":"answered","answer":"':
            self.assertIn(ch, parser.get_allowed_characters())
            parser = parser.add_character(ch)
        self.assertEqual(parser.shortcut_key()[0], 'json_freetext')

    def test_rag_fields_and_exact_sources_are_accepted(self):
        self.assertTrue(accepts(rag_schema(HITS), ANSWER))

    def test_reported_paragraph_under_text_is_rejected_during_decoding(self):
        self.assertFalse(accepts(rag_schema(HITS), {'text': 'Copied fictional evidence.'}))

    def test_all_required_statuses_remain_model_choices(self):
        for status in ['answered', 'not_found', 'conflict', 'false_premise', 'session_override']:
            self.assertTrue(accepts(rag_schema(HITS), dict(ANSWER, status=status)))

    def test_invalid_or_missing_status_is_not_permitted(self):
        self.assertFalse(accepts(rag_schema(HITS), dict(ANSWER, status='success')))
        bad = dict(ANSWER)
        bad.pop('status')
        self.assertFalse(accepts(rag_schema(HITS), bad))

    def test_objects_cannot_replace_citation_strings(self):
        self.assertFalse(accepts(rag_schema(HITS), dict(ANSWER, citations=[{'source': 'DOC1:p1'}])))

    def test_unprovided_sources_and_figures_cannot_be_generated(self):
        self.assertFalse(accepts(rag_schema(HITS), dict(ANSWER, citations=['DOC3:p99'])))
        self.assertFalse(accepts(rag_schema(HITS), dict(ANSWER, figure_references=['Figure C-1'])))

    def test_no_figure_evidence_requires_empty_reference_list(self):
        hits = [dict(HITS[0], figure_id='')]
        self.assertTrue(accepts(rag_schema(hits), dict(ANSWER, figure_references=[])))
        self.assertFalse(accepts(rag_schema(hits), ANSWER))

    def test_answer_length_and_json_escapes(self):
        self.assertFalse(accepts(rag_schema(HITS), dict(ANSWER, answer='x' * 601)))
        self.assertTrue(accepts(rag_schema(HITS), dict(ANSWER, answer='A "quoted" fictional word.')))

    def test_extra_answer_lookup_fields_are_not_allowed(self):
        self.assertFalse(accepts(rag_schema(HITS), dict(ANSWER, expected_answer='Fixture')))

    def test_judge_has_one_required_entry_per_actual_citation(self):
        schema = citation_judge_schema(['DOC1:p1', 'DOC2:p2'])
        value = {'judgments': {'DOC1:p1': {'supported': True, 'reason': 'Fixture.'},
                                'DOC2:p2': {'supported': False, 'reason': 'Fixture.'}}}
        self.assertTrue(accepts(schema, value))
        del value['judgments']['DOC2:p2']
        self.assertFalse(accepts(schema, value))

    def test_agent_booleans_are_not_strings(self):
        good = dict(safe=True, accept=False, judgment='Fixture.', revision_instruction='', visible_text='')
        self.assertTrue(accepts(CRITIQUE_SCHEMA, good))
        self.assertFalse(accepts(CRITIQUE_SCHEMA, dict(good, safe='false')))

    def test_vision_question_fields_are_required(self):
        good = dict(unsafe=False, caption='Fixture.', answer='Fixture.', objects=['object'], visible_text='',
                    visual_questions={'What main objects are visible?': 'Fixture.',
                                      'Where is the main subject?': 'Fixture.'}, uncertainty='Fixture.')
        self.assertTrue(accepts(VISION_SCHEMA, good))
        good['visual_questions'].pop('Where is the main subject?')
        self.assertFalse(accepts(VISION_SCHEMA, good))


class NotebookRecoveryTests(unittest.TestCase):
    def recovery(self):
        tree = ast.parse((ROOT / 'tools/update_final_notebooks.py').read_text())
        return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'SCHEMA_RECOVERY' for t in n.targets))

    def test_old_local_patches_and_results_are_not_reset(self):
        class Commands:
            def __init__(self): self.calls = []
            def run(self, command, **kwargs):
                self.calls.append(command)
                sibling = Path(command[-1])
                (sibling / 'tools').mkdir(parents=True)
                (sibling / 'tools/response_schemas.py').write_text('# Temporary checkout fixture')
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / 'Lab-2'
            old.mkdir()
            patch = old / 'local_patch.py'
            patch.write_text('# User-owned fixture; preserve')
            result = old / 'result.json'
            result.write_text('{"fixture":true}')
            commands = Commands()
            env = {'ROOT': old, 'subprocess': commands}
            with redirect_stdout(io.StringIO()): exec(self.recovery(), env)
            self.assertEqual(patch.read_text(), '# User-owned fixture; preserve')
            self.assertEqual(result.read_text(), '{"fixture":true}')
            self.assertEqual(env['ROOT'].name, 'Lab-2-structured-v1')
            self.assertEqual(commands.calls[0][1], 'clone')
            self.assertEqual(len(commands.calls), 1)

    def test_current_checkout_does_not_clone_again(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'tools').mkdir()
            (root / 'tools/response_schemas.py').write_text('# Fixture')
            env = {'ROOT': root}
            exec(self.recovery(), env)
            self.assertEqual(env['ROOT'], root)

    def test_part1_has_the_real_generation_and_evaluation_cells(self):
        notebook = json.loads((ROOT / 'part1_rag/notebooks/Part1_RAG_Run.ipynb').read_text())
        sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertTrue(any('run_script("part1_rag/src/generate_submission.py"' in s for s in sources))
        self.assertTrue(any('run_script("part1_rag/src/evaluate_rag.py"' in s for s in sources))
        self.assertFalse(any('mm_path.write_text' in s for s in sources))


class TinyActualVisionGenerationTest(unittest.TestCase):
    def test_same_qwen_model_family_generates_schema_with_actual_image_forward(self):
        import torch
        from PIL import Image
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        from transformers import (PreTrainedTokenizerFast, BatchEncoding,
                                  Qwen2VLConfig, Qwen2VLForConditionalGeneration)
        torch.manual_seed(266)
        alphabet = ['<pad>', '<bos>', '<eos>', '<unk>', '<image>', '<vision_start>', '<vision_end>']
        alphabet += [chr(i) for i in range(32, 127)]
        backend = Tokenizer(models.WordLevel({s: i for i, s in enumerate(alphabet)}, unk_token='<unk>'))
        backend.pre_tokenizer = pre_tokenizers.Split('', behavior='isolated')
        backend.decoder = decoders.Fuse()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', bos_token='<bos>',
            eos_token='<eos>', unk_token='<unk>', additional_special_tokens=alphabet[4:7])
        config = Qwen2VLConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
            image_token_id=4, video_token_id=3, vision_start_token_id=5, vision_end_token_id=6,
            vision_config=dict(depth=1, embed_dim=32, hidden_size=32, mlp_ratio=2,
                               num_heads=4, patch_size=14, spatial_merge_size=2, temporal_patch_size=2),
            rope_scaling=dict(type='mrope', mrope_section=[1, 1, 2]))
        config._attn_implementation = 'sdpa'
        class Processor:
            def __init__(self): self.tokenizer = tokenizer
            def apply_chat_template(self, *args, **kwargs): return 'Fictional fixture prompt.'
            def __call__(self, **kwargs):
                ids = torch.tensor([[1, 5, 4, 4, 4, 4, 6, 15]])
                return BatchEncoding(dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                    pixel_values=torch.randn(16, 1176), image_grid_thw=torch.tensor([[1, 4, 4]])))
            def batch_decode(self, *args, **kwargs): return tokenizer.batch_decode(*args, **kwargs)
        model = VisionLanguageModel(require_cuda=False)
        model.model = Qwen2VLForConditionalGeneration(config).eval()
        model.processor = Processor()
        schema = copy.deepcopy(rag_schema(HITS))
        schema['properties']['answer'] = {'type': 'string', 'enum': ['Synthetic format fixture.']}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fictional.png'
            Image.new('RGB', (56, 56), 'blue').save(path)
            answer = model.json('Fixture only.', images=[path], schema=schema, max_new_tokens=256)
        self.assertEqual(answer['answer'], 'Synthetic format fixture.')
        Conversation.validate(answer, HITS)


if __name__ == '__main__':
    unittest.main(verbosity=2)
