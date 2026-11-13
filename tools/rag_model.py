"""Part 1's larger local model, independent of the Part 3 model configuration."""
import random
from pathlib import Path

from vlm_runtime import VisionLanguageModel

MODEL = 'Qwen/Qwen2-VL-7B-Instruct'
REVISION = 'eed13092ef92e448dd6875b2a00151bd3f7db0ac'


def quantized_plan(total_bytes):
    if total_bytes < 13 * 2**30:
        raise RuntimeError('The 7B Part 1 profile needs one nominal 16-GB GPU. Select a Kaggle T4/P100; two smaller cards are not pooled.')
    return dict(device_map={'': 0}, max_pixels=768 * 768)


class RagVisionLanguageModel(VisionLanguageModel):
    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError('The real Part 1 7B model needs CUDA. CPU fixture tests are not a GPU model run.')
        plan = quantized_plan(torch.cuda.get_device_properties(0).total_memory)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION,
            min_pixels=56 * 56, max_pixels=plan['max_pixels'])
        config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=['visual', 'lm_head'])
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL, revision=REVISION,
            quantization_config=config, device_map=plan['device_map'], torch_dtype=torch.float16,
            attn_implementation='sdpa', low_cpu_mem_usage=True).eval()
        if not getattr(self.model, 'is_loaded_in_4bit', False):
            raise RuntimeError('The 7B model was not loaded in 4-bit. Refusing an unsafe full-precision fallback.')
        self.profile = dict(model=MODEL, revision=REVISION, quantization='NF4 double quantization',
            visual_precision='float16', max_image_pixels=plan['max_pixels'],
            actual_device_map={k: str(v) for k, v in getattr(self.model, 'hf_device_map', {}).items()})
        print('Loaded Part 1 model:', self.profile, flush=True)
