"""Lazy CPU vision and offloaded img2img, with one diffusion pipeline at a time."""
from pathlib import Path
import math
import torch
import yaml
from PIL import Image


class ModelRuntime:
    def __init__(self, lora_path=None, seed=266):
        if not torch.cuda.is_available():
            raise RuntimeError('Full agent image generation requires a CUDA GPU. Run smoke tests on CPU.')
        self.lora_path = Path(lora_path) if lora_path else None
        self.seed = seed
        self.captioner = self.vqa = self.pipe = self.clip = self.processor = None
        self.adapter_loaded = False
        self.config = yaml.safe_load((Path(__file__).resolve().parents[2]/'part2_diffusion/config.yaml').read_text())

    def understand(self, path, question):
        from transformers import pipeline
        if self.captioner is None:
            self.captioner = pipeline('image-to-text', model='Salesforce/blip-image-captioning-base', device=-1)
        if self.vqa is None:
            self.vqa = pipeline('visual-question-answering', model='dandelin/vilt-b32-finetuned-vqa', device=-1)
        with Image.open(path) as im:
            image = im.convert('RGB')
        caption = self.captioner(image, max_new_tokens=40)[0]['generated_text']
        questions = ['What objects are visible?', 'Where is the main subject?']
        answers = {q: self.vqa(image=image, question=q, top_k=1)[0]['answer'] for q in questions}
        visible_text = self.vqa(image=image, question='What text is visible?', top_k=1)[0]['answer']
        answer = caption if 'caption' in question.lower() else self.vqa(image=image, question=question, top_k=1)[0]['answer']
        return {'caption': caption, 'objects': answers[questions[0]], 'visible_text': visible_text,
                'text_method': 'VQA guess, not verified OCR', 'visual_questions': answers, 'answer': answer}

    def generate(self, state):
        from diffusers import StableDiffusionImg2ImgPipeline
        if state.route == 'domain_style' and (self.lora_path is None or
              not (self.lora_path / 'pytorch_lora_weights.safetensors').exists()):
            raise FileNotFoundError('Domain-style generation requires a completed Part 2 LoRA adapter.')
        if self.pipe is None:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.config['base_model'], torch_dtype=torch.float16,
                revision=self.config['base_revision'],variant=self.config['variant'])
            if self.pipe.safety_checker is None:
                raise RuntimeError('The diffusion safety checker must remain enabled.')
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_vae_tiling()
            self.pipe.set_progress_bar_config(disable=True)
            if self.lora_path and (self.lora_path / 'pytorch_lora_weights.safetensors').exists():
                self.pipe.load_lora_weights(str(self.lora_path))
                self.adapter_loaded = True
        if self.adapter_loaded:
            if state.route == 'domain_style': self.pipe.enable_lora()
            else: self.pipe.disable_lora()
        with Image.open(state.image_path) as im:
            image = im.convert('RGB').resize((512, 512))
        state.generation_metadata = {'model': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
            'adapter': str(self.lora_path) if state.route == 'domain_style' else None,
            'mode': 'img2img', 'seed': self.seed, 'steps': 20, 'strength': .6, 'prompt': state.prompt}
        return self.pipe(prompt=state.prompt, image=image, strength=.6, num_inference_steps=20,
                         generator=torch.Generator('cpu').manual_seed(self.seed))

    def review(self, state):
        from transformers import CLIPModel, CLIPProcessor
        if self.clip is None:
            self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').cpu().eval()
            self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        with Image.open(state.output_path) as im:
            image = im.convert('RGB')
        inputs = self.processor(text=[state.request], images=image, return_tensors='pt',
                                padding=True, truncation=True, max_length=77)
        with torch.inference_mode():
            result = self.clip(**inputs)
        score = (result.image_embeds * result.text_embeds).sum().item()
        if not math.isfinite(score):
            raise RuntimeError('The image critic returned a nonfinite score.')
        caption = self.captioner(image, max_new_tokens=40)[0]['generated_text']
        return {'clip_similarity': score, 'accept': score >= .22,
                'visual_judgment': 'Generated-image description: ' + caption,
                'revision_instruction': 'Make the requested change more visible while retaining the described main subject: ' + caption,
                'limitation': 'A CLIP threshold plus a caption is a weak critic, not a verified human judgment.'}
