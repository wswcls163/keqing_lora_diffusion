import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import os
os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_MODEL_PATH = r"D:\PythonStudy\pytorch_study\formal_project\keqing_lora_diffusion\keqing_lora_model\final_lora_model"
pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None,  # 新增：禁用安全检查器
    requires_safety_checker=False  # 新增：确认不需要安全检查
)
pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_MODEL_PATH)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
pipe.enable_vae_slicing()
prompt = "1girl, keqing (genshin impact), solo, purple eyes, hair bun, purple hair, twintails, hair ornament, looking at viewer, braid, long hair, cone hair bun, bangs, detached sleeves, bare shoulders, upper body, double bun, closed mouth, choker, dress, blush, white background, hair between eyes, simple background"
negative_prompt = "ugly, blurry, deformed, bad anatomy, worst quality, low resolution, watermark, text"
image = pipe(prompt, negative_prompt=negative_prompt, width=512, height=512, num_inference_steps=25, guidance_scale=7.5).images[0]
image.save("keqing_generated.png")