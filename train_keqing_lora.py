# 导入 Python 内置的文件操作库，用于读取数据集路径、创建模型保存文件夹，是文件系统操作的基础
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
"""导入 PyTorch 的神经网络函数库，简写为 F，包含训练所需的损失函数、激活函数，后续计算 loss 的核心函数就来自这里"""

import torch.nn.functional as F
"""导入数值计算库 numpy，简写为 np，用于图片的数组转换和数值预处理，是图片处理的必备库"""

from accelerate import Accelerator
"""
导入训练加速工具，自动处理 GPU 设备分配、混合精度、梯度累积,
是 4G 显存能跑训练的核心，避免新手踩设备不匹配的坑
"""
import glob

from pathlib import Path

import numpy as np

from datasets import load_dataset
# 从列表创建 dataset
from datasets import Dataset, DatasetDict
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
"""
导入 Stable Diffusion 的核心组件，是扩散模型的核心：
UNet2DConditionModel：SD 的核心降噪模型，我们的 LoRA 仅微调这个模型的部分权重，是训练的核心对象
AutoencoderKL：VAE 模型，把 512x512 的图片压缩成 64x64 的隐空间特征，大幅降低显存占用和计算量
DDPMScheduler：降噪调度器，控制训练过程中的噪声添加和预测逻辑，是扩散模型的核心
StableDiffusionPipeline：SD 推理管道，后续用于验证模型效果
"""

from transformers import CLIPTextModel, CLIPTokenizer
"""导入 CLIP 文本分词器和编码器，SD 用 CLIP 把文本提示词转换成模型能理解的特征张量，是文生图的关键组件"""

from peft import LoraConfig, get_peft_model
"""
导入 LoRA 的核心实现：
LoraConfig：配置 LoRA 的超参数（秩、学习率、目标模块等）
get_peft_model：给原始 UNet 添加 LoRA 分支，仅训练 LoRA 权重，不修改原始模型，实现轻量化微调
"""

from PIL import Image

from tqdm.auto import tqdm
"""导入进度条工具，训练过程中实时显示训练进度、剩余时间、loss 数值，新手友好"""

# # import bitsandbytes as bnb
#
# sys.modules['bitsandbytes'] = None

BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DATASET_DIR = "./keqing_dataset"
OUTPUT_DIR = "./keqing_lora_model" # 训练完成后，模型的保存路径
TRIGGER_WORD = "keqing"# LoRA 的触发词，和你标注时的触发词完全一致，后续生成时输入这个词，就能触发训练好的刻晴特征
RESOLUTION = 512
TRAIN_BATCH_SIZE = 1 # 训练批次大小，也就是每次给模型喂 1 张图片
GRADIENT_ACCUMULATION_STEPS = 4 # 梯度累积步数
LEARNING_RATE = 1e-4
LR_SCHEDULER = "cosine" #余弦退火
TRAIN_EPOCHS = 20
LORA_RANK = 8 #LoRA 的秩 r，核心超参数，r 越大，LoRA 参数量越多，能学到的特征越丰富
LORA_ALPHA = 16 #LoRA 的缩放系数，通常设置为 r 的 2 倍，控制 LoRA 权重对原始模型的影响程度，alpha 越大，LoRA 的影响越强
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
MIXED_PRECISION = "fp16"
GRADIENT_CHECKPOINTING = True
##
USE_8BIT_ADAM = False

accelerator = Accelerator(
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
    mixed_precision=MIXED_PRECISION,
    log_with="tensorboard",
    project_dir=OUTPUT_DIR
)
hyperparams = {
    "learning_rate": LEARNING_RATE,
    "lr_scheduler": LR_SCHEDULER,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "train_epochs": TRAIN_EPOCHS,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
    "resolution": RESOLUTION,
    "use_8bit_adam": USE_8BIT_ADAM
}
accelerator.init_trackers("keqing_lora_training", config=hyperparams)
"""初始化训练跟踪器，把所有超参数同步到 wandb，后续可以复盘不同参数的训练效果，是算法研究的必备习惯"""

tokenizer = CLIPTokenizer.from_pretrained(BASE_MODEL_ID, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(BASE_MODEL_ID, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(BASE_MODEL_ID, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(BASE_MODEL_ID, subfolder="unet")

text_encoder.requires_grad_(False)
"""关闭文本编码器的梯度计算，固定权重，不训练，不分配梯度显存，大幅降低显存占用"""
vae.requires_grad_(False)
unet.enable_gradient_checkpointing()


#-------lora核心配置---------#
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,#LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
    lora_dropout=0.05,#设置 5% 的 dropout 率，随机丢弃部分神经元，防止模型过拟合，提升泛化能力
    bias="none",
    # task_type="UNET"
)
"""
PEFT 内部会针对 UNet 结构，自动将 LoRA 层添加到合适的子模块上。即便你手动指定了 target_modules，
设置 task_type 也能确保适配器保存的元数据正确标记模型类型，便于后续加载
"""
unet = get_peft_model(unet, lora_config)

unet.print_trainable_parameters()
"""
：打印可训练的参数量，你会看到可训练参数仅占 0.1% 左右，仅训练十万级别的参数，
而原始 UNet 有十几亿参数，这就是 LoRA 轻量化的核心魅力
"""

def tokenize_captions(examples):
    """ 这个函数的作用是将数据集中的文本描述（提示词）转换为 CLIP 模型可以处理的 token ID 序列。"""
    captions = [caption for caption in examples["text"]]
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    """
    分词参数设置：
max_length：最大长度 77，SD1.5 的 CLIP 最多支持 77 个 token 的提示词
padding="max_length"：不足 77 个 token 的，用空白 token 补齐
truncation=True：超过 77 个 token 的，自动截断
return_tensors="pt"：返回 PyTorch 张量格式，模型可直接使用
    """
    return inputs.input_ids

def preprocess_train(examples):
    images = []
    for img in examples["image"]:
        if isinstance(img, str):
            # 如果是字符串路径，手动打开
            img = Image.open(img).convert("RGB")
        else:
            # 如果已经是 Image 对象，直接转换
            img = img.convert("RGB")
        images.append(img)
    examples["pixel_values"] = [torch.tensor(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0 for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples


data_files = []
image_paths = glob.glob(os.path.join(DATASET_DIR, "*.jpg")) + glob.glob(os.path.join(DATASET_DIR, "*.png"))

for img_path in image_paths:
    txt_path = Path(img_path).with_suffix('.txt')
    caption = ""
    if txt_path.exists():
        with open(txt_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()

    data_files  .append({"image": img_path, "text": caption})


dataset = Dataset.from_list(data_files)
dataset = dataset.with_transform(preprocess_train)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

# if USE_8BIT_ADAM:
#     optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=LEARNING_RATE)
# else:
#     optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
# 修改后（直接使用普通 AdamW，不依赖 bitsandbytes）
optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader) * TRAIN_EPOCHS)



unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler
)

text_encoder = text_encoder.to(accelerator.device)

vae = vae.to(accelerator.device)

noise_scheduler = DDPMScheduler.from_pretrained(BASE_MODEL_ID, subfolder="scheduler")



global_step = 0

for epoch in range(TRAIN_EPOCHS):
    unet.train()
    progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch+1}/{TRAIN_EPOCHS}")

    for batch in train_dataloader:
        with accelerator.accumulate(unet):
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)


    progress_bar.close()
    if accelerator.is_main_process:
        unet.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch_{epoch + 1}"))
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    unet.save_pretrained(os.path.join(OUTPUT_DIR, "final_lora_model"))
accelerator.end_training()
