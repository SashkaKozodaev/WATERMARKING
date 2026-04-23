#CUDA_VISIBLE_DEVICES=0 uv run python /home/jovyan/kozodaev/WATERMARKING/main.py

import numpy as np
import torch
import scipy.io.wavfile as wav

from keys import WatermarkKeys
from watermark import generate_watermarked_audio
from watermark_extractor import compare_with_original
from model import load_model
import yaml


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

config = load_config("/home/jovyan/kozodaev/WATERMARKING/config.yaml")

repo_id = config["repo_id"]
device = config["device"]
cache_dir = config["cache_dir"]

prompt = config["prompt"]
audio_length_in_s = config["audio_length_in_s"]
num_inference_steps = config["num_inference_steps"]

n_bits = config["n_bits"]
epsilon = config["epsilon"]
lambda_wm = config["lambda_wm"]
lambda_qual = config["lambda_qual"]

optimization_steps = config["optimization_steps"]
lr = config["lr"]
target_length = config["target_length"]

output_dir = config["output_dir"]
keys_dir = config["keys_dir"]




print("\n   Загрузка модели")
pipe = load_model(repo_id, cache_dir, torch.float32, device)

print("\n   Генерация ключей")
wk = WatermarkKeys(n_bits=n_bits, keys_dir = keys_dir)
public_key, secret_key = wk.generate_keys("test_user", target_length)

print("\n   Внедрение watermark")
audio_watermarked, latent_optimized = generate_watermarked_audio(
    pipe=pipe,
    prompt=prompt,
    public_key=public_key,
    secret_key=secret_key,
    audio_length_in_s=audio_length_in_s,
    num_inference_steps=num_inference_steps,
    target_length=target_length,
    epsilon=epsilon,
    lambda_wm=lambda_wm,
    lambda_qual=lambda_qual,
    optimization_steps=optimization_steps,
    lr=lr,
    output_dir=output_dir,
    device=device
)

print("ПРОВЕРКА WATERMARK")

audio_path = f"{output_dir}/watermarked_audio.wav"
sample_rate, audio = wav.read(audio_path)

if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32767.0


original_public, original_secret = wk.load_keys("test_user")


comparison = compare_with_original(
    audio_path, 
    "test_user",
    original_public_key=original_public,
    original_secret_key=original_secret
)