import os
from diffusers import AudioLDMPipeline
import torch


def load_model(repo_id, cache_dir="model_cache", torch_dtype=torch.float32, device="cuda"):

    model_path = os.path.join(cache_dir, repo_id.replace("/", "_"))
    
    if os.path.exists(model_path):
        print(f"Загрузка модели из: {model_path}")
        pipe = AudioLDMPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    else:
        print(f"Скачивание модели: {repo_id}")
        pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype, use_safetensors=True)
        print(f"Сохранение модели в: {model_path}")
        os.makedirs(cache_dir, exist_ok=True)
        pipe.save_pretrained(model_path)
    
    pipe = pipe.to(device)
    return pipe