import torch
import torch.nn.functional as F
import scipy
import numpy as np
import os
from tqdm import tqdm

from Losses import wm_loss, quality_loss


def generate_initial_audio(pipe, prompt, num_inference_steps, audio_length_in_s, device, seed=42):

    original_vae_decode = pipe.vae.decode
    captured_latents = []
    
    def capture_and_decode(latents, *args, **kwargs):
        captured_latents.append(latents.clone().detach())
        return original_vae_decode(latents, *args, **kwargs)
    
    pipe.vae.decode = capture_and_decode
    
    with torch.no_grad():
        result = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            audio_length_in_s=audio_length_in_s,
            generator=torch.Generator(device=device).manual_seed(seed)
        )
        audio_original = result.audios[0]
    
    pipe.vae.decode = original_vae_decode
    
    latent_init = captured_latents[-1].to(device)
    
    return audio_original, latent_init


def audio_from_latent(pipe, latent, target_length=48000):
    mel = pipe.vae.decode(latent).sample
    mel = mel.squeeze(1)
    audio = pipe.vocoder(mel)
    audio = audio[0]
    audio = audio[:target_length]
    
    return audio


def optimize_latent(pipe, latent_init, audio_original, public_key, secret_key,
                    target_length=48000, epsilon=0.2, lambda_wm=0.9, lambda_qual=150.0,
                    optimization_steps=500, lr=0.01, device="cuda"):


    pipe.vae.train()
    pipe.vocoder.train()

    latent = latent_init.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([latent], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    audio_original_tensor = torch.from_numpy(audio_original).float().to(device)
        
    for _ in tqdm(range(optimization_steps)):
        audio_current = audio_from_latent(pipe, latent, target_length)
        
        loss_wm = wm_loss(audio_current, public_key, secret_key, epsilon)
        loss_qual = quality_loss(audio_current, audio_original_tensor)
        total_loss = lambda_wm * loss_wm + lambda_qual * loss_qual
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        scheduler.step()
    
    
    pipe.vae.eval()
    pipe.vocoder.eval()

    return latent


def save_audio(audio, path, sample_rate=16000):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    max_val = np.max(np.abs(audio))
    audio = audio / max_val
    audio_int16 = (audio * 32767).astype(np.int16)
    scipy.io.wavfile.write(path, rate=sample_rate, data=audio_int16)


def generate_watermarked_audio(pipe, prompt, public_key, secret_key, audio_length_in_s, num_inference_steps,
                               target_length, epsilon, lambda_wm, lambda_qual, optimization_steps, lr,
                               output_dir="output_watermarked", device="cuda"):

    os.makedirs(output_dir, exist_ok=True)
    
    print("\n   Генерация Оригинального аудио")
    audio_original, latent_init = generate_initial_audio(pipe, prompt, num_inference_steps, audio_length_in_s, device)
    save_audio(audio_original, f"{output_dir}/original_audio.wav")
    print(f"Форма латента: {latent_init.shape}")
    
    print("\n Внедрение watermark в латент")
    latent_optimized = optimize_latent(
        pipe, latent_init, audio_original, public_key, secret_key,
        target_length, epsilon, lambda_wm, lambda_qual,
        optimization_steps, lr, device
    )
    with torch.no_grad():
        audio_final = audio_from_latent(pipe, latent_optimized, target_length).cpu().numpy()
    
    save_audio(audio_final, f"{output_dir}/watermarked_audio.wav")
    
    torch.save({
        'latent': latent_optimized.detach().cpu(),
        'prompt': prompt
    }, f"{output_dir}/watermarked_latent.pt")
    
    return audio_final, latent_optimized