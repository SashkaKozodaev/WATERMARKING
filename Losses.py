import torch
import torch.nn.functional as F


def wm_loss(audio, public_key, secret_key, epsilon=0.2):

    audio = audio.unsqueeze(0)
    audio_flat = audio.flatten()

    loss = torch.tensor(0.0, device=audio.device)
    
    n_bits = len(public_key)
    
    for j in range(n_bits):
        a, b = secret_key[j]
        
        diff = audio_flat[a] - audio_flat[b]
        
        if public_key[j] == 0:
            violation = epsilon - diff
        else:
            violation = epsilon + diff
        
        loss = loss + torch.clamp(violation, min=0.0)
    
    return loss / n_bits


def quality_loss(audio_current, audio_original):

    audio_current = audio_current.unsqueeze(0)
    audio_original = audio_original.unsqueeze(0)
    
    min_len = min(audio_current.shape[1], audio_original.shape[1])
    audio_current = audio_current[:, :min_len]
    audio_original = audio_original[:, :min_len]
    
    return F.mse_loss(audio_current, audio_original)