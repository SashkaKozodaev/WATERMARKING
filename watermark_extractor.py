import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path


def extract_watermark(audio, secret_key):
  
    extracted_bits = []
    
    for a, b in secret_key:
        if audio[a] >= audio[b]:    extracted_bits.append(0)
        else:                       extracted_bits.append(1)
    
    return np.array(extracted_bits)


def calculate_hamming_distance(bits1, bits2):

    if len(bits1) != len(bits2):
        min_len = min(len(bits1), len(bits2))
        bits1 = bits1[:min_len]
        bits2 = bits2[:min_len]
    
    return np.sum(bits1 != bits2)


def compare_with_original(audio_path, user_id, original_public_key, original_secret_key):

    sample_rate, audio = wav.read(audio_path)
    audio = audio.astype(np.float32) / 32767.0
      
    
    public_key = original_public_key
    secret_key = original_secret_key
    public_key = public_key.cpu().numpy()
    
    extracted = extract_watermark(audio, secret_key)
    
    distance = calculate_hamming_distance(extracted, public_key)
    n_bits = len(public_key)
    
    differences = np.where(extracted != public_key)[0]
    
    result = {
        'user_id': user_id,
        'distance': distance,
        'n_bits': n_bits,
        'extracted_bits': extracted.tolist(),
        'expected_bits': public_key.tolist(),
        'differences': differences.tolist(),
        'num_differences': len(differences)
    }
    
    print(f"User: {user_id}")
    print(f"watermark length: {n_bits}")
    print(f"Hamming distance: {distance}/{n_bits}")
    
    print(f"\nDifferent indexes: {differences}")
    
    print(f"\n  Keys comparison:")
    print(f"Original  : {public_key}")
    print(f"Extracted : {extracted}")
    
    return result