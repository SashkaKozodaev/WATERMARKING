import torch
import numpy as np
import os
import json

class WatermarkKeys:
    
    def __init__(self, n_bits, keys_dir):
        self.n_bits = n_bits
        self.keys_dir = keys_dir
        
    def generate_keys(self, user_id, audio_length_samples=48000):
    
        public_key = torch.randint(0, 2, (self.n_bits,))
        
        secret_key = []
        used_indices = set()
        
        max_index = audio_length_samples - 1
        
        for _ in range(self.n_bits):
            while True:
                a = torch.randint(0, max_index, (1,)).item()
                b = torch.randint(0, max_index, (1,)).item()
                
                if a != b and a not in used_indices and b not in used_indices:
                    used_indices.add(a)
                    used_indices.add(b)
                    break
            
            secret_key.append((a, b))
        
        os.makedirs(self.keys_dir, exist_ok=True)
        
        #публичный ключ
        torch.save({
            'user_id': user_id,
            'public_key': public_key,
            'n_bits': self.n_bits
        }, f"{self.keys_dir}/public_key_{user_id}.pt")
        
        #секретный ключ
        with open(f"{self.keys_dir}/secret_key_{user_id}.json", 'w') as f:
            json.dump({
                'user_id': user_id,
                'pairs': secret_key,
                'n_bits': self.n_bits,
                'audio_length_samples': audio_length_samples
            }, f, indent=2)
        
        print(f"   Ключи для пользователя '{user_id}':")
        print(f"   Публичный ключ: {public_key.tolist()}")
        print(f"   Секретный ключ: {len(secret_key)} пар индексов")
        
        return public_key, secret_key
    
    def load_keys(self, user_id):
        
        public_path = f"{self.keys_dir}/public_key_{user_id}.pt"
        secret_path = f"{self.keys_dir}/secret_key_{user_id}.json"
        
        public_data = torch.load(public_path, map_location='cpu')
        public_key = public_data['public_key']
        
        with open(secret_path, 'r') as f:
            secret_data = json.load(f)
            secret_key = secret_data['pairs']
        
        return public_key, secret_key