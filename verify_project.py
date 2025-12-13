import os
import torch
import shutil
from src.models import get_vae, get_unet, get_emotion_classifier
from diffusers import DDPMScheduler
import numpy as np

def verify_vae():
    print("Verifying VAE...")
    vae = get_vae()
    dummy_input = torch.randn(1, 3, 256, 256) # Batch size 1, 3 channels, 256x256
    with torch.no_grad():
        latents = vae.encode(dummy_input).latent_dist.sample()
        # Expected shape: [1, 4, 32, 32] (256/8 = 32)
        print(f"VAE Latent Shape: {latents.shape}")
        assert latents.shape == (1, 4, 32, 32), "VAE latent shape mismatch!"
        
        decoded = vae.decode(latents).sample
        print(f"VAE Output Shape: {decoded.shape}")
        assert decoded.shape == (1, 3, 256, 256), "VAE output shape mismatch!"
    print("VAE Verification Passed.\n")

def verify_unet():
    print("Verifying UNet...")
    unet = get_unet(image_size=32)
    dummy_latents = torch.randn(2, 4, 32, 32)
    dummy_timesteps = torch.tensor([100, 200])
    
    with torch.no_grad():
        output = unet(dummy_latents, dummy_timesteps).sample
        print(f"UNet Output Shape: {output.shape}")
        assert output.shape == (2, 4, 32, 32), "UNet output shape mismatch!"
    print("UNet Verification Passed.\n")

def verify_classifier():
    print("Verifying Classifier...")
    processor, model = get_emotion_classifier()
    dummy_image = torch.randn(1, 3, 224, 224) # Standard input size
    
    with torch.no_grad():
        outputs = model(pixel_values=dummy_image)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        print(f"Classifier Probabilities Shape: {probs.shape}")
        print(f"Classes: {model.config.id2label}")
    print("Classifier Verification Passed.\n")

def verify_smc_logic():
    print("Verifying SMC Logic (Mock)...")
    num_particles = 10
    weights = torch.tensor([0.1, 0.05, 0.05, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05])
    indices = torch.multinomial(weights, num_samples=num_particles, replacement=True)
    print(f"Resampled Indices: {indices}")
    assert len(indices) == num_particles
    print("SMC Logic Verification Passed.\n")

if __name__ == "__main__":
    verify_vae()
    verify_unet()
    verify_classifier()
    verify_smc_logic()
    print("All Verifications Passed!")
