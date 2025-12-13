import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch.nn.functional as F

import sys
sys.path.append(".")
from src.models import get_vae, get_emotion_classifier

def main():
    parser = argparse.ArgumentParser(description="SMC Inference for Emotion Editing")
    parser.add_argument("--unet_path", type=str, required=True, help="Path to trained UNet checkpoint")
    parser.add_argument("--target_emotion", type=str, required=True, help="Target emotion (e.g., 'happy', 'sad')")
    parser.add_argument("--num_particles", type=int, default=30, help="Number of SMC particles")
    parser.add_argument("--resample_interval", type=int, default=5, help="Resample every N steps")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Temperature for weighting (higher = stronger guidance)")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Models
    print("Loading models...")
    vae = get_vae().to(device)
    
    # Load UNet
    try:
        unet = UNet2DModel.from_pretrained(args.unet_path).to(device)
    except:
        print(f"Could not load from pretrained path {args.unet_path}, trying to load as state dict or assuming it is a local diffusers save.")
        unet = UNet2DModel.from_pretrained(args.unet_path).to(device)
    unet.eval()

    # Load Classifier
    processor, classifier = get_emotion_classifier()
    classifier.to(device)
    
    # Check emotion labels
    id2label = classifier.config.id2label
    label2id = {v.lower(): k for k, v in id2label.items()}
    print(f"Available emotions: {list(label2id.keys())}")
    
    if args.target_emotion.lower() not in label2id:
        raise ValueError(f"Target emotion '{args.target_emotion}' not found. Available: {list(label2id.keys())}")
    
    target_class_id = label2id[args.target_emotion.lower()]
    print(f"Target emotion ID: {target_class_id} ({args.target_emotion})")

    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        clip_sample=False
    )
    scheduler.set_timesteps(1000)

    # Initialize Particles (Latents)
    latents = torch.randn((args.num_particles, 4, 32, 32), device=device)

    # SMC Loop
    timesteps = scheduler.timesteps
    
    print("Starting SMC Inference...")
    for i, t in enumerate(tqdm(timesteps)):
        # 1. Denoise Step (Prediction)
        with torch.no_grad():
            noise_pred = unet(latents, t).sample
            output = scheduler.step(noise_pred, t, latents)
            latents = output.prev_sample

        # 2. Resampling Step
        if (i + 1) % args.resample_interval == 0 and (i + 1) < len(timesteps):
            scaled_latents = latents / vae.config.scaling_factor
            
            with torch.no_grad():
                # Decode
                images_tensor = vae.decode(scaled_latents).sample
                images_tensor = (images_tensor / 2 + 0.5).clamp(0, 1)
                
                # Resize/Preprocess for classifier
                target_size = (224, 224) 
                images_resized = F.interpolate(images_tensor, size=target_size, mode='bilinear', align_corners=False)
                
                mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                
                classifier_inputs = (images_resized - mean) / std
                
                # Forward pass
                outputs = classifier(pixel_values=classifier_inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                
                # Get weight for target emotion
                target_probs = probs[:, target_class_id]
                weights = target_probs ** args.guidance_scale
                
                # Normalize weights
                if weights.sum() == 0:
                    weights = torch.ones_like(weights) / len(weights)
                else:
                    weights = weights / weights.sum()
                
                # Resample
                indices = torch.multinomial(weights, num_samples=args.num_particles, replacement=True)
                latents = latents[indices]

    # Final Decode
    scaled_latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        images = vae.decode(scaled_latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).astype(np.uint8)

    # Save images
    for i in range(args.num_particles):
        Image.fromarray(images[i]).save(os.path.join(args.output_dir, f"sample_{i}_{args.target_emotion}.png"))

    print(f"Saved {args.num_particles} images to {args.output_dir}")

if __name__ == "__main__":
    main()
