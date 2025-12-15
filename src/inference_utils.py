import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, UNet2DModel

from src.config import DEVICE, TRAIN_CONFIG
from src.models import get_vae, get_emotion_classifier
from src.dataset import get_transforms

def smc_inference_core(image_path, unet, vae, classifier, processor, target_emotion, num_particles, steps, guidance_scale, noise_strength, leash, resample_interval):
    device = unet.device
    
    # Check emotion
    id2label = classifier.config.id2label
    label2id = {v.lower(): k for k, v in id2label.items()}
    if target_emotion.lower() not in label2id:
        print(f"Emotion '{target_emotion}' not found. Available: {list(label2id.keys())}")
        return None, 0.0
    target_class_id = label2id[target_emotion.lower()]

    # Load & Preprocess
    transform = get_transforms(256)
    try:
        original_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, 0.0
        
    image_tensor = transform(original_pil).unsqueeze(0).to(device)
    
    # Encode
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
    latents = latents.repeat(num_particles, 1, 1, 1)
    
    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=TRAIN_CONFIG["num_train_timesteps"],
        beta_schedule=TRAIN_CONFIG["beta_schedule"],
        prediction_type=TRAIN_CONFIG["prediction_type"],
        clip_sample=False
    )
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps
    
    # Add Noise
    noise = torch.randn_like(latents)
    start_idx = int((1.0 - noise_strength) * (len(timesteps) - 1))
    t_start = timesteps[start_idx]
    curr_latents = scheduler.add_noise(latents, noise, t_start)
    
    # Normalization constants
    image_mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    image_std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    norm_mean = torch.tensor(image_mean, device=device).view(1, 3, 1, 1)
    norm_std = torch.tensor(image_std, device=device).view(1, 3, 1, 1)

    # SMC Loop
    for i, t in enumerate(timesteps[start_idx:]):
        # Denoise
        with torch.no_grad():
            noise_pred = unet(curr_latents, t).sample
            curr_latents = scheduler.step(noise_pred, t, curr_latents).prev_sample
            
            leash_t = leash * (1-i)/t_start
            curr_latents = (1 - leash_t) * curr_latents + leash_t * latents
            
        # Resample
        if (i + 1) % resample_interval == 0 and (i + 1) < len(timesteps[start_idx:]):
            with torch.no_grad():
                temp_latents = curr_latents / vae.config.scaling_factor
                decoded = vae.decode(temp_latents).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                images_resized = F.interpolate(decoded, size=(224, 224), mode='bilinear', align_corners=False)
                classifier_inputs = (images_resized - norm_mean) / norm_std
                outputs = classifier(pixel_values=classifier_inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                target_probs = probs[:, target_class_id]
                
                weights = target_probs ** guidance_scale
                if weights.sum() == 0:
                    weights = torch.ones_like(weights) / len(weights)
                else:
                    weights = weights / weights.sum()
                    
                indices = torch.multinomial(weights, num_samples=num_particles, replacement=True)
                curr_latents = curr_latents[indices]
                
    # Final Decode
    with torch.no_grad():
        curr_latents = curr_latents / vae.config.scaling_factor
        result_images_tensor = vae.decode(curr_latents).sample
        result_images_tensor = (result_images_tensor / 2 + 0.5).clamp(0, 1)
        
        images_resized = F.interpolate(result_images_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        classifier_inputs = (images_resized - norm_mean) / norm_std
        outputs = classifier(pixel_values=classifier_inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        final_scores = probs[:, target_class_id].cpu().numpy()
        
        best_idx = np.argmax(final_scores)
        best_score = final_scores[best_idx]
        best_image_tensor = result_images_tensor[best_idx] # [3, H, W]
        
    return best_image_tensor, best_score

def smc_image_inference(image_path, unet_path, target_emotion, num_particles=10, steps=50, guidance_scale=2.0, noise_strength=0.3, resample_interval=1, leash=0.0, output_path=None):
    """
    Applies SMC inference to modify the emotion of an existing image.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    device = DEVICE
    
    # Load Models
    print(f"Loading UNet from {unet_path}...")
    unet = UNet2DModel.from_pretrained(unet_path)
    unet.to(device)
    unet.eval()
    
    vae = get_vae().to(device)
    processor, classifier = get_emotion_classifier()
    classifier.to(device)
    
    best_image_tensor, best_score = smc_inference_core(
        image_path, unet, vae, classifier, processor, target_emotion, 
        num_particles, steps, guidance_scale, noise_strength, leash, resample_interval
    )
    
    if best_image_tensor is None:
        return

    # Save/Show
    img_np = best_image_tensor.cpu().permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(Image.open(image_path))
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title(f"Result ({target_emotion})\nScore: {best_score:.4f}")
    plt.imshow(img_np)
    plt.axis("off")
    
    if output_path:
        plt.savefig(output_path)
        print(f"Result saved to {output_path}")
    else:
        plt.show()

def smc_image_inference_dual(image_path, unet_path, target_emotion1, target_emotion2, num_particles=10, steps=50, guidance_scale=2.0, noise_strength=0.3, resample_interval=1, leash=0.0, output_path=None):
    """
    Applies SMC inference for two target emotions.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    device = DEVICE
    
    # Load Models
    print(f"Loading UNet from {unet_path}...")
    unet = UNet2DModel.from_pretrained(unet_path)
    unet.to(device)
    unet.eval()
    
    vae = get_vae().to(device)
    processor, classifier = get_emotion_classifier()
    classifier.to(device)
    
    results = []
    for emo in [target_emotion1, target_emotion2]:
        print(f"Running SMC for {emo}...")
        best_image_tensor, best_score = smc_inference_core(
            image_path, unet, vae, classifier, processor, emo, 
            num_particles, steps, guidance_scale, noise_strength, leash, resample_interval
        )
        if best_image_tensor is not None:
            img_np = best_image_tensor.cpu().permute(1, 2, 0).numpy()
            results.append((emo, img_np, best_score))
            
    if len(results) < 2:
        print("Failed to generate both images.")
        return

    # Display
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(Image.open(image_path))
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title(f"{results[0][0]}\nScore: {results[0][2]:.4f}")
    plt.imshow(results[0][1])
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title(f"{results[1][0]}\nScore: {results[1][2]:.4f}")
    plt.imshow(results[1][1])
    plt.axis("off")
    
    if output_path:
        plt.savefig(output_path)
        print(f"Result saved to {output_path}")
    else:
        plt.show()
