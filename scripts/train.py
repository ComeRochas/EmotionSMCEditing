import argparse
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
from src.dataset import CelebAHQDataset, get_transforms, download_dataset
from src.models import get_vae, get_unet

def main():
    parser = argparse.ArgumentParser(description="Train Latent Diffusion Model")
    parser.add_argument("--data_dir", type=str, default="./data/celebahq-resized-256x256", help="Path to dataset")
    parser.add_argument("--download_data", action="store_true", help="Download dataset from Kaggle")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if args.download_data and accelerator.is_main_process:
        download_dataset(os.path.dirname(args.data_dir))
    accelerator.wait_for_everyone()

    # Create output dir
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load Models
    vae = get_vae()
    unet = get_unet(image_size=32) # 256 / 8 = 32
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        clip_sample=False
    )

    # Freeze VAE and move to device
    vae.to(accelerator.device)
    vae.eval()

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # Dataset & Dataloader
    dataset = CelebAHQDataset(root_dir=args.data_dir, transform=get_transforms(256))
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # Prepare with Accelerator
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Training Loop
    global_step = 0
    for epoch in range(args.epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            
            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(clean_images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            bs = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device).long()

            # Add noise
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            with accelerator.accumulate(unet):
                noise_pred = unet(noisy_latents, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1

        # Save Checkpoint
        if accelerator.is_main_process:
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.save_pretrained(os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}"))
                print(f"Saved model to {os.path.join(args.output_dir, f'checkpoint-epoch-{epoch}')}")

    print("Training finished.")

if __name__ == "__main__":
    main()
