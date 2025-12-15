import argparse
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
import wandb

import sys
sys.path.append(".")
from src.dataset import CelebAHQDataset, get_transforms, download_dataset
from src.models import get_vae, get_unet
from src.config import TRAIN_CONFIG

def main():
    parser = argparse.ArgumentParser(description="Train Latent Diffusion Model")
    parser.add_argument("--data_dir", type=str, default=TRAIN_CONFIG["data_dir"], help="Path to dataset")
    parser.add_argument("--download_data", action="store_true", help="Download dataset from Kaggle")
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"], help="Number of epochs")
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["lr"], help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=TRAIN_CONFIG["output_dir"], help="Directory to save models")
    parser.add_argument("--mixed_precision", type=str, default=TRAIN_CONFIG["mixed_precision"], choices=["no", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TRAIN_CONFIG["gradient_accumulation_steps"], help="Gradient accumulation steps")
    parser.add_argument("--checkpoint_path", type=str, default=TRAIN_CONFIG["checkpoint_path"], help="Path to checkpoint to resume from")
    parser.add_argument("--start_epoch", type=int, default=TRAIN_CONFIG["start_epoch"], help="Starting epoch number")
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb"
    )

    if args.download_data and accelerator.is_main_process:
        download_dataset(os.path.dirname(args.data_dir))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.init_trackers("celebahq-latent-diffusion", config=vars(args))

    # Load Models
    vae = get_vae()
    
    start_epoch = 0
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Resuming training from: {args.checkpoint_path}")
        from diffusers import UNet2DModel
        unet = UNet2DModel.from_pretrained(args.checkpoint_path)
        start_epoch = args.start_epoch
    else:
        unet = get_unet(image_size=32)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=TRAIN_CONFIG["num_train_timesteps"],
        beta_schedule=TRAIN_CONFIG["beta_schedule"],
        prediction_type=TRAIN_CONFIG["prediction_type"],
        clip_sample=False
    )

    # Freeze VAE and move to device
    vae.to(accelerator.device)
    vae.eval()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    dataset = CelebAHQDataset(root_dir=args.data_dir, transform=get_transforms(256))
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    global_step = 0
    print(f"Starting training from epoch {start_epoch} to {start_epoch + args.epochs - 1}")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        unet.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            
            with torch.no_grad():
                latents = vae.encode(clean_images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bs = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.accumulate(unet):
                noise_pred = unet(noisy_latents, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            global_step += 1
            accelerator.log({"train_loss": loss.item()}, step=global_step)

        if accelerator.is_main_process:
            if (epoch + 1) % 5 == 0 or epoch == start_epoch + args.epochs - 1:
                unwrapped_unet = accelerator.unwrap_model(unet)
                save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
                unwrapped_unet.save_pretrained(save_path)
                print(f"Saved model to {save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    main()
