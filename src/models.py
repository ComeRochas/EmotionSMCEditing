import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DModel
from transformers import AutoImageProcessor, AutoModelForImageClassification

def get_vae(model_id="runwayml/stable-diffusion-v1-5"):
    """
    Loads the VAE from Stable Diffusion 1.5.
    Returns the VAE model on CPU.
    """
    print(f"Loading VAE from {model_id}...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    vae.eval()
    vae.requires_grad_(False) # Freeze VAE
    return vae

def get_unet(image_size=32, in_channels=4, out_channels=4):
    """
    Creates an unconditional UNet for latent diffusion.
    Note: 'image_size' here refers to the spatial dimension of the LATENTS.
    For 256x256 images, VAE with f=8 produces 32x32 latents.
    """
    print(f"Creating UNet2DModel with input/output channels={in_channels}, latent_size={image_size}...")
    unet = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return unet

def get_emotion_classifier(model_id="dima806/facial_emotions_image_detection"):
    """
    Loads the emotion classifier.
    """
    print(f"Loading Emotion Classifier from {model_id}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    model.requires_grad_(False)
    return processor, model
