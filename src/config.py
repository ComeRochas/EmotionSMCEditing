import torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

TRAIN_CONFIG = {
    "data_dir": "/kaggle/input/celebahq-resized-256x256",
    "output_dir": "checkpoints",
    "batch_size": 32,
    "epochs": 100, 
    "lr": 1e-4,
    "mixed_precision": "fp16",
    "gradient_accumulation_steps": 1,
    "num_train_timesteps": 1000,
    "beta_schedule": "scaled_linear",
    "prediction_type": "epsilon",
    "checkpoint_path": None,  # For continuing training from a checkpoint
    "start_epoch": 0  # You may specify the starting epoch if resuming
}

TEST_CONFIG = {
    "data_dir": "/kaggle/input/celebahq-resized-256x256",
    "trained_unet_path": "checkpoint-epoch-89",
    "test_image_path": "testimage.jpg",
    "target_emotion": "happy",
    "target_emotion2": None,
    "num_particles": 10,
    "steps": 50,
    "guidance_scale": 2.0,
    "noise_strength": 0.3,
    "resample_interval": 1,
    "leash": 0.0,
    "output_path": "inference_result.png"
}

BENCHMARK_CONFIG = {
    "data_dir": "/kaggle/input/celebahq-resized-256x256",
    "trained_unet_path": "checkpoint-epoch-89",
    "tests": [
        {"var": "particles", "values": [5, 10, 20]},
        {"var": "steps", "values": [25, 50, 100]},
        {"var": "guidance_scale", "values": [1.0, 2.0, 4.0]},
        {"var": "noise_strength", "values": [0.1, 0.3, 0.5]}
    ],
    "fixed": {
        "particles": 10,
        "steps": 50,
        "guidance_scale": 2.0,
        "noise_strength": 0.3,
        "resample_interval": 1,
        "leash": 0.0
    }
}