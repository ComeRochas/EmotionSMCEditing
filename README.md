# EmotionSMCEditing

## Description

This project implements emotion editing on facial images using Latent Diffusion Models (LDM) and Sequential Monte Carlo (SMC) sampling. It allows for controlled modification of facial expressions (e.g., making a face look happy, sad, angry) while preserving the identity and other facial features.

## Structure

- `src/`: Contains the core source code (models, dataset, inference logic, configuration).
- `scripts/`: Contains executable scripts for training, inference, and benchmarking.
- `checkpoints/`: Directory for saving trained models.
- `results/`: Directory for inference outputs and for the FinalReport.pdf descriptive of the experiment and its results

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Configuration is centralized in `src/config.py`. You can modify hyperparameters for training, inference, and benchmarking there.

- `TRAIN_CONFIG`: Parameters for training the LDM.
- `TEST_CONFIG`: Default parameters for inference.
- `BENCHMARK_CONFIG`: Parameters for running benchmarks.

## Usage

You can follow the steps through the notebook to see the training process and use the different inference functions, or run benchmark tests. All hyperparameters are set in Cell 5. Once set, you can simply run all cells.

### Training

To train the Latent Diffusion Model. You must set the path to the dataset in the config.py file. It is also possible to resume training from a checkpoint model.

Example:
```bash
python scripts/train.py --data_dir /path/to/dataset --output_dir checkpoints --batch_size 32 --epochs 100 --lr 1e-4 --mixed_precision fp16
```


#### Optional arguments

- `--data_dir` <path> : Path to the dataset directory
- `--output_dir` <path> : Directory to save checkpoints and logs (default: `checkpoints`)
- `--batch_size` <int> : Training batch size (default: `32`)
- `--epochs` <int> : Number of training epochs (default: `100`)
- `--lr` : Learning rate (default: `1e-4`)
- `--mixed_precision` <fp16|fp32> : Mixed precision mode (default: `fp16`)
- `--gradient_accumulation_steps` <int>  : Gradient accumulation steps (default: `1`)
- `--num_train_timesteps` <int> : Number of diffusion timesteps (default: `1000`)
- `--checkpoint_path` <path> : Path to a checkpoint to resume training (default: `None`)
- `--start_epoch` <int> : Starting epoch index when resuming training (default: `0`)



### Inference

To run emotion editing on a specific image. If you set `--target_emotion2`, two emotion inference will be generated.

Example:
```bash
python scripts/inference.py --image_path path/to/image.jpg --target_emotion happy --output_path result.png
```

#### Optional arguments

- `--image_path` <path> : Path to input image (default: from config)
- `--unet_path` <path> : Path to trained UNet checkpoint (default: from config)
- `--target_emotion` <str> : Target emotion (e.g., 'happy') (default: from config)
- `--target_emotion2` <str> : Second target emotion for dual inference (default: `None`)
- `--num_particles` <int> : Number of SMC particles (default: `10`)
- `--steps` <int> : Number of diffusion steps (default: `50`)
- `--guidance_scale` <float> : Guidance scale (default: `2.0`)
- `--noise_strength` <float> : Noise strength (default: `0.3`)
- `--resample_interval` <int> : Resample interval (default: `1`)
- `--leash` <float> : Leash factor (default: `0.0`)
- `--output_path` <path> : Path to save result (default: `result.png`)

### Benchmarking

To run a benchmark to evaluate the performance across different parameters. This will execute tests defined in `BENCHMARK_CONFIG` and save results to `benchmark_results.csv`. You have to manually define your tests in the config.py

Example:
```bash
python scripts/benchmark.py --images_per_test 20
```

#### Optional arguments

- `--data_dir` <path> : Path to dataset (default: from config)
- `--unet_path` <path> : Path to trained UNet (default: from config)
- `--images_per_test` <int> : Number of images per test (default: `10`)

## Credits

Developed for the Image Analysis and Computer Vision course at École polytechnique.
Based on Latent Diffusion Models and SMC sampling techniques (Feynamn-Kac Steering). See results/FinalReport.pdf for more detailed information.