import argparse
import sys
sys.path.append(".")
from src.inference_utils import smc_image_inference, smc_image_inference_dual
from src.config import TEST_CONFIG

def main():
    parser = argparse.ArgumentParser(description="SMC Inference for Emotion Editing")
    parser.add_argument("--image_path", type=str, default=TEST_CONFIG["test_image_path"], help="Path to input image")
    parser.add_argument("--unet_path", type=str, default=TEST_CONFIG["trained_unet_path"], help="Path to trained UNet checkpoint")
    parser.add_argument("--target_emotion", type=str, default=TEST_CONFIG["target_emotion"], help="Target emotion (e.g., 'happy')")
    parser.add_argument("--target_emotion2", type=str, default=TEST_CONFIG["target_emotion2"], help="Second target emotion for dual inference")
    parser.add_argument("--num_particles", type=int, default=TEST_CONFIG.get("num_particles", 10), help="Number of SMC particles")
    parser.add_argument("--steps", type=int, default=TEST_CONFIG.get("steps", 50), help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=TEST_CONFIG.get("guidance_scale", 2.0), help="Guidance scale")
    parser.add_argument("--noise_strength", type=float, default=TEST_CONFIG.get("noise_strength", 0.3), help="Noise strength")
    parser.add_argument("--resample_interval", type=int, default=TEST_CONFIG.get("resample_interval", 1), help="Resample interval")
    parser.add_argument("--leash", type=float, default=TEST_CONFIG.get("leash", 0.0), help="Leash factor")
    parser.add_argument("--output_path", type=str, default=TEST_CONFIG.get("output_path", "result.png"), help="Path to save result")
    
    args = parser.parse_args()

    if args.target_emotion2:
        print(f"Running Dual Inference: {args.target_emotion} & {args.target_emotion2}")
        smc_image_inference_dual(
            args.image_path, args.unet_path, args.target_emotion, args.target_emotion2,
            num_particles=args.num_particles, steps=args.steps,
            guidance_scale=args.guidance_scale, noise_strength=args.noise_strength,
            resample_interval=args.resample_interval, leash=args.leash,
            output_path=args.output_path
        )
    else:
        print(f"Running Single Inference: {args.target_emotion}")
        smc_image_inference(
            args.image_path, args.unet_path, args.target_emotion,
            num_particles=args.num_particles, steps=args.steps,
            guidance_scale=args.guidance_scale, noise_strength=args.noise_strength,
            resample_interval=args.resample_interval, leash=args.leash,
            output_path=args.output_path
        )

if __name__ == "__main__":
    main()
