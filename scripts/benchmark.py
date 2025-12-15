import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from facenet_pytorch import MTCNN
import random

import sys
sys.path.append(".")
from src.inference_utils import smc_inference_core
from src.models import get_vae, get_unet, get_emotion_classifier
from src.config import TRAIN_CONFIG, DEVICE, BENCHMARK_CONFIG

def run_benchmark(data_dir, unet_path, images_per_test=10):
    print("Initializing benchmark...")
    
    if not os.path.exists(data_dir):
        print(f"Path {data_dir} not found.")
        return

    # Select first N images
    all_files = sorted(os.listdir(data_dir))
    image_paths = []
    for filename in all_files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(data_dir, filename))
        if len(image_paths) >= images_per_test:
            break
    
    print(f"Selected {len(image_paths)} images for testing.")

    # Load Models
    print("Loading models...")
    if os.path.exists(unet_path):
        from diffusers import UNet2DModel
        unet = UNet2DModel.from_pretrained(unet_path)
    else:
        print(f"Checkpoint {unet_path} not found.")
        return
    
    unet.to(DEVICE)
    unet.eval()
    
    vae = get_vae().to(DEVICE)
    processor, classifier = get_emotion_classifier()
    classifier.to(DEVICE)
    
    print("Initializing MTCNN...")
    mtcnn = MTCNN(keep_all=True, device=DEVICE)
    
    # Construct tests from BENCHMARK_CONFIG
    tests = []
    for t in BENCHMARK_CONFIG["tests"]:
        test_entry = {
            "name": f"{t['var']} Test",
            "fixed": BENCHMARK_CONFIG["fixed"],
            "var": t["var"],
            "values": t["values"]
        }
        tests.append(test_entry)
    
    results_log = []

    for test in tests:
        print(f"\n=== Running {test['name']} ===")
        for val in test['values']:
            params = test['fixed'].copy()
            params[test['var']] = val
            
            print(f"Testing {test['var']} = {val} ...")
            
            scores = []
            face_probs = []
            
            for img_path in tqdm(image_paths, leave=False):
                target_emotion = random.choice(["happy", "sad", "angry", "surprise", "neutral", "fear", "disgust"])
                final_img_tensor, score = smc_inference_core(
                    img_path, unet, vae, classifier, processor, target_emotion, 
                    num_particles=params.get("particles"),
                    steps=params.get("steps"),
                    guidance_scale=params.get("guidance_scale"),
                    noise_strength=params.get("noise_strength"),
                    leash=0.0,
                    resample_interval=1
                )
                
                if final_img_tensor is not None:
                    scores.append(score)
                    
                    try:
                        img_np = final_img_tensor.cpu().permute(1, 2, 0).numpy()
                        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
                        _, probs = mtcnn.detect(img_pil)
                        if probs is not None and len(probs) > 0:
                            face_probs.append(np.max(probs))
                        else:
                            face_probs.append(0.0)
                    except Exception as e:
                        print(f"MTCNN Error: {e}")
                        face_probs.append(0.0)
                else:
                    scores.append(0.0)
                    face_probs.append(0.0)
                    
            avg_score = np.mean(scores) if scores else 0.0
            avg_face_prob = np.mean(face_probs) if face_probs else 0.0
            
            print(f"-> Result: Avg Emotion Score={avg_score:.4f}, Avg Face Prob={avg_face_prob:.4f}")
            
            entry = params.copy()
            entry["tested_var"] = test['var']
            entry["tested_value"] = val
            entry["Avg_Emotion_Score"] = avg_score
            entry["Avg_Face_Prob"] = avg_face_prob
            results_log.append(entry)
            
    df_results = pd.DataFrame(results_log)
    print("\n=== Complete Results ===")
    print(df_results)
    df_results.to_csv("benchmark_results.csv", index=False)
    print("Results saved to benchmark_results.csv")

def main():
    parser = argparse.ArgumentParser(description="Run Benchmark Tests")
    parser.add_argument("--data_dir", type=str, default=BENCHMARK_CONFIG["data_dir"], help="Path to dataset")
    parser.add_argument("--unet_path", type=str, default=BENCHMARK_CONFIG["trained_unet_path"], help="Path to trained UNet")
    parser.add_argument("--images_per_test", type=int, default=10, help="Number of images per test")
    args = parser.parse_args()
    
    run_benchmark(args.data_dir, args.unet_path, args.images_per_test)

if __name__ == "__main__":
    main()
