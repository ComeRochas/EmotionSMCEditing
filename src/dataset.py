import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset(download_path="./data"):
    """
    Downloads the CelebA-HQ dataset from Kaggle.
    Requires kaggle.json to be set up.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    api = KaggleApi()
    api.authenticate()
    
    dataset_name = "badasstechie/celebahq-resized-256x256"
    
    print(f"Downloading {dataset_name} to {download_path}...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print("Download complete.")

class CelebAHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        # Find all jpg/png images recursively
        self.image_paths = glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True) + \
                           glob.glob(os.path.join(root_dir, "**", "*.png"), recursive=True)
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
            
        if self.transform:
            image = self.transform(image)
            
        return image

def get_transforms(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
    ])
