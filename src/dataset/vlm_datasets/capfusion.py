import os
import json
import requests
import io
from PIL import Image
from ..base_dataset import BaseImageDataset
from datasets import load_dataset, get_dataset_config_names
import numpy as np

class CapsFusionDataset(BaseImageDataset):
    """
    Dataset class for the BAAI/CapsFusion-120M dataset, loading from a local Parquet file.
    Inherits from BaseImageDataset.
    """
    def __init__(self, data_path=None, split='train', image_transform=None, text_transform=None):
        """
        Initializes the CapsFusion-120M dataset from a local Parquet file.

        Args:
            data_path (str): Path to the local parquet file or directory containing parquet files.
            split (str): The dataset split to load. Defaults to 'train'.
            image_transform (callable, optional): A function/transform to apply to the image.
            text_transform (callable, optional): A function/transform to apply to the caption.
        """
        if not data_path:
            raise ValueError("`data_path` must be provided to load a local dataset.")

        self.image_transform = image_transform
        self.text_transform = text_transform
        self.data_path = data_path
        self.split = split
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads CapsFusion-120M data from a local Parquet file.
        """
        print(f"Loading CapsFusion-120M data from local path: {self.data_path}")

        
        # Use the 'parquet' script to load local data files
        dataset = load_dataset(
            "parquet",
            data_files=f"{self.data_path}/*.parquet",
            split=self.split,
            cache_dir="/share_data/datasets/VLM/CapsFusion-120M/.hf_datasets_cache",
        )
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample by downloading the image from its URL.
        """
        sample = self.data[idx]
        image_url = sample['image_url']

        try:
            # Download the image from the URL
            image = Image.open(requests.get(image_url, stream=True).raw)
        except Exception as e:
            print(f"Warning: Could not download image {image_url}. Error: {e}. Using a placeholder image.")
            # Create a black placeholder image if download fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        answer = sample['capsfusion']
        question = "caption en\n"

        if self.image_transform:
            image = self.image_transform(image)
        
        if self.text_transform:
            answer = self.text_transform(answer)

        if isinstance(image, Image.Image):
            image_arr = np.array(image)  # (H,W,C)
            image_arr = np.transpose(image_arr, (2, 0, 1))  # (C,H,W)
        else:
            # assume it's already tensor-like (C,H,W)
            image_arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.array(image)
            if image_arr.ndim == 3 and image_arr.shape[-1] in (1,3):  # (H,W,C)
                image_arr = np.transpose(image_arr, (2,0,1))
        image_arr = image_arr[None, ...]  # [T=1, C, H, W]
        return {
            'image': image_arr,
            'question': question,
            'answer': answer,
        }

if __name__ == '__main__':
    print("CapsFusionDataset class defined to load from a local Parquet file.")
    print("This class inherits from BaseDataset and is specialized for CapsFusion-120M.")
    

    # --- USAGE EXAMPLE ---
    # **IMPORTANT**: Update this path to point to where you downloaded your Parquet file(s).
    # This can be a single file like '0000.parquet' or a directory.
    local_parquet_path = "/share_data/datasets/VLM/CapsFusion-120M"

    print(f"\nAttempting to load 'train' split from: {local_parquet_path}...")
    
    # Pass the local path to the dataset class
    dataset = CapsFusionDataset(data_path=local_parquet_path)

    print(f"Successfully loaded CapsFusion dataset metadata with {len(dataset)} samples.")
    
    if len(dataset) > 0:
        print("\nAttempting to fetch and download the first item...")
        sample = dataset[0]
        print("\nSample data from the first item:")
        print(f"  Image type: {type(sample['image'])}")
        print(f"  Image size: {sample['image'].shape}")
        print(f"  Question: {sample['question'].strip()}")
        print(f"  Answer: {sample['answer']}")
            


