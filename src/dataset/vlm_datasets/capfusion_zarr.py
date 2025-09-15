import os
import requests
import zarr
from tqdm import tqdm

# --- Dependency Imports ---
# You might need to install these: pip install datasets pillow numpy requests zarr tqdm
from datasets import load_dataset
from PIL import Image
import numpy as np
import numcodecs

# --- Dataset Class Definition (from your prompt) ---

# Define a dummy base class so the script is self-contained
class BaseImageDataset:
    def __init__(self):
        pass

class CapsFusionDataset(BaseImageDataset):
    """
    Dataset class for the BAAI/CapsFusion-120M dataset, loading from a local Parquet file.
    Inherits from BaseImageDataset.
    """
    def __init__(self, data_path=None, split='train', image_transform=None, text_transform=None):
        """
        Initializes the CapsFusion-120M dataset from a local Parquet file.
        """
        if not data_path:
            raise ValueError("`data_path` must be provided to load a local dataset.")

        self.image_transform = image_transform
        self.text_transform = text_transform
        self.data_path = data_path
        self.split = split
        # We only need the metadata, so we'll load it directly.
        self.data = self._load_data()

    def _load_data(self):
        """
        Loads CapsFusion-120M data from a local Parquet file.
        """
        print(f"Loading CapsFusion-120M metadata from local path: {self.data_path}")
        dataset = load_dataset(
            "parquet",
            data_files=f"{self.data_path}/*.parquet",
            split=self.split,
            cache_dir=os.path.join(self.data_path, ".hf_datasets_cache"),
        )
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        This method is kept for compatibility but our script will access .data directly
        for more efficient raw data retrieval.
        """
        sample = self.data[idx]
        return {
            'image_url': sample['image_url'],
            'question': "caption en\n",
            'answer': sample['capsfusion'],
        }

# --- Main Conversion Logic ---

def save_dataset_to_zarr(dataset, output_path, chunk_size=1000):
    """
    Downloads images from a dataset and saves the data to a Zarr store.
    Skips images that fail to download.

    Args:
        dataset (CapsFusionDataset): The initialized dataset object.
        output_path (str): The path to the output Zarr directory.
        chunk_size (int): The number of items to store in each Zarr chunk.
    """
    n_samples = len(dataset)
    print(f"Preparing to process {n_samples} samples and save to {output_path}")

    # Open the Zarr store in write mode
    root = zarr.open(output_path, mode='w')

    # Create resizable arrays for each data field.
    # We use dtype=object for variable-length data like image bytes and strings.
    image_array = root.create_dataset(
        'image',
        shape=(0,),
        chunks=(chunk_size,),
        dtype=object,
        object_codec=numcodecs.Pickle() # Efficient codec for raw bytes
    )
    question_array = root.create_dataset(
        'question',
        shape=(0,),
        chunks=(chunk_size,),
        dtype=object,
        object_codec=numcodecs.VLenUTF8() # Codec for UTF-8 strings
    )
    answer_array = root.create_dataset(
        'answer',
        shape=(0,),
        chunks=(chunk_size,),
        dtype=object,
        object_codec=numcodecs.VLenUTF8() # Codec for UTF-8 strings
    )

    skipped_count = 0
    # Use tqdm for a progress bar
    for idx in tqdm(range(n_samples), desc="Processing samples"):
        # Access the raw data directly to get the URL
        sample_meta = dataset.data[idx]
        image_url = sample_meta['image_url']
        answer = sample_meta['capsfusion']
        question = "caption en\n"

        try:
            # Download the image from the URL
            image = Image.open(requests.get(image_url, stream=True).raw)
        except Exception as e:
            print(f"Warning: Could not download image {image_url}. Error: {e}. Using a placeholder image.")
            skipped_count += 1
            continue # Move to the next item
        # Get the raw image bytes
        if isinstance(image, Image.Image):
            image_arr = np.array(image)  # (H,W,C)
            image_arr = np.transpose(image_arr, (2, 0, 1))  # (C,H,W)
        else:
            # assume it's already tensor-like (C,H,W)
            image_arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.array(image)
            if image_arr.ndim == 3 and image_arr.shape[-1] in (1,3):  # (H,W,C)
                image_arr = np.transpose(image_arr, (2,0,1))
        image_arr = image_arr[None, ...]  # [T=1, C, H, W]

        # Append the valid data to the Zarr arrays
        image_array.append([image_arr])
        question_array.append([question])
        answer_array.append([answer])


    print("\n--- Conversion Complete ---")
    print(f"Successfully saved {len(image_array)} samples to Zarr.")
    print(f"Skipped {skipped_count} samples due to download errors.")
    print(f"Zarr store location: {output_path}")

# --- USAGE EXAMPLE ---
if __name__ == '__main__':
    # **IMPORTANT**: Update this path to point to where you downloaded your Parquet file(s).
    local_parquet_path = "/share_data/datasets/VLM/CapsFusion-120M"
    
    # Define the output path for your Zarr dataset
    zarr_output_path = "/share_data/datasets/VLM/CapsFusion-120M/CapsFusion-120M.zarr"

    print("Step 1: Loading dataset metadata...")
    dataset = CapsFusionDataset(data_path=local_parquet_path)
    print(f"Successfully loaded metadata for {len(dataset)} samples.")

    print("\nStep 2: Starting conversion to Zarr format...")
    save_dataset_to_zarr(dataset, zarr_output_path)

    # --- VERIFICATION STEP ---
    print("\nStep 3: Verifying the created Zarr store...")
    try:
        # Load the Zarr store in read-only mode
        saved_data = zarr.open(zarr_output_path, mode='r')
        
        print(f"Zarr store keys: {list(saved_data.keys())}")
        print(f"Total items in store: {len(saved_data['image'])}")

        if len(saved_data['image']) > 0:
            print("\nInspecting the first saved sample:")
            
            print(f"  Image size: {saved_data['image'][0].shape}")
            print(f"  Question: {saved_data['question'][0].strip()}")
            print(f"  Answer: {saved_data['answer'][0]}")
            # img.show() # Uncomment to display the image
        else:
            print("Zarr store is empty. No data was saved.")
            
    except Exception as e:
        print(f"An error occurred during verification: {e}")