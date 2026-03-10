import os                                                                                     
import webdataset as wds                                                                      
                                                                                            
tar_path = "/share_data/guantianrui/datasets/Webdataset/epic_train_rechunked/shard-w0120-000000.tar"  # your tar file path          
output_dir = "outputs/preview_samples"
start_idx = 114
num_samples = 5

os.makedirs(output_dir, exist_ok=True)

dataset = wds.WebDataset(tar_path).decode("pil")

for i, sample in enumerate(dataset):
    if i < start_idx:
        continue
    if i >= start_idx + num_samples:
        break

    key = sample["__key__"]  # e.g. "000000"
    sample_dir = os.path.join(output_dir, key)
    os.makedirs(sample_dir, exist_ok=True)

    for field, value in sample.items():
        if field.startswith("__"):
            continue
        filepath = os.path.join(sample_dir, f"{key}.{field}")
        if hasattr(value, "save"):  # PIL Image
            value.save(filepath)
        elif isinstance(value, bytes):
            with open(filepath, "wb") as f:
                f.write(value)
        else:
            with open(filepath, "w") as f:
                f.write(str(value))

    print(f"Saved sample {i}: {key}")
