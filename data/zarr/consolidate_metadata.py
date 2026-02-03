import json
import zarr
zarr_paths = [
    '/share_data/guantianrui/datasets/taco/taco_train_filtered_v1.zarr', 
    '/share_data/guantianrui/datasets/Oakink-v2/oakink2_train_filtered_v1.zarr', 
    '/share_data/guantianrui/datasets/HoloAssist/holoassist_filtered_v2.zarr', 
    '/share_data/guantianrui/datasets/EgoDex/egodex_train_filtered_v2.zarr', 
    '/share_data/guantianrui/datasets/VITRA-1M/zarr/epic_train_rechunked.zarr', 
    '/share_data/guantianrui/datasets/VITRA-1M/zarr/ego4d_cooking_and_cleaning_train_rechunked.zarr', 
    '/share_data/guantianrui/datasets/VITRA-1M/zarr/ego4d_other_train_rechunked.zarr'
]

for zarr_path in zarr_paths:
    print(f"Consolidating metadata for {zarr_path}")
    try:
        def is_zarr_key(key: str) -> bool:
            return key.endswith(".zarray") or key.endswith(".zgroup") or key.endswith(".zattrs")

        def json_loads_safe(value):
            if isinstance(value, (bytes, bytearray)):
                return json.loads(value.decode("utf-8"))
            if isinstance(value, str):
                return json.loads(value)
            return json.loads(value.tobytes().decode("utf-8"))

        store = zarr.storage.DirectoryStore(zarr_path)
        bad_keys = []
        for key in list(store.keys()):
            if not is_zarr_key(key):
                continue
            try:
                json_loads_safe(store[key])
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                bad_keys.append((key, str(exc)))
        if bad_keys:
            print(f"  Found {len(bad_keys)} invalid metadata keys:")
            for key, reason in bad_keys:
                print(f"   - {key}: {reason}")
            confirm_1 = input("  Delete these keys? Type 'yes' to continue: ").strip().lower()
            if confirm_1 == "yes":
                confirm_2 = input("  Are you sure? Type 'yes' to delete: ").strip().lower()
                if confirm_2 == "yes":
                    for key, reason in bad_keys:
                        print(f"   - remove {key}: {reason}")
                        del store[key]
                else:
                    print("  Skip deletion.")
                    continue
            else:
                print("  Skip deletion.")
                continue
        zarr.consolidate_metadata(store)
        print("  ✓ consolidated")
    except Exception as exc:
        print(f"  ✗ failed: {exc}")
