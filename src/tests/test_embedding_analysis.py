"""Test script for embedding analysis visualization."""

from src.utils.embedding_analysis import plot_tsne_from_zarrs


def main():
    zarr_paths = [
        "/home/chenzhang/projects/diffloss-ar/outputs/2026.02.12/18.27_legendvla_inference/inference_results.zarr",
    ]
    keys = [
        "prefill_image_hidden_states",
        "prefill_state_hidden_states",
        "prefill_text_hidden_states",
        "generated_hidden_states",
    ]
    plot_tsne_from_zarrs(
        zarr_paths=zarr_paths,
        keys=keys,
        sample_per_zarr=2000,
        save_path="outputs/tsne_by_key.png",
    )


if __name__ == "__main__":
    main()
