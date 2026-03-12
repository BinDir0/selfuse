"""Test script for embedding analysis visualization from inference NPZs."""

import argparse
import pathlib

from src.utils.embedding_analysis import plot_tsne_from_npzs


def main():
    parser = argparse.ArgumentParser(
        description="Visualize hidden-state embeddings from saved inference NPZ files"
    )
    parser.add_argument(
        "--npz",
        type=str,
        nargs="+",
        required=True,
        help="One or more inference result NPZ paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/tsne_by_key.png",
        help="Output image path",
    )
    parser.add_argument(
        "--sample_per_npz",
        type=int,
        default=2000,
        help="How many embeddings to sample per NPZ per key",
    )
    args = parser.parse_args()

    npz_paths = [str(pathlib.Path(path)) for path in args.npz]
    keys = [
        "prefill_image_hidden_states",
        "prefill_state_hidden_states",
        "prefill_text_hidden_states",
        "generated_hidden_states",
    ]

    plot_tsne_from_npzs(
        npz_paths=npz_paths,
        keys=keys,
        sample_per_npz=args.sample_per_npz,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
