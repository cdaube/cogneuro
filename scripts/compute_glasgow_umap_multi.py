"""
Compute 10 UMAP projections (seeds 0–9) from Glasgow embeddings and save the result.

Reads:   data/glasgow_embeddings.npy   — shape (N, D)
Writes:  data/glasgow_umap_coords_multi.npy  — shape (10, N, 2)

After running this, `make_glasgow_explorer.py` will pick up the cached file
and embed all 10 projections into the HTML so users can switch between them.

Usage:
    uv run python scripts/compute_glasgow_umap_multi.py
"""

import os
import numpy as np
import umap

N_RUNS = 10
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

EMB_FILE = os.path.join(DATA_DIR, "glasgow_embeddings.npy")
OUT_FILE = os.path.join(DATA_DIR, "glasgow_umap_coords_multi.npy")


def main():
    if not os.path.exists(EMB_FILE):
        raise FileNotFoundError(
            f"Embeddings not found: {EMB_FILE}\n"
            "Copy glasgow_embeddings.npy into data/ first."
        )

    print(f"Loading embeddings from {EMB_FILE}...")
    embeddings = np.load(EMB_FILE)
    print(f"  Shape: {embeddings.shape}")

    runs = []
    for seed in range(N_RUNS):
        print(f"  UMAP run {seed + 1}/{N_RUNS} (seed={seed})...", flush=True)
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            n_components=2,
            random_state=seed,
        )
        coords = reducer.fit_transform(embeddings)
        runs.append(coords)

    result = np.stack(runs, axis=0).astype(np.float32)  # (10, N, 2)
    np.save(OUT_FILE, result)
    print(f"\nSaved {N_RUNS} projections → {OUT_FILE}  (shape {result.shape})")


if __name__ == "__main__":
    main()
