# persist_utils.py
"""
Utility to persist campaign footprints and examples.
- save_campaign_footprint(out_dir, prefix, campaigns_df, centroids, campaign_examples_df, points_meta, points_2d)
Saves:
 - {out_dir}/{prefix}_campaigns.csv
 - {out_dir}/{prefix}_campaign_centroids.npy
 - {out_dir}/{prefix}_campaign_examples.csv
 - {out_dir}/{prefix}_points.csv  (meta + umap + label)
"""

import os
import numpy as np
import pandas as pd

def save_campaign_footprint(
    out_dir: str,
    prefix: str,
    campaigns_df: pd.DataFrame,
    centroids: np.ndarray,
    campaign_examples_df: pd.DataFrame,
):
    os.makedirs(out_dir, exist_ok=True)
    # filenames
    campaigns_csv = os.path.join(out_dir, f"{prefix}_campaigns.csv")
    centroids_npy = os.path.join(out_dir, f"{prefix}_campaign_centroids.npy")
    examples_csv = os.path.join(out_dir, f"{prefix}_campaign_examples.csv")
    points_csv = os.path.join(out_dir, f"{prefix}_points.csv")

    # Save campaigns meta (ensure deterministic order by row_index)
    if "row_index" in campaigns_df.columns:
        campaigns_df = campaigns_df.sort_values("row_index").reset_index(drop=True)
    campaigns_df.to_csv(campaigns_csv, index=False, encoding="utf-8")

    # Save centroids matrix
    if centroids is None:
        centroids = np.zeros((0, 0), dtype=np.float32)
    np.save(centroids_npy, centroids.astype(np.float32))

    # Save campaign examples
    if campaign_examples_df is None:
        campaign_examples_df = pd.DataFrame(columns=[
            "campaign_row_index", "campaign_label", "rank", "message_id",
            "template_hash_xx64", "text_sample", "sim_score", "count_in_window"
        ])
    campaign_examples_df.to_csv(examples_csv, index=False, encoding="utf-8")

    print("Saved artifacts:")
    print(" - campaigns_csv:", campaigns_csv)
    print(" - centroids_npy:", centroids_npy)
    print(" - campaign_examples_csv:", examples_csv)

