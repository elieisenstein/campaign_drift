# ref_data_helpers.py

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List

import os
import numpy as np
import pandas as pd
import hdbscan

from sms_norm import normalize_and_hash_series, dedupe_by_hash
from sms_embed import embed_dedup_dataframe, save_embeddings, load_embeddings
#from llm_client_openai import summarize_samples
from llm_client_azure_openai import summarize_samples # alternate LLM client for Azure OpenAI
from persist_utils import save_campaign_footprint

def _debug_print_hdbscan_params(params: Dict[str, Any]) -> None:
    print("[HDBSCAN] Using parameters:")
    for k, v in params.items():
        print(f"  {k} = {v}")


def build_reference_embeddings_from_csv(
    csv_path: str,
    data_dir: str,
    prefix: str,
    text_col: str = "raw_text",
    model_path: str = r"./models/all-MiniLM-L6-v2",
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Stage 1 (part 1):
    - Load raw reference batch from CSV.
    - Normalize + dedupe templates.
    - Embed with offline MiniLM.
    - Save meta CSV + NPY via save_embeddings and return them.

    Returns
    -------
    meta_df : pd.DataFrame
        Deduplicated prototypes metadata.
    X : np.ndarray
        Embedding matrix, shape [num_prototypes, dim].
    """
    df_raw = pd.read_csv(csv_path, encoding="utf-8")

    # Normalize + dedupe
    norm = normalize_and_hash_series(df_raw[text_col].astype(str), seed=seed)

    # Preserve identifiers if present
    if "originator_id" in df_raw.columns:
        norm.insert(0, "originator_id", df_raw["originator_id"].values)  # type: ignore
    if "message_id" in df_raw.columns:
        # Make sure message_id is before normalized_text/template_hash
        if "message_id" not in norm.columns:
            norm.insert(1, "message_id", df_raw["message_id"].values)  # type: ignore

    dedup_df, _ = dedupe_by_hash(norm)

    if verbose:
        n_total_rows = len(norm)
        n_prototypes = len(dedup_df)
        print(f"[Stage 1] rows = {n_total_rows}; prototypes = {n_prototypes}")

    # Embed (offline MiniLM) and save CSV+NPY
    meta_df, X = embed_dedup_dataframe(
        dedup_df,
        text_col="normalized_text",
        id_col="template_hash_xx64",
        batch_size=64,
        normalize=True,
        model_name=model_path,
    )

    csv_path_saved, npy_path_saved = save_embeddings(
        meta_df, X, out_dir=data_dir, prefix=prefix
    )

    if verbose:
        print(f"[Stage 1] Saved embeddings to: {csv_path_saved}, {npy_path_saved}")

    return meta_df, X


def _unit_normalize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


def build_reference_profile(
    data_dir: str,
    prefix: str,
    hdbscan_params: Optional[Dict[str, Any]] = None,
    n_nearest: int = 5,
    verbose: bool = True,
    write_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Stage 1 (part 2):
    - Load prototypes meta + embeddings from artifacts.
    - Cluster prototypes with HDBSCAN.
    - Compute centroids + nearest examples.
    - Call LLM to summarize each campaign.
    - Persist campaigns + centroids + examples via save_campaign_footprint.

    Returns
    -------
    campaigns_df : pd.DataFrame
        One row per campaign (cluster), including derived metadata
        and the LLM campaign_name.
    examples_df : pd.DataFrame
        Nearest examples for each campaign.
    C : np.ndarray
        Centroids matrix, shape [num_campaigns, dim].
    """
    # Load embeddings as saved by build_reference_embeddings_from_csv / save_embeddings
    meta, X = load_embeddings(out_dir=data_dir, prefix=prefix, encoding="utf-8")

    X = _unit_normalize(X)

    if verbose:
        _debug_print_hdbscan_params(hdbscan_params)

    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    labels = clusterer.fit_predict(X)

    # Ignore noise
    is_valid = labels != -1
    labels_valid = labels[is_valid]
    unique_clusters = sorted(set(int(l) for l in labels_valid))
    
    # ------------------------------------------------------
    # If HDBSCAN found NO clusters (all noise), bail out early
    # ------------------------------------------------------
    if len(unique_clusters) == 0:
        if verbose:
            print(
                "[Stage 1] WARNING: no clusters found for "
                f"prefix={prefix} (all prototypes are noise)."
            )

        # Empty centroid matrix: 0 x dim
        # If you don't have X in scope here yet, you can just use 0x0,
        # it won't be used downstream in your current Stage 1 flow.
        C = np.zeros((0, 0), dtype=np.float32)

        # Minimal empty DataFrames — good enough for export_campaign_names_csv
        campaigns_df = pd.DataFrame(columns=["campaign_name"])
        examples_df = pd.DataFrame(columns=["campaign_name"])

        # Optional: if you don't want to write any footprint artifacts
        # in this case, just skip save_campaign_footprint entirely.
        # If you *do* want to persist an empty footprint, you can add:
        #
        # if write_outputs:
        #     save_campaign_footprint(
        #         out_dir=data_dir,
        #         prefix=prefix,
        #         campaigns_df=campaigns_df,
        #         centroids=C,
        #         campaign_examples_df=examples_df,
        #     )

        return campaigns_df, examples_df, C


    if verbose:
        n_total = len(labels)
        n_noise = int((labels == -1).sum())
        n_clusters = len(unique_clusters)
        print(
            f"[Stage 1] HDBSCAN: total={n_total}, noise={n_noise}, "
            f"clusters={n_clusters}"
        )

    centroids: List[np.ndarray] = []
    campaign_rows: List[Dict[str, Any]] = []
    examples_rows: List[Dict[str, Any]] = []

    for cluster_label in unique_clusters:
        member_idx = np.where(labels == cluster_label)[0]
        if member_idx.size == 0:
            continue

        # centroid
        cvec = X[member_idx].mean(axis=0)
        cvec = _unit_normalize(cvec[None, :])[0]
        centroids.append(cvec)

        # campaign row index (in campaigns_df)
        row_index = len(campaign_rows)

        # counts
        if "count_in_window" in meta.columns:
            msg_count = int(meta.iloc[member_idx]["count_in_window"].sum())
        else:
            msg_count = int(member_idx.size)

        # date range if available
        if {"window_start", "window_end"}.issubset(meta.columns):
            dr_start = meta.iloc[member_idx]["window_start"].min()
            dr_end = meta.iloc[member_idx]["window_end"].max()
        elif {"timestamp"}.issubset(meta.columns):
            dr_start = meta.iloc[member_idx]["timestamp"].min()
            dr_end = meta.iloc[member_idx]["timestamp"].max()
        else:
            dr_start = None
            dr_end = None

        campaign_rows.append(
            {
                "row_index": row_index,
                "cluster_label": int(cluster_label),
                "prototypes_count": int(member_idx.size),
                "messages_count": msg_count,
                "window_start": dr_start,
                "window_end": dr_end,
            }
        )

        # nearest examples by cosine similarity to centroid
        sims = X[member_idx] @ cvec
        order = np.argsort(-sims)
        k = min(n_nearest, order.size)

        for rank, idx_local in enumerate(order[:k], start=1):
            global_i = member_idx[idx_local]
            sim_value = float(sims[idx_local])
            row_meta = meta.iloc[global_i]

            examples_rows.append(
                {
                    "campaign_row_index": row_index,
                    "cluster_label": int(cluster_label),
                    "rank": rank,
                    "message_id": row_meta.get("message_id", ""),
                    "template_hash_xx64": row_meta.get("template_hash_xx64", ""),
                    # Prefer normalized_text (masked)
                    "text_sample": row_meta.get(
                        "normalized_text", row_meta.get("raw_text", "")
                    ),
                    "sim_score": sim_value,
                    "count_in_window": int(row_meta.get("count_in_window", 1)),  # type: ignore
                }
            )

    if centroids:
        C = np.vstack(centroids)
    else:
        # no clusters found
        C = np.zeros((0, X.shape[1]), dtype=np.float32)
        if verbose:
            print("[Stage 1] WARNING: no clusters found (all prototypes are noise).")

    campaigns_df = pd.DataFrame(campaign_rows)
    examples_df = pd.DataFrame(examples_rows)

    # LLM summaries for each campaign
    campaign_names: List[str] = []
    for row in campaigns_df.itertuples(index=False):
        row_idx = int(row.row_index)  # type: ignore
        samples = (
            examples_df[examples_df["campaign_row_index"] == row_idx]
            .sort_values("rank")["text_sample"]
            .tolist()
        )

        if samples:
            label, raw = summarize_samples(samples)
            summary = label
        else:
            summary = ""

        campaign_names.append(summary)

    campaigns_df["campaign_name"] = campaign_names

    #add campaign name col to examples
    examples_df = examples_df.merge(
    campaigns_df[["cluster_label", "campaign_name"]],
    on="cluster_label",
    how="left"
)


    # Persist all artifacts in a single helper
    if write_outputs:
        save_campaign_footprint(
            out_dir=data_dir,
            prefix=prefix,
            campaigns_df=campaigns_df,
            centroids=C,
            campaign_examples_df=examples_df,
        )

        if verbose:
            print(
                f"[Stage 1] Reference build complete. "
                f"Saved artifacts for prefix={prefix} under {data_dir}"
            )
        else:
            if verbose:
                print(
                    f"[Stage 1] Reference build complete for prefix={prefix}. "
                    f"artifacts NOT saved for prefix={prefix} under {data_dir}"
                )

    return campaigns_df, examples_df, C

from pathlib import Path
import pandas as pd

def export_campaign_names_csv(
    campaigns_df: pd.DataFrame,
    out_dir: str,
    originator: str,
    filename: str | None = None,
) -> str:
    """
    Save campaign names for an originator to a CSV.

    Behavior:
    - If campaigns_df is empty: writes a CSV with only the header 'campaign_name'.
    - Otherwise: extracts unique campaign names and writes them.

    Returns path to CSV.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"{originator}_campaign_names.csv"

    csv_path = out_path / filename

    # -----------------------------------------------------------
    # Case 1: no campaigns found → write an EMPTY CSV with header
    # -----------------------------------------------------------
    if campaigns_df.empty:
        pd.DataFrame(columns=["campaign_name"]).to_csv(
            csv_path, index=False, encoding="utf-8"
        )
        print(f"[Stage 1] No campaigns found. Wrote EMPTY CSV: {csv_path}")
        return str(csv_path)

    # -----------------------------------------------------------
    # Case 2: normal campaigns
    # -----------------------------------------------------------
    (
        campaigns_df["campaign_name"]
        .dropna()
        .drop_duplicates()
        .to_frame("campaign_name")
        .to_csv(csv_path, index=False, encoding="utf-8")
    )

    print(f"[Stage 1] Saved campaign-name CSV to: {csv_path}")
    return str(csv_path)

