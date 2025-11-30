# new_data_helpers.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import os
import numpy as np
import pandas as pd
import hdbscan

from sms_norm import normalize_and_hash_series, dedupe_by_hash
from sms_embed import embed_dedup_dataframe
from llm_client_openai import summarize_samples


from params import (
    SIM_THRESHOLD,
    MIN_COVERAGE,
    MIN_UNASSIGNED,
    MIN_CLUSTER_MEMBERS,
    HDBSCAN_UNKNOWN_DEFAULT,
    N_SAMPLES_FOR_LLM,
    MAX_WORDS_LLM,
    CENTROID_SIM_THRESHOLD,
)


def _unit_normalize(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


# -------------------------------------------------------------------
# Stage 2a: embed NEW batch & assign to existing campaign centroids
# -------------------------------------------------------------------
def assign_new_batch_to_reference(
    data_dir: str,
    ref_prefix: str,
    new_batch_csv: str,
    sim_threshold: float = SIM_THRESHOLD,
    model_path: str = r"C:/models/all-MiniLM-L6-v2",
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Stage 2a:
    - Load saved reference centroids + campaigns for ref_prefix.
    - Load NEW batch CSV, normalize + dedupe (prototypes).
    - Embed prototypes.
    - Assign each prototype to nearest campaign centroid with a cosine threshold.
    - Save prototype-level CSV + NPY for downstream steps.

    Files used (under data_dir):
        {ref_prefix}_campaign_centroids.npy
        {ref_prefix}_campaigns.csv
        {ref_prefix}.csv                 # reference prototypes meta (not strictly needed here)

    Files produced:
        {ref_prefix}_new_prototypes.csv
        {ref_prefix}_new_prototypes.npy

    Returns
    -------
    meta_df : pd.DataFrame
        Prototype-level metadata for the NEW batch (with assignment columns).
    X_proto : np.ndarray
        Prototype embeddings array, shape [P, D].
    """
    # ---- paths ----
    centroids_path = os.path.join(data_dir, f"{ref_prefix}_campaign_centroids.npy")
    campaigns_csv = os.path.join(data_dir, f"{ref_prefix}_campaigns.csv")
    ref_meta_csv = os.path.join(data_dir, f"{ref_prefix}.csv")  # stage-1 prototypes meta

    missing = [
        p for p in [centroids_path, campaigns_csv, ref_meta_csv, new_batch_csv]
        if not os.path.exists(p)
    ]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    centroids = np.load(centroids_path).astype(np.float32)  # [K, D]
    campaigns_df = pd.read_csv(campaigns_csv, encoding="utf-8")
    # ref_meta_df = pd.read_csv(ref_meta_csv, encoding="utf-8")  # currently unused but kept for symmetry

    # normalize centroids for cosine
    centroids = _unit_normalize(centroids)

    # ---- load new batch and produce deduplicated prototypes (as Stage-1) ----
    df_new = pd.read_csv(new_batch_csv, encoding="utf-8")

    # Choose raw text column (Stage-1 used "raw_text")
    if "raw_text" in df_new.columns:
        raw_col = "raw_text"
    elif "text" in df_new.columns:
        raw_col = "text"
    else:
        # fallback: first object column
        obj_cols = [c for c in df_new.columns if df_new[c].dtype == object]
        if not obj_cols:
            raise ValueError("Could not find a text column in new_batch_csv.")
        raw_col = obj_cols[0]

    # Normalize + compute template hashes (identical function as Stage-1)
    norm_df = normalize_and_hash_series(df_new[raw_col].astype(str), seed=seed)

    # Preserve identifiers if present
    if "message_id" in df_new.columns:
        norm_df.insert(0, "message_id", df_new["message_id"].values)  # type: ignore
    if "originator_id" in df_new.columns and "originator_id" not in norm_df.columns:
        norm_df.insert(0, "originator_id", df_new["originator_id"].values)  # type: ignore

    # Deduplicate by hash -> dedup_df is prototypes (one row per unique normalized_text)
    dedup_df, _ = dedupe_by_hash(norm_df)

    n_total_rows = len(df_new)
    n_prototypes = len(dedup_df)
    if verbose:
        print(f"[Stage 2a] New batch rows = {n_total_rows}; dedup prototypes = {n_prototypes}")

    # ---- Embed prototypes ----
    meta_df, X_proto = embed_dedup_dataframe(
        dedup_df,
        text_col="normalized_text",
        id_col="template_hash_xx64",
        batch_size=64,
        normalize=True,
        model_name=model_path,
    )

    X_proto = _unit_normalize(np.asarray(X_proto, dtype=np.float32))

    # ---- Assign prototypes to nearest campaign centroids ----
    sims_proto = X_proto @ centroids.T  # [P, K]
    assigned_proto_idx = np.argmax(sims_proto, axis=1)
    assigned_proto_score = np.max(sims_proto, axis=1)
    proto_status = np.where(assigned_proto_score >= sim_threshold, "Known", "Unknown")

    # Attach assignment results to meta_df (prototype-level output)
    meta_df = meta_df.copy()
    meta_df["assigned_campaign_row_index"] = assigned_proto_idx
    meta_df["assigned_campaign_score"] = assigned_proto_score
    meta_df["assigned_campaign_label"] = [
        campaigns_df.loc[i, "cluster_label"] if i < len(campaigns_df) else None
        for i in assigned_proto_idx
    ]
    meta_df["assigned_campaign_name"] = [
        campaigns_df.loc[i, "campaign_name"] if i < len(campaigns_df) else ""
        for i in assigned_proto_idx
    ]
    meta_df["status"] = proto_status

    # ---- Save prototype-level assignments & embeddings ----
    proto_meta_path = os.path.join(data_dir, f"{ref_prefix}_new_prototypes.csv")
    proto_npy_path = os.path.join(data_dir, f"{ref_prefix}_new_prototypes.npy")

    meta_df.to_csv(proto_meta_path, index=False, encoding="utf-8")
    np.save(proto_npy_path, X_proto.astype(np.float32))

    if verbose:
        print(f"[Stage 2a] Saved new prototypes meta to: {proto_meta_path}")
        print(f"[Stage 2a] Saved new prototypes embeddings to: {proto_npy_path}")

    return meta_df, X_proto


# -------------------------------------------------------------------
# Stage 2b: cluster Unknown prototypes, LLM-summarize, propose names
# -------------------------------------------------------------------
def cluster_unknown_prototypes(
    data_dir: str,
    ref_prefix: str,
    min_coverage: float = MIN_COVERAGE,
    min_unassigned: int = MIN_UNASSIGNED,
    min_cluster_members: int = MIN_CLUSTER_MEMBERS,
    hdbscan_params: Optional[Dict[str, Any]] = None,
    n_samples: int = N_SAMPLES_FOR_LLM,
    max_words: int = MAX_WORDS_LLM,
    centroid_sim_threshold: float = CENTROID_SIM_THRESHOLD,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Stage 2b:
    - Load new prototypes meta + embeddings.
    - Check coverage of Known vs Unknown prototypes.
    - If enough Unknown and coverage < threshold, run HDBSCAN over Unknown.
    - For each cluster:
        * Compute centroid, get top-N samples, call LLM for proposed label.
    - Propagate proposed labels back into meta_df in conservative way.
    - Save updated meta_df for review.

    Files used:
        {ref_prefix}_new_prototypes.csv
        {ref_prefix}_new_prototypes.npy

    File produced:
        {ref_prefix}_new_prototypes_assignments_upserted.csv

    Returns
    -------
    meta_df : pd.DataFrame
        Updated prototypes DataFrame (with proposed_* columns).
    proposed_clusters : list of dict
        Metadata for each proposed Unknown cluster.
    """
    proto_meta_csv = os.path.join(data_dir, f"{ref_prefix}_new_prototypes.csv")
    proto_npy_path = os.path.join(data_dir, f"{ref_prefix}_new_prototypes.npy")
    output_path = os.path.join(data_dir, f"{ref_prefix}_new_prototypes_assignments_upserted.csv")

    if not os.path.exists(proto_meta_csv) or not os.path.exists(proto_npy_path):
        raise FileNotFoundError("Missing new_prototypes files. Run Stage 2a first.")

    meta_df = pd.read_csv(proto_meta_csv, encoding="utf-8")
    X = np.load(proto_npy_path).astype(np.float32)

    if len(meta_df) != X.shape[0]:
        raise ValueError("meta rows and embeddings rows mismatch in new_prototypes.")

    # Coverage (prototype-level)
    total_protos = len(meta_df)
    known_mask = meta_df["status"].astype(str).str.lower() == "known"
    n_known = int(known_mask.sum())
    coverage = n_known / total_protos if total_protos > 0 else 0.0

    if verbose:
        print(f"[Stage 2b] Total prototypes: {total_protos}, Known: {n_known}, coverage={coverage:.3f}")

    # Unknown prototypes
    unknown_idx = np.where(~known_mask)[0]
    n_unknown = len(unknown_idx)
    if verbose:
        print(f"[Stage 2b] Unknown prototypes: {n_unknown}")

    proposed_clusters: List[Dict[str, Any]] = []

    # Decide whether to cluster unknowns
    if coverage >= min_coverage:
        if verbose:
            print(f"Coverage {coverage:.3f} >= MIN_COVERAGE {min_coverage}; skipping HDBSCAN.")
        meta_df.to_csv(output_path, index=False, encoding="utf-8")
        if verbose:
            print("[Stage 2b] Saved (no-op):", output_path)
        return meta_df, proposed_clusters

    if n_unknown < min_unassigned:
        if verbose:
            print(f"Not enough unknown prototypes (n_unknown={n_unknown} < MIN_UNASSIGNED={min_unassigned}); skipping HDBSCAN.")
        meta_df.to_csv(output_path, index=False, encoding="utf-8")
        if verbose:
            print("[Stage 2b] Saved (no-op):", output_path)
        return meta_df, proposed_clusters

    # ---- Run HDBSCAN on unknown prototypes ----
    if verbose:
        print(f"[Stage 2b] Running HDBSCAN on {n_unknown} unknown prototypes...")

    X_unknown = X[unknown_idx]  # [n_unknown, D]
    params = dict(HDBSCAN_UNKNOWN_DEFAULT)
    if hdbscan_params is not None:
        params.update(hdbscan_params)

    clusterer = hdbscan.HDBSCAN(**params)  # type: ignore
    labels_unknown = clusterer.fit_predict(X_unknown)  # -1 = noise

    unique_labels, counts = np.unique(labels_unknown, return_counts=True)
    if verbose:
        print("[Stage 2b] HDBSCAN labels (label, count):", list(zip(unique_labels.tolist(), counts.tolist())))

    # ensure proposal columns exist
    meta_df = meta_df.copy()
    if "proposed_cluster_id" not in meta_df.columns:
        meta_df["proposed_cluster_id"] = pd.NA
    if "proposed_campaign_name" not in meta_df.columns:
        meta_df["proposed_campaign_name"] = ""
    if "proposed_campaign_score" not in meta_df.columns:
        meta_df["proposed_campaign_score"] = pd.NA

    # For each non-noise cluster
    for lbl in sorted([int(l) for l in unique_labels if l != -1]):
        member_pos = np.where(labels_unknown == lbl)[0]  # positions in X_unknown
        cluster_size = len(member_pos)
        if cluster_size < min_cluster_members:
            if verbose:
                print(f"Skipping cluster {lbl} (size {cluster_size} < MIN_CLUSTER_MEMBERS={min_cluster_members})")
            continue

        # global indices
        global_indices = unknown_idx[member_pos]

        # centroid
        centroid = X_unknown[member_pos].mean(axis=0)
        centroid = _unit_normalize(centroid[None, :])[0]

        # similarities to centroid
        sims = X_unknown[member_pos] @ centroid
        order = np.argsort(-sims)
        ordered_pos = member_pos[order]
        ordered_global_idx = unknown_idx[ordered_pos]

        k = min(n_samples, cluster_size)
        top_globals = ordered_global_idx[:k]
        samples = meta_df.loc[top_globals, "normalized_text"].astype(str).tolist()

        # LLM summarization
        proposed_label, raw_resp = summarize_samples(samples, max_words=max_words, model="gpt-4o-mini", temperature=0.0)

        proposed_clusters.append(
            {
                "cluster_label": int(lbl),
                "cluster_size": int(cluster_size),
                "centroid": centroid,
                "proposed_label": proposed_label or "",
                "raw_llm_response": raw_resp,
                "member_global_indices": ordered_global_idx.tolist(),
                "member_sims": sims[order].tolist(),
            }
        )

        if verbose:
            print(
                f"Found cluster {lbl} size={cluster_size} "
                f"proposed_label='{proposed_label}' (top-{k} sent to LLM)"
            )

    # Propagate proposed labels back into meta_df
    for pc in proposed_clusters:
        centroid = np.asarray(pc["centroid"], dtype=np.float32)
        members = pc["member_global_indices"]
        member_vecs = X[members]
        sims_to_centroid = member_vecs @ centroid

        for gi, sim_val in zip(members, sims_to_centroid):
            if sim_val >= centroid_sim_threshold:
                meta_df.at[gi, "proposed_cluster_id"] = int(pc["cluster_label"])
                meta_df.at[gi, "proposed_campaign_name"] = pc["proposed_label"]
                meta_df.at[gi, "proposed_campaign_score"] = float(sim_val)
            else:
                meta_df.at[gi, "proposed_cluster_id"] = int(pc["cluster_label"])
                meta_df.at[gi, "proposed_campaign_score"] = float(sim_val)
                # leave proposed_campaign_name blank

    meta_df.to_csv(output_path, index=False, encoding="utf-8")
    if verbose:
        print("[Stage 2b] Saved updated prototype assignments with proposed clusters to:", output_path)

    if verbose and proposed_clusters:
        print("\n[Stage 2b] Proposed clusters summary:")
        for pc in proposed_clusters:
            print(
                f" - cluster {pc['cluster_label']}: "
                f"size={pc['cluster_size']}, "
                f"proposed_label='{pc['proposed_label']}'"
            )
    elif verbose:
        print("[Stage 2b] No new clusters found by HDBSCAN.")

    return meta_df, proposed_clusters


# -------------------------------------------------------------------
# Stage 2c: export new campaigns in reference format
# -------------------------------------------------------------------
def export_new_campaigns(
    data_dir: str,
    ref_prefix: str,
    meta_df: pd.DataFrame,
    proposed_clusters: List[Dict[str, Any]],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Stage 2c:
    - From proposed_clusters + meta_df, build a campaigns-style DataFrame
      for the newly discovered Unknown campaigns.
    - Save to CSV.

    File produced:
        {ref_prefix}_new_campaigns.csv

    Returns
    -------
    new_campaigns_df : pd.DataFrame
        New campaigns in reference format (to be merged / reviewed).
    """
    if not proposed_clusters:
        if verbose:
            print("\n[Stage 2c] No new campaigns to export.")
        return pd.DataFrame()

    if verbose:
        print("\n[Stage 2c] Exporting new campaigns to reference format...")

    new_campaigns_records: List[Dict[str, Any]] = []

    for idx, pc in enumerate(proposed_clusters):
        cluster_lbl = pc["cluster_label"]
        cluster_size = pc["cluster_size"]
        member_indices = pc["member_global_indices"]

        cluster_messages = meta_df.loc[member_indices]

        # message count: use cluster size as proxy
        msg_count = cluster_size

        # determine date range
        if "timestamp" in meta_df.columns or "date" in meta_df.columns:
            ts_col = "timestamp" if "timestamp" in meta_df.columns else "date"
            dates = pd.to_datetime(cluster_messages[ts_col], errors="coerce")
            if dates.notna().any():
                date_range_start = dates.min()
                date_range_end = dates.max()
            else:
                now = pd.Timestamp.now()
                date_range_start = date_range_end = now
        else:
            now = pd.Timestamp.now()
            date_range_start = date_range_end = now

        new_campaigns_records.append(
            {
                "row_index": idx,
                "cluster_label": cluster_lbl,
                "proto_count": cluster_size,
                "msg_count": msg_count,
                "date_range_start": date_range_start.isoformat(),
                "date_range_end": date_range_end.isoformat(),
                "campaign_name": pc["proposed_label"],
                "status": "Unknown",  # needs review
            }
        )

    new_campaigns_df = pd.DataFrame(new_campaigns_records)

    new_campaigns_path = os.path.join(data_dir, f"{ref_prefix}_new_campaigns.csv")
    new_campaigns_df.to_csv(new_campaigns_path, index=False, encoding="utf-8")

    if verbose:
        print(f"[Stage 2c] Saved {len(new_campaigns_df)} new campaigns to: {new_campaigns_path}")
        print("\n[Stage 2c] New campaigns summary:")
        print(new_campaigns_df[["cluster_label", "proto_count", "campaign_name"]].to_string(index=False))

    return new_campaigns_df
