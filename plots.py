# Plot: distinct color/marker per campaign; centroids as stars with black edge
import matplotlib.pyplot as plt
import numpy as np


def _nongray_colors(n: int):
    # Curated, high-contrast, non-gray colors (repeat if n > len(base))
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#4c72b0",
        "#dd8452", "#55a868", "#c44e52", "#8172b2", "#937860",
        "#da8bc3", "#8c8c00", "#00a2d3", "#a55194", "#636efa",
    ]
    if n <= len(base):
        return base[:n]
    # If more needed, cycle (colors stay non-gray)
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out

def plot_campaigns(X_2d, labels, cluster_ids, save_path, campaign_names=None):
    """
    Plot UMAP visualization with campaigns
    
    Parameters:
    -----------
    X_2d : array
        2D UMAP coordinates
    labels : array
        Cluster labels for each point
    cluster_ids : list
        List of cluster IDs to plot
    save_path : str
        Path to save the plot
    campaign_names : dict or list, optional
        If dict: maps cluster_id to campaign name
        If list: aligned with cluster_ids order
        If None: uses generic "Campaign {id}" labels
    """
    
    plt.figure(figsize=(10, 8), dpi=160)

    # --- Noise fixed gray ---
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx):
        plt.scatter(
            X_2d[noise_idx, 0], X_2d[noise_idx, 1],
            s=50, c="#9e9e9e", marker=".", alpha=0.7, label="Noise", linewidths=0
        )

    # --- Colors that are explicitly NOT gray ---
    colors = _nongray_colors(len(cluster_ids))

    # --- Clusters ---
    for i, c in enumerate(cluster_ids):
        idx = np.where(labels == c)[0]
        col = colors[i]
        
        # Get campaign name if provided
        if campaign_names is not None:
            if isinstance(campaign_names, dict):
                name = campaign_names.get(c, "Unknown")
            elif isinstance(campaign_names, list):
                name = campaign_names[i] if i < len(campaign_names) and campaign_names[i] else "Unknown"
            else:
                name = "Unknown"
        else:
            name = f"Campaign {c}"
        
        # Format legend label as "cluster_id: campaign_name"
        # Truncate name if too long for legend
        if len(name) > 30:
            name = name[:27] + "..."
        legend_label = f"{c}: {name}"
        
        # points
        plt.scatter(
            X_2d[idx, 0], X_2d[idx, 1],
            s=80, c=col, marker="o", alpha=0.85,
            linewidths=0.4, edgecolors="white", label=legend_label
        )
        # centroid (mean in 2D) with big star + black edge
        cx, cy = X_2d[idx].mean(axis=0)
        plt.scatter([cx], [cy],
            s=700, c=col, marker="*", edgecolors="black", linewidths=1.8,
            zorder=5  # No separate label for centroid to avoid legend clutter
        )
        # label
        plt.text(cx, cy, f"C{c}", fontsize=11, weight="bold", color="black",
                 ha="center", va="center", zorder=6,
                 bbox=dict(facecolor="white", edgecolor=col, boxstyle="round,pad=0.25", alpha=0.9))

    plt.title("UMAP (2D) + HDBSCAN campaigns\nColors = campaigns, Stars = centroids")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.grid(True, alpha=0.3)

    # de-dup legend (though with new approach we shouldn't have duplicates)
    handles, labels_ = plt.gca().get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels_):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    plt.legend(H, L, loc="best", fontsize=8, ncol=1, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.show()
    plt.close()


# === Debug plot (fixed): stars = mean of REF POINTS in UMAP-2D; colors aligned by labels ===
# Requirements:
#   - reducer: the SAME fitted UMAP reducer from Stage 5
#   - X_ref:  (M, D) reference message embeddings used in Stage 5
#   - ref_labels: (M,) HDBSCAN labels for X_ref (same labels used to define campaigns)
#   - C: (K, D) saved reference centroids (unit-norm)  [used only for legend/consistency, not for star positions]
#   - centroid_labels: optional (K,) array mapping each row in C to its cluster label; if None -> np.arange(K)
#   - X_new: (N, D) new unit-norm embeddings
#   - a_idx: (N,) argmax indices into rows of C
#   - noise_mask: (N,) boolean, True if 1-cosine > tau_d
# -------------------------------------------------------------------------------------------


def plot_ref_stars_mean2d_with_new(
    reducer,                # fitted UMAP reducer from Stage 5
    X_ref: np.ndarray,      # (M, D) reference embeddings (unit-norm)
    ref_labels: np.ndarray, # (M,) HDBSCAN labels for X_ref (e.g., {-1,0,1,...})
    C: np.ndarray,          # (K, D) centroids (unit-norm); row order may differ from labels
    X_new: np.ndarray,      # (N, D) new embeddings (unit-norm)
    a_idx: np.ndarray,      # (N,) nearest centroid row indices (into C)
    noise_mask: np.ndarray, # (N,) True => noise
    centroid_labels: np.ndarray | None = None, # (K,), row->cluster label; default [0..K-1]
    title: str = "Stage 6 Debug: Centroids + New Messages",
    save_path: str | None = None,
    X_ref_2d: np.ndarray | None = None,  # <-- NEW: exact Stage-5 2D coords (M,2)
):
    """
    Stars sit at the mean of the REFERENCE points in 2D per label.
    If X_ref_2d is provided (saved Stage-5 coords), stars will match the original plot exactly.
    Otherwise we fall back to reducer.transform(X_ref[ref_labels!=-1]) which can shift slightly.
    New points: colored by assigned centroid's LABEL if provided via centroid_labels; noise in gray.
    """
    # filter out noise labels in the reference set
    ref_valid = ref_labels != -1
    ref_labels_valid = ref_labels[ref_valid]
    X_ref_valid = X_ref[ref_valid]

    # pick 2D coords for reference points
    if X_ref_2d is not None:
        # assume X_ref_2d aligns row-wise with X_ref/ref_labels
        X_ref_2d_valid = X_ref_2d[ref_valid]
    else:
        X_ref_2d_valid = reducer.transform(X_ref_valid)

    # project new points to 2D with the same reducer
    X_new_2d = reducer.transform(X_new)

    # colors keyed by cluster labels present in the reference (excluding -1)
    cluster_ids = np.unique(ref_labels_valid)
    colors = _nongray_colors(len(cluster_ids))
    label_to_color = {lab: colors[i] for i, lab in enumerate(cluster_ids)}

    # row index in C -> cluster label mapping (for coloring new points)
    if centroid_labels is None:
        centroid_labels = np.arange(C.shape[0], dtype=int)

    # compute star positions as **mean of 2D reference points** per label
    label_to_star = {}
    for lab in cluster_ids:
        idx = np.where(ref_labels_valid == lab)[0]
        if idx.size:
            label_to_star[lab] = X_ref_2d_valid[idx].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)

    # 1) noise first
    noise_idx = np.where(noise_mask)[0]
    if noise_idx.size:
        ax.scatter(
            X_new_2d[noise_idx, 0], X_new_2d[noise_idx, 1],
            s=50, c="#9e9e9e", marker=".", alpha=0.7, linewidths=0, label="Noise"
        )

    # 2) covered new points colored by their assigned centroid's LABEL
    covered_idx = np.where(~noise_mask)[0]
    if covered_idx.size:
        new_labels = centroid_labels[a_idx[covered_idx]]
        for lab in cluster_ids:
            idx = covered_idx[new_labels == lab]
            if idx.size:
                ax.scatter(
                    X_new_2d[idx, 0], X_new_2d[idx, 1],
                    s=70, c=label_to_color[lab], marker="o", alpha=0.9,
                    edgecolors="white", linewidths=0.4, label=f"New → C{lab}"
                )

    # 3) centroid stars at mean-of-2D ref points, outlined in black
    for lab in cluster_ids:
        if lab in label_to_star:
            sx, sy = label_to_star[lab]
            ax.scatter(
                [sx], [sy], s=700, c=label_to_color[lab], marker="*",
                edgecolors="black", linewidths=1.8, zorder=5, label=f"Centroid C{lab}"
            )
            ax.text(
                sx, sy, f"C{lab}",
                fontsize=11, weight="bold", color="black",
                ha="center", va="center", zorder=6,
                bbox=dict(facecolor="white", edgecolor=label_to_color[lab], boxstyle="round,pad=0.25", alpha=0.9)
            )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.grid(True, alpha=0.3)

    # dedupe legend
    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    ax.legend(H, L, loc="best", fontsize=8, ncol=2, frameon=True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.show()
    plt.close()