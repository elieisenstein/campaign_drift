# params.py
"""
Global configuration parameters shared across Stage-1 and Stage-2.

Includes:
- Laptop-specific HDBSCAN settings
- File structure and prefixes
- Similarity thresholds
- LLM summarization settings
"""

import os

# ----------------------------------------------------------
# Laptop ID (set once in the main notebook)
# ----------------------------------------------------------
LAPTOP_ID = os.getenv("LAPTOP_ID", "LAPTOP_A")

# ----------------------------------------------------------
# Base directories
# ----------------------------------------------------------
DATA_DIR = "./data"
LOCAL_MODEL = r"C:/models/all-MiniLM-L6-v2"

# ----------------------------------------------------------
# File prefixes (Stage-1)
# ----------------------------------------------------------
INPUT_PREFIX = "input/ref"            # input raw CSV (Stage-1)
OUTPUT_PREFIX = "output/ref_proto"    # prefix for all reference artifacts (Stage-1 + Stage-2)

# ----------------------------------------------------------
# Stage-1 HDBSCAN parameters (different laptops)
# ----------------------------------------------------------
HDBSCAN_PARAMS_A = dict(
    min_cluster_size=5,
    min_samples=1,
    cluster_selection_epsilon=0.0,
    cluster_selection_method="eom",
    metric="euclidean",
)

HDBSCAN_PARAMS_B = dict(
    min_cluster_size=20,
    min_samples=10,
    cluster_selection_epsilon=0.1,
    cluster_selection_method="eom",
    metric="euclidean",
)

HDBSCAN_STAGE1 = HDBSCAN_PARAMS_A if LAPTOP_ID == "LAPTOP_A" else HDBSCAN_PARAMS_B

# ----------------------------------------------------------
# Stage-2 similarity + thresholds
# ----------------------------------------------------------
SIM_THRESHOLD = 0.5  # cosine similarity for Known vs Unknown

# ----------------------------------------------------------
# Stage-2b (clustering Unknown prototypes)
# ----------------------------------------------------------
MIN_COVERAGE = 0.999         # Skip HDBSCAN if most prototypes are Known
MIN_UNASSIGNED = 5           # Require at least this many Unknown prototypes
MIN_CLUSTER_MEMBERS = 3      # Filter small clusters

# Default HDBSCAN params for unknown-prototype clustering
HDBSCAN_UNKNOWN_DEFAULT = dict(
    min_cluster_size=5,
    min_samples=2,
    metric="euclidean",
)

# LLM settings
N_SAMPLES_FOR_LLM = 5        # number of examples to summarize
MAX_WORDS_LLM = 5            # max words in campaign label
CENTROID_SIM_THRESHOLD = 0.6 # threshold for propagating proposed labels

# ----------------------------------------------------------
# Convenience: resolve main dataset paths
# ----------------------------------------------------------
def resolve_path(prefix: str, suffix: str, data_dir: str = DATA_DIR) -> str:
    """Helper to build full paths inside DATA_DIR."""
    return os.path.join(data_dir, f"{prefix}{suffix}")


def adaptive_hdbscan_params(
    base_params: dict,
    n_points: int,
    min_cluster_frac: float = 0.01,
    min_cluster_size_floor: int = 5,
    min_samples_floor: int = 3,
) -> dict:
    """
    Build an HDBSCAN params dict adapted to dataset size.

    - base_params: default dict (e.g. HDBSCAN_STAGE1 / HDBSCAN_UNKNOWN_DEFAULT)
    - n_points:    number of points to cluster
    - min_cluster_frac: fraction of n_points for min_cluster_size (e.g. 0.01 = 1%)
    """
    if n_points <= 0:
        return dict(base_params)

    mcs = max(min_cluster_size_floor, int(min_cluster_frac * n_points))
    ms = max(min_samples_floor, int(n_points ** 0.5))

    params = dict(base_params)
    params["min_cluster_size"] = mcs
    # don't override if caller already set min_samples explicitly
    params.setdefault("min_samples", ms)

    return params

