# sms_embed.py
# Embeddings only (no centroid/footprint yet).
# Requirements: pip install sentence-transformers numpy pandas

from typing import Iterable, Optional, Tuple
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config
# ---------------------------
# Fast default (384-d). To switch to 768-d later, set:
# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# --- make sure we are fully offline ---
os.environ["HF_HUB_OFFLINE"] = "1"         # disables hub network calls
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # disables transformers downloads
os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
os.environ.pop("HF_TOKEN", None)

MODEL_NAME = r"C:/models/all-MiniLM-L6-v2"   # your local folder path

# Where to store vectors/metadata (Parquet recommended)
# - You can later move the metadata rows to sqlite if desired (id/text/shape).
DEFAULT_OUT_DIR = "./artifacts"   # change as needed

# ---------------------------
# Model loading
# ---------------------------
_model: Optional[SentenceTransformer] = None

def get_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the embedding model once (cached in process)."""
    global _model
    if _model is None or getattr(_model, "_model_name", None) != model_name:
        m = SentenceTransformer(model_name)
        m._model_name = model_name # type: ignore[attr-defined]
        _model = m
    return _model

# ---------------------------
# Embedding helpers
# ---------------------------
def embed_texts(
    texts: Iterable[str],
    batch_size: int = 64,
    normalize: bool = True,
    model_name: str = MODEL_NAME,
) -> np.ndarray:
    """
    Return an array of shape [N, D] for the given texts.
    - normalize=True -> L2 unit vectors (cosine-ready).
    """
    model = get_model(model_name)
    X = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return X.astype(np.float32, copy=False)

def embed_dedup_dataframe(
    dedup_df: pd.DataFrame,
    text_col: str = "normalized_text",
    id_col: str = "template_hash_xx64",
    batch_size: int = 64,
    normalize: bool = True,
    model_name: str = MODEL_NAME,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Embed a deduped dataframe (one row per unique message).
    Returns:
      - meta_df: DataFrame with [id_col, text_col, n_tokens (approx), vec_len]
      - X: np.ndarray shape [N, D] of embeddings (float32, unit length if normalize=True)
    """
    if text_col not in dedup_df.columns or id_col not in dedup_df.columns:
        raise ValueError(f"dedup_df must contain '{text_col}' and '{id_col}'")

    texts = dedup_df[text_col].astype(str).tolist()
    ids   = dedup_df[id_col].astype(str).tolist()

    X = embed_texts(texts, batch_size=batch_size, normalize=normalize, model_name=model_name)
    meta_df = pd.DataFrame({
        id_col: ids,
        text_col: texts,
        "vec_len": [X.shape[1]] * len(texts),
        # a cheap length proxy (can be useful for diagnostics)
        "n_chars": [len(t) for t in texts],
    })
    return meta_df, X

# ---------------------------
# Persistence
# ---------------------------
def save_embeddings(
    meta_df: pd.DataFrame,
    X: np.ndarray,
    out_dir: str = "./artifacts",
    prefix: str = "embeds",
) -> tuple[str, str]:
    """
    Save:
      - <out_dir>/<prefix>.csv : metadata rows (ids, text, vec_len, n_chars)
      - <out_dir>/<prefix>.npy : raw vectors (float32, [N, D])
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    npy_path = os.path.join(out_dir, f"{prefix}.npy")

    meta_df.to_csv(csv_path, index=False, encoding="utf-8")
    np.save(npy_path, X)

    return csv_path, npy_path

def load_embeddings(
    out_dir: str = DEFAULT_OUT_DIR,
    prefix: str = "embeds",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load metadata and vectors saved by save_embeddings.
    """
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    npy_path = os.path.join(out_dir, f"{prefix}.npy")
    meta_df = pd.read_csv(csv_path)
    X = np.load(npy_path, mmap_mode="r")  # memory-mapped; cast to contiguous if you need to write
    return meta_df, X
