"""Frozen text encoders and the instruction feature files L1 reads.

Features are computed once per split and stored as data/features/text/<encoder>_<split>.npy,
one row per cache row in the same order, computed once. A sidecar .json keeps the sha256
of the instruction strings so a stale or reordered file is caught before training.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "clip": "openai/clip-vit-base-patch32",
    "map": "nuscenes-map-expansion-v1.3 lane features (drivex26.map_features v1, not a text encoder)",
    "intent": "Qwen/Qwen2.5-3B-Instruct structured intents, one-hot (scripts/language/encode_intent.py)",
}


def encode(name: str, texts: list[str]) -> tuple[np.ndarray, dict]:
    """L2-normalised embeddings (N, D) float32 and provenance info, CPU only."""
    import torch
    t0 = time.perf_counter()
    if name == "clip":
        from transformers import CLIPModel, CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained(ENCODERS[name])
        model = CLIPModel.from_pretrained(ENCODERS[name]).eval()
        with torch.no_grad():
            batch = tok(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
            emb = model.get_text_features(**batch)
            emb = getattr(emb, "pooler_output", emb)  # transformers >= 5 may return a model output
        emb = torch.nn.functional.normalize(emb, dim=-1).numpy()
        n_params = sum(p.numel() for p in model.text_model.parameters()) + model.text_projection.weight.numel()
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(ENCODERS[name], device="cpu")
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        n_params = sum(p.numel() for p in model.parameters())
    info = {"hf_id": ENCODERS[name], "dim": int(emb.shape[1]), "params_m": round(n_params / 1e6, 1),
            "cpu_sec_per_1k": round((time.perf_counter() - t0) / len(texts) * 1000, 2)}
    return emb.astype(np.float32), info


def texts_sha256(texts) -> str:
    return hashlib.sha256("\n".join(str(t) for t in texts).encode()).hexdigest()


def save_features(feature_dir, name: str, split: str, emb: np.ndarray, texts, info: dict) -> Path:
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    path = feature_dir / f"{name}_{split}.npy"
    np.save(path, emb)
    meta = {**info, "split": split, "n": int(len(emb)), "texts_sha256": texts_sha256(texts)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def load_features(feature_dir, name: str, split: str, texts) -> np.ndarray:
    """Embeddings for `split`, checked against the cache's instruction column."""
    path = Path(feature_dir) / f"{name}_{split}.npy"
    emb = np.load(path).astype(np.float32)
    if emb.ndim != 2 or len(emb) != len(texts):
        raise ValueError(f"{path}: shape {emb.shape} does not match {len(texts)} cache rows")
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing; rebuild with scripts/language/encode_text.py")
    if json.loads(meta_path.read_text())["texts_sha256"] != texts_sha256(texts):
        raise ValueError(f"{path} was built from different instructions than the cache")
    if not np.isfinite(emb).all():
        raise ValueError(f"{path}: nonfinite embeddings")
    return emb


def shuffle_across_scenes(scene_names: np.ndarray, seed: int = 0) -> np.ndarray:
    """Row index permutation where every row gets the instruction of a *different* scene.

    Sanity check: if L1 uses language, ADE must get worse under this permutation.
    """
    scene_names = np.asarray(scene_names)
    if len(np.unique(scene_names)) < 2:
        raise ValueError("need at least two scenes to shuffle across scenes")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(scene_names))
    for _ in range(1000):
        clash = np.flatnonzero(scene_names[perm] == scene_names)
        if not len(clash):
            return perm
        # swap each clashing row with a random row; keeps perm a permutation
        other = rng.integers(0, len(perm), size=len(clash))
        for i, j in zip(clash, other):
            perm[i], perm[j] = perm[j], perm[i]
    raise RuntimeError("could not build a cross-scene shuffle")
