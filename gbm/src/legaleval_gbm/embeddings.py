import hashlib
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from fastembed import TextEmbedding

E5_MODEL = "intfloat/multilingual-e5-large"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
MODEL_CACHE_DIR = Path.home() / ".cache" / "fastembed"


class EmbeddingModel(Protocol):
    def embed(self, documents: Iterable[str], batch_size: int = ...) -> Iterable[np.ndarray]: ...


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class CachedEmbedder:
    def __init__(self, model: EmbeddingModel, cache_path: Path, batch_size: int = 32, chunk_size: int = 4096):
        if cache_path.suffix != ".npz":
            raise ValueError("cache_path must end in .npz")
        self._model = model
        self._path = cache_path
        self._batch_size = batch_size
        self._chunk_size = chunk_size
        self._vectors = self._load()

    def _load(self) -> dict[str, np.ndarray]:
        if not self._path.exists():
            return {}
        with np.load(self._path) as stored:
            return dict(zip(stored["keys"].tolist(), stored["vectors"], strict=True))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._vectors))
        np.savez(self._path, keys=keys, vectors=np.vstack(list(self._vectors.values())))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [_key(t) for t in texts]
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys, strict=True) if k not in self._vectors))
        started = time.perf_counter()
        for start in range(0, len(missing), self._chunk_size):
            chunk = missing[start : start + self._chunk_size]
            fresh = np.array(list(self._model.embed(chunk, batch_size=self._batch_size)), dtype=np.float32)
            fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
            self._vectors.update(zip((_key(t) for t in chunk), fresh, strict=True))
            self._save()
            done = start + len(chunk)
            elapsed = time.perf_counter() - started
            print(f"embedded {done}/{len(missing)} new texts ({elapsed:.0f}s)", flush=True)
        return np.vstack([self._vectors[k] for k in keys])


def e5_embedder(cache_dir: Path) -> CachedEmbedder:
    model = TextEmbedding(E5_MODEL, cache_dir=str(MODEL_CACHE_DIR))
    return CachedEmbedder(model, cache_dir / "multilingual-e5-large.npz")
