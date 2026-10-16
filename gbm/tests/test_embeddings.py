import zlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest

from legaleval_gbm.data import BLOCKS
from legaleval_gbm.embeddings import PASSAGE_PREFIX, QUERY_PREFIX, CachedEmbedder
from legaleval_gbm.features import EMBEDDING_COLUMNS, embedding_features


class BagOfWordsModel:
    def __init__(self, dim: int = 4096) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, documents: Iterable[str], batch_size: int = 32) -> Iterable[np.ndarray]:
        batch = list(documents)
        self.calls.append(batch)
        for text in batch:
            vector = np.zeros(self.dim)
            for word in text.removeprefix(QUERY_PREFIX).removeprefix(PASSAGE_PREFIX).split():
                vector[zlib.crc32(word.encode()) % self.dim] += 1.0
            yield vector


class FailingModel:
    def embed(self, documents: Iterable[str], batch_size: int = 32) -> Iterable[np.ndarray]:
        raise AssertionError("cache miss")


def test_embedder_returns_unit_vectors_and_embeds_each_text_once(tmp_path):
    model = BagOfWordsModel()
    embedder = CachedEmbedder(model, tmp_path / "cache.npz")
    vectors = embedder.embed(["a b", "c", "a b"])
    embedder.embed(["c"])
    assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0, 1.0])
    assert vectors[0] == pytest.approx(vectors[2])
    assert model.calls == [["a b", "c"]]


def test_embedder_reads_vectors_back_from_disk(tmp_path):
    first = CachedEmbedder(BagOfWordsModel(), tmp_path / "cache.npz").embed(["a b", "c"])
    again = CachedEmbedder(FailingModel(), tmp_path / "cache.npz").embed(["c", "a b"])
    assert again == pytest.approx(first[::-1])


def test_embedder_rejects_non_npz_path(tmp_path):
    with pytest.raises(ValueError):
        CachedEmbedder(BagOfWordsModel(), tmp_path / "cache.bin")


def test_embedding_features_locate_the_copied_window(tmp_path):
    words = [f"w{i}" for i in range(150)]
    df = pd.DataFrame(
        {
            "row_id": [f"c1|{BLOCKS[0]}", f"c1|{BLOCKS[6]}"],
            "id_sud_resh": ["c1", "c1"],
            "block_name": [BLOCKS[0], BLOCKS[6]],
            "extracted_text": [" ".join(words[:50]), " ".join(words[100:])],
            "source_text": [" ".join(words)] * 2,
        }
    )
    features = embedding_features(df, CachedEmbedder(BagOfWordsModel(), tmp_path / "cache.npz"))
    assert list(features.columns) == list(EMBEDDING_COLUMNS)
    assert features["emb_max_sim"].to_numpy() == pytest.approx([1.0, 1.0])
    assert features["emb_best_pos"].to_numpy() == pytest.approx([0.0, 1.0])
    assert features["emb_cross_block_max"].to_numpy() == pytest.approx([0.0, 0.0], abs=0.05)
