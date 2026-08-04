from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import math
import struct
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Protocol


class EmbeddingBackend(Protocol):
    """Minimal async embedding interface used by the memory core."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def normalize_vector(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    magnitude = math.sqrt(sum(value * value for value in values))
    if not values or not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("embedding vector must have a finite non-zero norm")
    return [value / magnitude for value in values]


def encode_vector(vector: Sequence[float]) -> bytes:
    values = normalize_vector(vector)
    return struct.pack(f"<{len(values)}f", *values)


def decode_vector(value: bytes, dimensions: int) -> tuple[float, ...]:
    expected = int(dimensions) * 4
    if dimensions <= 0 or len(value) != expected:
        raise ValueError(
            f"invalid embedding blob: dimensions={dimensions}, bytes={len(value)}"
        )
    return struct.unpack(f"<{dimensions}f", value)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


class LocalFastEmbedBackend:
    """Plugin-owned CPU embedding backend powered by FastEmbed and ONNX.

    The model is loaded lazily on the first embedding request. Network access is
    used only when FastEmbed needs to download a missing model artifact; no text
    is ever sent to a remote inference API.
    """

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: str | Path,
        cpu_threads: int = 1,
        batch_size: int = 16,
        model_factory: Callable[..., object] | None = None,
    ):
        normalized_name = str(model_name).strip()
        if not normalized_name:
            raise ValueError("local embedding model name is required")
        self._model_name = normalized_name
        self._cache_dir = Path(cache_dir)
        self._cpu_threads = max(1, min(8, int(cpu_threads)))
        self._batch_size = max(1, min(128, int(batch_size)))
        self._model_factory = model_factory
        self._model: object | None = None
        self._dimensions = 0
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return f"fastembed/{self._model_name}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def dependency_available(self) -> bool:
        return self._model_factory is not None or importlib.util.find_spec(
            "fastembed"
        ) is not None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            factory = self._model_factory
            if factory is None:
                try:
                    from fastembed import TextEmbedding
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "fastembed is not installed; install plugin requirements"
                    ) from exc
                factory = TextEmbedding
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            model = factory(
                model_name=self._model_name,
                cache_dir=str(self._cache_dir),
                threads=self._cpu_threads,
                providers=["CPUExecutionProvider"],
                lazy_load=True,
            )
            dimensions = int(getattr(model, "embedding_size", 0))
            if dimensions <= 0:
                raise ValueError("local embedding model returned an invalid dimension")
            self._dimensions = dimensions
            self._model = model
            return model

    def _validate(self, vectors: Sequence[Any]) -> list[list[float]]:
        normalized: list[list[float]] = []
        for index, vector in enumerate(vectors):
            to_list = getattr(vector, "tolist", None)
            raw = to_list() if callable(to_list) else list(vector)
            if len(raw) != self.dimensions:
                raise ValueError(
                    "embedding dimension mismatch at index "
                    f"{index}: expected {self.dimensions}, got {len(raw)}"
                )
            normalized.append(normalize_vector(raw))
        return normalized

    def _embed_texts_sync(self, texts: list[str]) -> list[list[float]]:
        with self._inference_lock:
            model = self._load_model()
            embed = getattr(model, "passage_embed", None)
            if not callable(embed):
                raise TypeError("local embedding model does not implement passage_embed()")
            vectors = list(embed(texts, batch_size=self._batch_size))
            if len(vectors) != len(texts):
                raise ValueError(
                    f"embedding count mismatch: expected {len(texts)}, "
                    f"got {len(vectors)}"
                )
            return self._validate(vectors)

    def _embed_query_sync(self, text: str) -> list[float]:
        with self._inference_lock:
            model = self._load_model()
            embed = getattr(model, "query_embed", None)
            if not callable(embed):
                raise TypeError("local embedding model does not implement query_embed()")
            vectors = list(embed([text], batch_size=1))
            if len(vectors) != 1:
                raise ValueError(
                    f"query embedding count mismatch: expected 1, got {len(vectors)}"
                )
            return self._validate(vectors)[0]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        values = [str(text).strip() for text in texts]
        if not values:
            return []
        if any(not value for value in values):
            raise ValueError("embedding passages must not be empty")
        return await asyncio.to_thread(self._embed_texts_sync, values)

    async def embed_query(self, text: str) -> list[float]:
        value = str(text).strip()
        if not value:
            raise ValueError("embedding query must not be empty")
        return await asyncio.to_thread(self._embed_query_sync, value)


class LocalSentenceTransformerBackend:
    """Plugin-owned CPU embedding backend powered by Sentence Transformers.

    The model is loaded lazily and kept separate from AstrBot's embedding
    provider abstraction.  Query prompts are applied only to queries; passage
    vectors remain unprompted as required by asymmetric retrieval models such
    as Microsoft Harrier.
    """

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: str | Path,
        batch_size: int = 4,
        query_prompt_name: str = "",
        max_seq_length: int = 512,
        device: str = "cpu",
        model_factory: Callable[..., object] | None = None,
    ):
        normalized_name = str(model_name).strip()
        if not normalized_name:
            raise ValueError("local embedding model name is required")
        self._model_name = normalized_name
        self._cache_dir = Path(cache_dir)
        self._batch_size = max(1, min(32, int(batch_size)))
        self._query_prompt_name = str(query_prompt_name).strip()
        self._max_seq_length = max(32, min(4096, int(max_seq_length)))
        self._device = str(device).strip() or "cpu"
        self._model_factory = model_factory
        self._model: object | None = None
        self._dimensions = 0
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        prompt = self._query_prompt_name or "none"
        return (
            f"sentence-transformers/{self._model_name}"
            f"?query_prompt={prompt}"
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def dependency_available(self) -> bool:
        return self._model_factory is not None or importlib.util.find_spec(
            "sentence_transformers"
        ) is not None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            factory = self._model_factory
            if factory is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "sentence-transformers is not installed; install plugin "
                        "requirements"
                    ) from exc
                factory = SentenceTransformer
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            model = factory(
                self._model_name,
                device=self._device,
                cache_folder=str(self._cache_dir),
                trust_remote_code=False,
                model_kwargs={"dtype": "auto"},
            )
            current_max_length = int(
                getattr(model, "max_seq_length", self._max_seq_length)
                or self._max_seq_length
            )
            setattr(
                model,
                "max_seq_length",
                min(current_max_length, self._max_seq_length),
            )
            dimension_getter = getattr(model, "get_sentence_embedding_dimension", None)
            if not callable(dimension_getter):
                dimension_getter = getattr(model, "get_embedding_dimension", None)
            dimensions = int(dimension_getter() if callable(dimension_getter) else 0)
            if dimensions <= 0:
                raise ValueError("local embedding model returned an invalid dimension")
            self._dimensions = dimensions
            self._model = model
            return model

    def _validate(self, vectors: Sequence[Any]) -> list[list[float]]:
        normalized: list[list[float]] = []
        for index, vector in enumerate(vectors):
            to_list = getattr(vector, "tolist", None)
            raw = to_list() if callable(to_list) else list(vector)
            if len(raw) != self.dimensions:
                raise ValueError(
                    "embedding dimension mismatch at index "
                    f"{index}: expected {self.dimensions}, got {len(raw)}"
                )
            normalized.append(normalize_vector(raw))
        return normalized

    def _embed_sync(self, texts: list[str], *, query: bool) -> list[list[float]]:
        with self._inference_lock:
            model = self._load_model()
            encode = getattr(model, "encode", None)
            if not callable(encode):
                raise TypeError("local embedding model does not implement encode()")
            kwargs: dict[str, Any] = {
                "batch_size": 1 if query else self._batch_size,
                "normalize_embeddings": True,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            }
            if query and self._query_prompt_name:
                kwargs["prompt_name"] = self._query_prompt_name
            vectors = encode(texts, **kwargs)
            return self._validate(list(vectors))

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        values = [str(text).strip() for text in texts]
        if not values:
            return []
        if any(not value for value in values):
            raise ValueError("embedding passages must not be empty")
        return await asyncio.to_thread(self._embed_sync, values, query=False)

    async def embed_query(self, text: str) -> list[float]:
        value = str(text).strip()
        if not value:
            raise ValueError("embedding query must not be empty")
        vectors = await asyncio.to_thread(self._embed_sync, [value], query=True)
        return vectors[0]


class HashEmbeddingBackend:
    """Dependency-free character n-gram embedding for deterministic offline tests.

    This backend proves the vector indexing and candidate-initialization path. It
    is not intended to replace a semantic embedding model in production.
    """

    def __init__(self, dimensions: int = 256):
        if dimensions < 32:
            raise ValueError("hash embedding dimensions must be at least 32")
        self._dimensions = int(dimensions)

    @property
    def model_id(self) -> str:
        return f"hash-char-ngram-v1:{self.dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed(self, text: str) -> list[float]:
        normalized = " ".join(str(text).casefold().split())
        if not normalized:
            normalized = "<empty>"
        vector = [0.0] * self.dimensions
        features: list[str] = []
        for size in (1, 2, 3):
            features.extend(
                normalized[index : index + size]
                for index in range(max(0, len(normalized) - size + 1))
            )
        features.extend(normalized.split())
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return normalize_vector(vector)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
