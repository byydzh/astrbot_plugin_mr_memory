"""Local Sentence Transformers encoder and float32 vector search."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np

from .settings import resolve_query_prompt


class Embedder:
    def __init__(
        self,
        model_name: str,
        cache_dir: str | Path,
        threads: int = 1,
        batch_size: int = 1,
        query_prompt: str = "auto",
        max_seq_length: int = 512,
    ):
        if not model_name.strip() or min(threads, batch_size, max_seq_length) < 1:
            raise ValueError("model name and positive embedding limits are required")
        self.model_name = model_name.strip()
        self.cache_dir = Path(cache_dir)
        self.threads = int(threads)
        self.batch_size = int(batch_size)
        self.query_prompt = resolve_query_prompt(self.model_name, query_prompt)
        self.max_seq_length = int(max_seq_length)
        self.model_id = (
            f"sentence-transformers/{self.model_name}"
            f"?query_prompt={self.query_prompt or 'none'}"
        )
        self.dimensions = 0
        self._model = None
        # Cancelling an async waiter cannot stop native inference. One executor
        # worker keeps it serialized with the next request even after cancellation.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mr-embedding")
        self._closed = False

    async def _run(self, function, *args, **kwargs):
        if self._closed:
            raise RuntimeError("embedding service is closed")
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, partial(function, *args, **kwargs),
        )

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(self.threads)
            model = SentenceTransformer(
                self.model_name,
                device="cpu",
                cache_folder=str(self.cache_dir),
                trust_remote_code=False,
                model_kwargs={"dtype": "auto"},
            )
            model.max_seq_length = min(int(model.max_seq_length), self.max_seq_length)
            dimensions = int(model.get_sentence_embedding_dimension())
            if dimensions <= 0:
                raise ValueError("local embedding model has no valid dimension")
            self.dimensions = dimensions
            self._model = model
        return self._model

    async def warmup(self) -> None:
        await self._run(self._load)

    def _encode(self, texts: list[str], *, query: bool) -> list[list[float]]:
        model = self._load()
        prompt = {"prompt_name": self.query_prompt} if query and self.query_prompt else {"prompt": ""}
        vectors = np.asarray(model.encode(
            texts, batch_size=1 if query else self.batch_size,
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, **prompt,
        ), dtype=np.float32)
        if vectors.shape != (len(texts), self.dimensions):
            raise ValueError("local model returned an unexpected embedding shape")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError("local model returned an invalid embedding")
        return (vectors / norms[:, None]).tolist()

    async def query(self, text: str) -> list[float]:
        text = text.strip()
        if not text:
            raise ValueError("embedding query must not be empty")
        return (await self._run(self._encode, [text], query=True))[0]

    async def texts(self, texts: list[str]) -> list[list[float]]:
        values = [text.strip() for text in texts]
        if any(not text for text in values):
            raise ValueError("embedding passages must not be empty")
        result = []
        # Yield between passage batches so live queries can share this model.
        for offset in range(0, len(values), self.batch_size):
            result.extend(await self._run(self._encode, values[offset:offset + self.batch_size], query=False))
        return result

    @staticmethod
    def _search_rows(rows, vector: list[float], limit: int) -> list[dict]:
        query = np.asarray(vector, dtype=np.float32)
        remaining = iter(rows)
        selected, offset = [], 0
        while batch := list(islice(remaining, 1024)):
            matrix = np.empty((len(batch), query.size), dtype=np.float32)
            for index, row in enumerate(batch):
                if int(row["dimensions"]) != query.size or len(row["vector"]) != query.size * 4:
                    raise ValueError("stored embedding dimensions or byte length differ from the query")
                matrix[index] = np.frombuffer(row["vector"], dtype="<f4")
            norms = np.linalg.norm(matrix, axis=1)
            if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
                raise ValueError("stored embedding contains an invalid vector")
            matrix /= norms[:, None]
            scores = np.clip(matrix @ query, -1.0, 1.0)
            for index in np.argsort(-scores, kind="stable")[:limit]:
                row = batch[index]
                selected.append((float(scores[index]), offset + int(index),
                                 str(row["owner_type"]), str(row["owner_key"])))
            # Stable ties keep the original database iteration order across blocks.
            selected.sort(key=lambda row: (-row[0], row[1]))
            del selected[limit:]
            offset += len(batch)
        return [{"owner_type": kind, "owner_key": key, "score": score}
                for score, _, kind, key in selected]

    async def search(self, store, query: str, limit: int = 8) -> list[dict]:
        if limit < 1:
            return []
        # Read the actual index each time: additions/deletions are visible without
        # an invented cache fingerprint or a second index-version protocol.
        if not await store.read_async(store.vector_rows, self.model_id, limit=1):
            return []
        vector = await self.query(query)
        return await store.read_async(store.search_vectors, self.model_id, vector, int(limit))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Release only after submitted native inference has finished. Reload may
        # retain this Embedder object while constructing the replacement model.
        def release_model():
            self._model = None

        barrier = asyncio.get_running_loop().run_in_executor(self._executor, release_model)
        try:
            await asyncio.shield(barrier)
        finally:
            # Even a cancelled close waiter must leave the queued release intact.
            self._executor.shutdown(wait=False)
