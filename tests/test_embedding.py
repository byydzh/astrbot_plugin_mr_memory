from __future__ import annotations

import asyncio
import sys
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from mr_memory.embedding import Embedder


MODEL = "microsoft/harrier-oss-v1-270m"


class EmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_settings_and_search_reads_changed_index(self):
        calls = []

        class Encoder:
            max_seq_length = 32768

            def get_sentence_embedding_dimension(self):
                return 3

            def encode(self, texts, **kwargs):
                calls.append((list(texts), kwargs))
                return [[0.0, 2.0, 0.0] for _ in texts]

        encoder = Encoder()
        factory = Mock(return_value=encoder)
        torch = SimpleNamespace(set_num_threads=Mock())
        rows = [
            {"owner_type": "semantic", "owner_key": "1", "dimensions": 3,
             "vector": np.asarray([1, 0, 0], dtype="<f4").tobytes()},
            {"owner_type": "episode", "owner_key": "2", "dimensions": 3,
             "vector": np.asarray([0, 3, 0], dtype="<f4").tobytes()},
        ]
        store = SimpleNamespace(vector_rows=Mock(side_effect=lambda _model: list(rows)))
        embedder = Embedder(MODEL, "existing-model-cache", batch_size=4)
        try:
            with patch.dict(sys.modules, {"torch": torch, "sentence_transformers": SimpleNamespace(SentenceTransformer=factory)}):
                await embedder.warmup()
                self.assertEqual(await embedder.texts(["a", "b"]), [[0.0, 1.0, 0.0]] * 2)
                first = await embedder.search(store, "query", limit=1)
                self.assertEqual(first, [{"owner_type": "episode", "owner_key": "2", "score": 1.0}])
                rows.pop()
                second = await embedder.search(store, "query", limit=1)
                self.assertEqual(second[0]["owner_key"], "1")
                self.assertAlmostEqual(second[0]["score"], 0.0)
            self.assertEqual(embedder.model_id, f"sentence-transformers/{MODEL}?query_prompt=web_search_query")
            self.assertEqual(store.vector_rows.call_args.args, (embedder.model_id,))
            factory.assert_called_once_with(MODEL, device="cpu", cache_folder="existing-model-cache",
                                            trust_remote_code=False, model_kwargs={"dtype": "auto"})
            torch.set_num_threads.assert_called_once_with(1)
            self.assertEqual(encoder.max_seq_length, 512)
            self.assertEqual(calls[0][1]["prompt"], "")
            self.assertEqual(calls[0][1]["batch_size"], 4)
            self.assertEqual(calls[1][1]["prompt_name"], "web_search_query")
            self.assertEqual(calls[1][1]["batch_size"], 1)
        finally:
            await embedder.close()

    async def test_cancelled_native_inference_does_not_overlap_next_query(self):
        started = threading.Event()
        release = threading.Event()
        calls = []
        active = 0
        peak = 0
        embedder = Embedder(MODEL, "unused")

        def encode(texts, *, query):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            calls.append(texts[0])
            try:
                if texts[0] == "first":
                    started.set()
                    if not release.wait(2):
                        raise TimeoutError("synthetic inference was not released")
                return [[1.0, 0.0, 0.0]]
            finally:
                active -= 1

        try:
            with patch.object(embedder, "_encode", side_effect=encode):
                first = asyncio.create_task(embedder.query("first"))
                self.assertTrue(await asyncio.to_thread(started.wait, 1))
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                second = asyncio.create_task(embedder.query("second"))
                await asyncio.sleep(0)
                self.assertEqual(calls, ["first"])
                release.set()
                self.assertEqual(await second, [1.0, 0.0, 0.0])
                self.assertEqual(calls, ["first", "second"])
                self.assertEqual(peak, 1)
        finally:
            release.set()
            await embedder.close()

    async def test_close_releases_model_only_after_submitted_inference_finishes(self):
        started, release = threading.Event(), threading.Event()

        class Encoder:
            def encode(self, texts, **kwargs):
                started.set()
                if not release.wait(2):
                    raise TimeoutError("synthetic inference was not released")
                return [[1.0, 0.0, 0.0]]

        embedder = Embedder(MODEL, "unused")
        embedder._model = Encoder()
        embedder.dimensions = 3
        model = weakref.ref(embedder._model)
        closing = None
        running = asyncio.create_task(embedder.query("still running"))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            closing = asyncio.create_task(embedder.close())
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            self.assertIs(embedder._model, model())
            self.assertIsNotNone(model())
            release.set()
            self.assertEqual(await running, [1.0, 0.0, 0.0])
            await closing
            self.assertIsNone(embedder._model)
            self.assertIsNone(model())
        finally:
            release.set()
            await running
            if closing is not None:
                await closing
            else:
                await embedder.close()


if __name__ == "__main__":
    unittest.main()
