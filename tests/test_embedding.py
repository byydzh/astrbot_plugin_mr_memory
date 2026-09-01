from __future__ import annotations

import asyncio
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from mr_memory.embedding import (
    LocalFastEmbedBackend,
    LocalSentenceTransformerBackend,
)


class _FakeFastEmbedModel:
    embedding_size = 3

    def __init__(self) -> None:
        self.passage_calls: list[tuple[list[str], int]] = []
        self.query_calls: list[tuple[list[str], int]] = []

    def passage_embed(self, texts, *, batch_size: int):
        values = list(texts)
        self.passage_calls.append((values, batch_size))
        for index, _text in enumerate(values, start=1):
            yield [float(index), 1.0, 0.0]

    def query_embed(self, texts, *, batch_size: int):
        values = list(texts)
        self.query_calls.append((values, batch_size))
        yield [0.0, 2.0, 0.0]


class LocalFastEmbedBackendTests(unittest.TestCase):
    def test_runs_documents_and_queries_locally_with_one_lazy_model(self) -> None:
        created: list[tuple[dict[str, object], _FakeFastEmbedModel]] = []

        def factory(**kwargs):
            model = _FakeFastEmbedModel()
            created.append((kwargs, model))
            return model

        test_root = Path.cwd() / ".dev" / "test-tmp"
        test_root.mkdir(parents=True, exist_ok=True)
        backend = LocalFastEmbedBackend(
            model_name="BAAI/bge-small-zh-v1.5",
            cache_dir=test_root,
            cpu_threads=1,
            batch_size=16,
            model_factory=factory,
        )

        async def exercise():
            passages = await backend.embed_texts(["方案 A", "方案 B"])
            query = await backend.embed_query("最终选择哪个方案？")
            return passages, query

        passages, query = asyncio.run(exercise())

        self.assertEqual(len(created), 1)
        kwargs, model = created[0]
        self.assertEqual(kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(kwargs["threads"], 1)
        self.assertTrue(kwargs["lazy_load"])
        self.assertEqual(model.passage_calls, [(["方案 A", "方案 B"], 16)])
        self.assertEqual(model.query_calls, [(["最终选择哪个方案？"], 1)])
        self.assertEqual(backend.model_id, "fastembed/BAAI/bge-small-zh-v1.5")
        self.assertEqual(backend.dimensions, 3)
        self.assertAlmostEqual(sum(value * value for value in query), 1.0)
        for vector in passages:
            self.assertAlmostEqual(sum(value * value for value in vector), 1.0)

    def test_rejects_empty_input_without_loading_model(self) -> None:
        created = 0

        def factory(**_kwargs):
            nonlocal created
            created += 1
            return _FakeFastEmbedModel()

        backend = LocalFastEmbedBackend(
            model_name="BAAI/bge-small-zh-v1.5",
            cache_dir="unused",
            model_factory=factory,
        )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            asyncio.run(backend.embed_query("  "))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            asyncio.run(backend.embed_texts(["valid", " "]))
        self.assertEqual(created, 0)

    def test_executor_submission_failure_releases_inference_slot(self) -> None:
        backend = LocalFastEmbedBackend(
            model_name="BAAI/bge-small-zh-v1.5",
            cache_dir="unused",
            model_factory=lambda **_kwargs: _FakeFastEmbedModel(),
        )

        async def exercise() -> None:
            loop = asyncio.get_running_loop()
            with patch.object(
                loop,
                "run_in_executor",
                side_effect=RuntimeError("executor unavailable"),
            ):
                with self.assertRaisesRegex(RuntimeError, "executor unavailable"):
                    await backend.embed_query("first")
            result = await backend.embed_query("second")
            self.assertEqual(len(result), 3)

        asyncio.run(exercise())

    def test_cancelled_waiter_queues_next_request_until_native_inference_exits(self) -> None:
        started = threading.Event()
        release = threading.Event()

        class BlockingModel(_FakeFastEmbedModel):
            def query_embed(self, texts, *, batch_size: int):
                values = list(texts)
                self.query_calls.append((values, batch_size))
                started.set()
                if not release.wait(timeout=2):
                    raise TimeoutError("test did not release native inference")
                yield [0.0, 2.0, 0.0]

        backend = LocalFastEmbedBackend(
            model_name="BAAI/bge-small-zh-v1.5",
            cache_dir="unused",
            model_factory=lambda **_kwargs: BlockingModel(),
        )

        async def exercise() -> None:
            first = asyncio.create_task(backend.embed_query("first"))
            while not started.is_set():
                await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(backend.embed_query("second"))
            await asyncio.sleep(0.01)
            self.assertFalse(second.done())
            release.set()
            result = await second
            self.assertEqual(len(result), 3)
            self.assertEqual(
                backend._model.query_calls,
                [(["first"], 1), (["second"], 1)],
            )

        asyncio.run(exercise())


class _FakeSentenceTransformerModel:
    max_seq_length = 32768

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, **kwargs):
        values = list(texts)
        self.calls.append((values, kwargs))
        if kwargs.get("prompt_name"):
            return [[0.0, 2.0, 0.0] for _text in values]
        return [
            [float(index), 1.0, 0.0]
            for index, _text in enumerate(values, start=1)
        ]


class LocalSentenceTransformerBackendTests(unittest.TestCase):
    def test_query_runs_between_bounded_passage_chunks(self) -> None:
        first_chunk_started = threading.Event()
        release_first_chunk = threading.Event()

        class ChunkedModel(_FakeSentenceTransformerModel):
            def encode(self, texts, **kwargs):
                values = list(texts)
                self.calls.append((values, kwargs))
                if values == ["p1", "p2"]:
                    first_chunk_started.set()
                    if not release_first_chunk.wait(timeout=2):
                        raise TimeoutError("test did not release first passage chunk")
                return [[1.0, 1.0, 0.0] for _text in values]

        model = ChunkedModel()
        backend = LocalSentenceTransformerBackend(
            model_name="microsoft/harrier-oss-v1-270m",
            cache_dir="unused",
            batch_size=2,
            model_factory=lambda *args, **kwargs: model,
        )

        async def exercise() -> None:
            passages = asyncio.create_task(
                backend.embed_texts(["p1", "p2", "p3", "p4"])
            )
            while not first_chunk_started.is_set():
                await asyncio.sleep(0)
            query = asyncio.create_task(backend.embed_query("query"))
            await asyncio.sleep(0)
            release_first_chunk.set()
            passage_vectors, query_vector = await asyncio.gather(passages, query)
            self.assertEqual(len(passage_vectors), 4)
            self.assertEqual(len(query_vector), 3)

        asyncio.run(exercise())
        self.assertEqual(
            [call[0] for call in model.calls],
            [["p1", "p2"], ["query"], ["p3", "p4"]],
        )

    def test_applies_query_prompt_only_to_queries_and_caps_length(self) -> None:
        created: list[tuple[tuple[object, ...], dict[str, object], object]] = []

        def factory(*args, **kwargs):
            model = _FakeSentenceTransformerModel()
            created.append((args, kwargs, model))
            return model

        test_root = Path.cwd() / ".dev" / "test-tmp-st"
        backend = LocalSentenceTransformerBackend(
            model_name="microsoft/harrier-oss-v1-270m",
            cache_dir=test_root,
            batch_size=4,
            query_prompt_name="web_search_query",
            max_seq_length=512,
            model_factory=factory,
        )

        async def exercise():
            passages = await backend.embed_texts(["旧消息 A", "旧消息 B"])
            query = await backend.embed_query("之前说了什么？")
            return passages, query

        passages, query = asyncio.run(exercise())

        self.assertEqual(len(created), 1)
        args, kwargs, model = created[0]
        self.assertEqual(args, ("microsoft/harrier-oss-v1-270m",))
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["model_kwargs"], {"dtype": "auto"})
        self.assertEqual(model.max_seq_length, 512)
        self.assertEqual(model.calls[0][0], ["旧消息 A", "旧消息 B"])
        self.assertNotIn("prompt_name", model.calls[0][1])
        self.assertEqual(model.calls[0][1]["batch_size"], 4)
        self.assertEqual(model.calls[1][0], ["之前说了什么？"])
        self.assertEqual(model.calls[1][1]["prompt_name"], "web_search_query")
        self.assertEqual(model.calls[1][1]["batch_size"], 1)
        self.assertEqual(backend.dimensions, 3)
        self.assertIn("query_prompt=web_search_query", backend.model_id)
        self.assertAlmostEqual(sum(value * value for value in query), 1.0)
        for vector in passages:
            self.assertAlmostEqual(sum(value * value for value in vector), 1.0)

    def test_rejects_empty_query_without_loading_model(self) -> None:
        created = 0

        def factory(*_args, **_kwargs):
            nonlocal created
            created += 1
            return _FakeSentenceTransformerModel()

        backend = LocalSentenceTransformerBackend(
            model_name="microsoft/harrier-oss-v1-270m",
            cache_dir="unused",
            model_factory=factory,
        )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            asyncio.run(backend.embed_query(" "))
        self.assertEqual(created, 0)


if __name__ == "__main__":
    unittest.main()
