import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from mr_memory.shared_embedding import SharedEmbedder
from mr_memory.store import Store


class SharedEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def temporary(self):
        root = Path(os.environ['ASTRBOT_ROOT'])
        root.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def provider(self, version='revision-1'):
        return SimpleNamespace(
            describe=lambda: {'api_version': 1, 'model_name': 'model', 'query_prompt': 'query',
                              'max_seq_length': 512, 'dimensions': 3, 'space_id': version},
            embed_query=AsyncMock(return_value=[1., 0., 0.]),
            embed_documents=AsyncMock(return_value=[[1., 0., 0.]]),
            terminate=AsyncMock(),
        )

    async def test_provider_reload_resolves_new_instance_and_mr_close_does_not_release_it(self):
        with self.temporary() as directory:
            current = self.provider()
            context = SimpleNamespace(get_provider_by_id=lambda _: current)
            client = SharedEmbedder(context, 'shared', model_name='model', query_prompt='query',
                                    max_seq_length=512, binding_path=Path(directory)/'space.json')
            await client.warmup()
            self.assertEqual(client.model_id, 'sentence-transformers/model?query_prompt=query')
            first = current
            current = self.provider()
            await client.query('live')
            current.embed_query.assert_awaited_once_with('live', client='mr', priority='interactive')
            await client.texts(['document'])
            current.embed_documents.assert_awaited_once_with(['document'], client='mr', priority='background')
            await client.close()
            first.terminate.assert_not_awaited()
            current.terminate.assert_not_awaited()

    async def test_space_change_never_reuses_old_vectors(self):
        with self.temporary() as directory:
            current = self.provider()
            context = SimpleNamespace(get_provider_by_id=lambda _: current)
            kwargs = dict(model_name='model', query_prompt='query', max_seq_length=512,
                          binding_path=Path(directory)/'space.json')
            client = SharedEmbedder(context, 'shared', **kwargs)
            await client.warmup()
            current = self.provider('revision-2')
            with self.assertRaisesRegex(RuntimeError, 'space changed'):
                await client.query('live')
            replacement = SharedEmbedder(context, 'shared', **kwargs)
            await replacement.warmup()
            self.assertEqual(replacement.model_id, 'shared-embedding/revision-2')

    async def test_initialization_order_and_background_search(self):
        with self.temporary() as directory:
            current = None
            context = SimpleNamespace(get_provider_by_id=lambda _: current)
            client = SharedEmbedder(context, 'shared', model_name='model', query_prompt='query',
                                    max_seq_length=512, binding_path=Path(directory)/'space.json')
            await client.warmup()
            self.assertFalse(client.loaded)
            with self.assertRaisesRegex(RuntimeError, 'not loaded'):
                await client.query('live')
            current = self.provider()
            import numpy as np
            store = Store(Path(directory) / 'group.db', 'test:GroupMessage:42')
            try:
                for key, vector in [('first', [0, 1, 0]), ('second', [1, 0, 0])]:
                    store.put_vector('cue', key, client.model_id, np.array(vector, dtype='<f4').tobytes(), 3)
                hits = await client.background_view().search(store, 'background search', limit=1)
                self.assertEqual([hit['owner_key'] for hit in hits], ['second'])
            finally:
                store.close()
            current.embed_query.assert_awaited_once_with('background search', client='mr', priority='background')


if __name__ == '__main__':
    unittest.main()
