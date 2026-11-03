"""Resolve a shared embedding Provider per call, never own its model lifetime."""
import asyncio
import json
from pathlib import Path

from .embedding import Embedder


class SharedEmbedder:
    _search_rows = staticmethod(Embedder._search_rows)

    def __init__(self, context, provider_id, *, model_name, query_prompt, max_seq_length, binding_path):
        self.context, self.provider_id = context, provider_id
        self.expected = {'model_name': model_name, 'query_prompt': query_prompt,
                         'max_seq_length': max_seq_length}
        self.legacy_id = f"sentence-transformers/{model_name}?query_prompt={query_prompt or 'none'}"
        self.binding_path = Path(binding_path)
        self.binding = json.loads(self.binding_path.read_text(encoding='utf-8')) if self.binding_path.exists() else None
        self.closed = False
        self.space_id = None
        self.index_id = None

    def _provider(self):
        if self.closed:
            raise RuntimeError('MR embedding client is closed')
        provider = self.context.get_provider_by_id(self.provider_id)
        if provider is None:
            raise RuntimeError(f'Embedding Provider {self.provider_id} is not loaded')
        metadata = provider.describe()
        if metadata.get('api_version') != 1:
            raise ValueError('Unsupported shared embedding API version')
        for key, expected in self.expected.items():
            if metadata[key] != expected:
                raise ValueError(f'Shared embedding {key} differs from MR index settings')
        if self.space_id is not None and metadata['space_id'] != self.space_id:
            raise RuntimeError('Embedding vector space changed; reload MR to select the new index')
        if self.binding is None:
            # The first migration keeps MR's existing index. Its old encoder
            # settings must match above; pin the full space from this point on.
            binding = {'space_id': metadata['space_id'], 'index_id': self.legacy_id}
            self.binding_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.binding_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(binding, ensure_ascii=False), encoding='utf-8')
            temporary.replace(self.binding_path)
            self.binding = binding
        self.space_id = metadata['space_id']
        self.index_id = (self.binding['index_id'] if self.space_id == self.binding['space_id']
                         else 'shared-embedding/' + self.space_id)
        return provider, metadata

    @property
    def model_id(self):
        self._provider()
        return self.index_id

    @property
    def dimensions(self):
        return self._provider()[1]['dimensions']

    @property
    def loaded(self):
        return self.context.get_provider_by_id(self.provider_id) is not None and not self.closed

    async def warmup(self):
        # Plugins can initialize in either order. Actual work still requires the
        # selected Provider and fails explicitly if it is disabled/unavailable.
        if self.loaded:
            self._provider()

    async def query(self, text):
        provider, _ = self._provider()
        return await provider.embed_query(text, client='mr', priority='interactive')

    async def texts(self, texts):
        provider, _ = self._provider()
        return await provider.embed_documents(texts, client='mr', priority='background')

    async def search(self, store, query, limit=8, *, priority='interactive'):
        if limit < 1:
            return []
        rows = await asyncio.to_thread(store.vector_rows, self.model_id)
        if not rows:
            return []
        provider, _ = self._provider()
        vector = await provider.embed_query(query, client='mr', priority=priority)
        return await asyncio.to_thread(self._search_rows, rows, vector, int(limit))

    def background_view(self):
        return BackgroundSearch(self)

    async def close(self):
        self.closed = True


class BackgroundSearch:
    def __init__(self, embedder):
        self.embedder = embedder

    async def search(self, store, query, limit=8):
        return await self.embedder.search(store, query, limit, priority='background')
