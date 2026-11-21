"""Automatic CPU embedding indexing and atomic Qdrant alias switching.

IMPORTANT: A new collection is indexed fully before the live alias switches.
Do not call switch_model concurrently from multiple workers.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Protocol
from qdrant_client import models


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str = 'BAAI/bge-m3'
    revision: str = 'main'  # Pin a commit hash in production if reproducibility matters.
    chunker_version: str = 'v1'


class Encoder(Protocol):
    def encode(self, sentences, **kwargs): ...
    def get_sentence_embedding_dimension(self): ...


def physical_collection(spec: EmbeddingSpec) -> str:
    identity = json.dumps(spec.__dict__, ensure_ascii=False, sort_keys=True)
    return 'rag_idx_' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]


class EmbeddingManager:
    def __init__(self, qdrant, store, *, alias='rag_active', batch_size=8):
        self.qdrant, self.store = qdrant, store
        self.alias, self.batch_size = alias, int(batch_size)
        if self.batch_size < 1:
            raise ValueError('batch_size must be >=1')

    def _alias_target(self):
        for a in self.qdrant.get_aliases().aliases:
            if a.alias_name == self.alias:
                return a.collection_name
        return None

    def _ensure_collection(self, name, dimension):
        if not self.qdrant.collection_exists(name):
            self.qdrant.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
            return
        cfg = self.qdrant.get_collection(name).config.params.vectors
        if isinstance(cfg, dict):
            raise ValueError(f'{name}: named vectors detected; expected a single dense vector')
        if cfg.size != dimension or cfg.distance != models.Distance.COSINE:
            raise ValueError(f'{name}: vector config mismatch; expected cosine dimension={dimension}')

    def _existing_ids(self, name):
        result = set()
        offset = None
        while True:
            points, offset = self.qdrant.scroll(
                collection_name=name, limit=256, offset=offset,
                with_payload=False, with_vectors=False,
            )
            result.update(str(p.id) for p in points)
            if offset is None:
                break
        return result

    def _sync_collection(self, name: str, encoder: Encoder, *, remove_orphans=True, progress=None):
        chunks = self.store.list_chunks()
        valid = {r['id'] for r in chunks}
        existing = self._existing_ids(name)
        missing = [row for row in chunks if row['id'] not in existing]
        dim = encoder.get_sentence_embedding_dimension()
        if progress:
            progress("embedding", 0, max(1, len(missing)), f"待编码 {len(missing)} 个新切片")
        for start in range(0, len(missing), self.batch_size):
            rows = missing[start:start+self.batch_size]
            vectors = encoder.encode(
                [r['text'] for r in rows],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            points = []
            for row, vec in zip(rows, vectors):
                values = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
                if len(values) != dim:
                    raise ValueError('Unexpected embedding dimension')
                payload = {**row['payload'], 'text': row['text']}
                points.append(models.PointStruct(id=row['id'], vector=values, payload=payload))
            self.qdrant.upsert(collection_name=name, points=points, wait=True)
            if progress:
                progress("embedding", min(start+len(rows), len(missing)), max(1,len(missing)), f"已编码并写入 {min(start+len(rows), len(missing))}/{len(missing)} 个切片")
        stale = existing - valid if remove_orphans else set()
        stale_list = list(stale)
        for start in range(0, len(stale_list), 256):
            self.qdrant.delete(
                collection_name=name,
                points_selector=models.PointIdsList(points=stale_list[start:start+256]),
                wait=True,
            )
        count = self.qdrant.count(collection_name=name, exact=True).count
        if count != len(chunks):
            raise RuntimeError(f'Index not complete: {count} vectors, {len(chunks)} chunks')
        return {'added': len(missing), 'deleted': len(stale), 'total': count}

    def switch_model(self, spec: EmbeddingSpec, encoder: Encoder):
        """Build/reuse the target index, verify it, then atomically move live alias."""
        name = physical_collection(spec)
        dimension = encoder.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f'Invalid embedding dimension: {dimension}')
        self._ensure_collection(name, dimension)
        stats = self._sync_collection(name, encoder)
        old = self._alias_target()
        if old != name:
            operations = []
            if old is not None:
                operations.append(models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.alias)
                ))
            operations.append(models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=name, alias_name=self.alias)
            ))
            self.qdrant.update_collection_aliases(change_aliases_operations=operations)
        return {'model': spec.name, 'collection': name, 'alias': self.alias,
                'previous_collection': old, **stats}

    def sync_current(self, spec: EmbeddingSpec, encoder: Encoder, progress=None):
        """After adding/deleting chunks: only encode missing IDs on unchanged model."""
        name = physical_collection(spec)
        if self._alias_target() != name:
            raise RuntimeError('This model is not active; run switch_model first')
        return self._sync_collection(name, encoder, progress=progress)
