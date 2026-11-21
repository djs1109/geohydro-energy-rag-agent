"""Usage:
 python manage_index.py inspect
 python manage_index.py seed-legacy --collection rag_knowledge
 python manage_index.py switch --model BAAI/bge-m3
 python manage_index.py sync --model BAAI/bge-m3
"""
import argparse
import os
from qdrant_client import QdrantClient
from document_store import DocumentStore
from embedding_manager import EmbeddingManager, EmbeddingSpec


def seed_legacy(qdrant, store, collection_name):
    """Copy original TEXT (not vectors) from an old collection to durable SQLite chunks."""
    offset = None
    docs = {}
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection_name, limit=128, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            payload = dict(p.payload or {})
            txt = payload.get('text', '')
            if not txt or not txt.strip():
                continue
            filename = str(payload.get('source') or 'legacy_qdrant')
            # Separate old records by point ID so no unrelated chunks are merged.
            key = str(p.id)
            docs[key] = (filename, txt, payload)
        if offset is None:
            break
    added = 0
    for point_id, (filename, txt, meta) in docs.items():
        # Synthetic identity for the existing point's TEXT; no new content is invented.
        identity = f'{collection_name}:{point_id}:{txt}'.encode('utf-8')
        result = store.add_document(
            filename, identity,
            [{'text': txt, **{k: v for k, v in meta.items() if k != 'text'}}],
            metadata={'migration_source': collection_name, 'legacy_point_id': point_id},
        )
        added += 0 if result['duplicate'] else 1
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect', 'seed-legacy', 'switch', 'sync'])
    parser.add_argument('--model', default=os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3'))
    parser.add_argument('--revision', default=os.getenv('EMBEDDING_REVISION', 'main'))
    parser.add_argument('--collection', default='rag_knowledge')
    args = parser.parse_args()

    url = os.getenv('QDRANT_URL', 'http://localhost:6333')
    db = os.getenv('CHUNK_DB', 'data/rag_chunks.sqlite')
    qdrant = QdrantClient(url=url, timeout=120)
    store = DocumentStore(db)
    manager = EmbeddingManager(qdrant, store, batch_size=int(os.getenv('EMBEDDING_BATCH_SIZE', '4')))
    if args.action == 'inspect':
        print('Saved documents:', store.list_documents())
        print('Saved chunks:', store.count())
        print('Aliases:', qdrant.get_aliases())
        return
    if args.action == 'seed-legacy':
        raise SystemExit('正式版不再自动 seed-legacy：该操作无法恢复原始文件的结构，请重新上传官方原件。')
    from sentence_transformers import SentenceTransformer
    print('Loading:', args.model)
    encoder = SentenceTransformer(args.model, device='cpu', revision=args.revision)
    spec = EmbeddingSpec(name=args.model, revision=args.revision, chunker_version='structure_parent_child_v3')
    result = manager.switch_model(spec, encoder) if args.action == 'switch' else manager.sync_current(spec, encoder)
    print('Result:', result)


if __name__ == '__main__':
    main()
