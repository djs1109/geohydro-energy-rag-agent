"""stdlib-only smoke test for SQLite store and Markdown ingestion.
Run: python test_local_store.py
"""
import tempfile
from pathlib import Path
from document_store import DocumentStore
from document_ingestion import ingest_document


def main():
    with tempfile.TemporaryDirectory() as d:
        store = DocumentStore(str(Path(d) / 'chunks.sqlite'))
        content = ('# 水文气象\n\n降雨入渗可能影响边坡稳定性。\n\n'
                   '## 新能源\n\n光伏系统发电量受到太阳辐照度的影响。').encode('utf-8')
        first = ingest_document(filename='smoke.md', file_bytes=content, store=store)
        assert first['chunk_count'] > 0
        assert store.count() == first['chunk_count']
        second = ingest_document(filename='smoke.md', file_bytes=content, store=store)
        assert second['duplicate'] is True
        assert store.count() == first['chunk_count']
        assert store.remove_document(first['document_id'])
        assert store.count() == 0
        print('PASS: SQLite persistence, MD ingestion, deduplication, deletion')


if __name__ == '__main__':
    main()
