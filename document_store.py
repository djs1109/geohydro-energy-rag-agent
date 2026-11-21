"""SQLite metadata + immutable source-file archive + Parent/Child corpus.

Compatible with the legacy documents/chunks database. The original-file archive
is at CHUNK_DB's sibling directory `originals`, mounted to /app/data in Docker.
"""
from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

VERSION = 'structure_parent_child_v3'


def _j(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class DocumentStore:
    def __init__(self, path='data/rag_chunks.sqlite'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.originals = self.path.parent / 'originals'
        self.originals.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute('CREATE TABLE IF NOT EXISTS documents (document_id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE, metadata_json TEXT NOT NULL)')
            con.execute('CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, text TEXT NOT NULL, metadata_json TEXT NOT NULL)')
            con.execute('CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)')
            con.execute('CREATE TABLE IF NOT EXISTS parents (parent_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, text TEXT NOT NULL, metadata_json TEXT NOT NULL)')
            con.execute('CREATE INDEX IF NOT EXISTS idx_parents_document ON parents(document_id)')

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(str(self.path), timeout=120)
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def archive_path(self, sha, filename):
        suffix = Path(filename).suffix.lower()
        return self.originals / f'{sha}{suffix}'

    def _archive(self, sha, filename, data):
        target = self.archive_path(sha, filename)
        if not target.exists():
            temp = target.with_name(target.name + '.' + str(uuid.uuid4()) + '.tmp')
            try:
                with temp.open('wb') as f:
                    f.write(data)
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)
        return target

    def add_document(self, filename, file_bytes, chunks, metadata=None, *, parents=None):
        if not chunks:
            raise ValueError('没有可持久化的子切片')
        sha = hashlib.sha256(file_bytes).hexdigest()
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f'document:{sha}'))
        base = dict(metadata or {})
        base.update(document_id=doc_id, file_sha256=sha, source=filename, chunker_version=VERSION)
        parents = parents or []
        parent_ids = {p['parent_id'] for p in parents}
        for ch in chunks:
            if ch.get('parent_id') and ch['parent_id'] not in parent_ids:
                raise ValueError('孤立的 child: parent_id 不存在')
        self._archive(sha, filename, file_bytes)
        with self._connect() as con:
            old = con.execute('SELECT document_id,metadata_json FROM documents WHERE sha256=?', (sha,)).fetchone()
            if old:
                previous = json.loads(old[1])
                if previous.get('chunker_version') == VERSION:
                    return {'document_id': old[0], 'duplicate': True, 'chunks': 0, 'reindexed': False}
                # The source was re-uploaded: replace older chunks transactionally.
                doc_id = old[0]
                con.execute('DELETE FROM chunks WHERE document_id=?', (doc_id,))
                con.execute('DELETE FROM parents WHERE document_id=?', (doc_id,))
                con.execute('UPDATE documents SET filename=?,metadata_json=? WHERE document_id=?', (filename, _j(base), doc_id))
                reindexed = True
            else:
                con.execute('INSERT INTO documents VALUES (?,?,?,?)', (doc_id, filename, sha, _j(base)))
                reindexed = False
            for p in parents:
                pid = p['parent_id']
                payload = {**base, **{k:v for k,v in p.items() if k != 'text'}}
                con.execute('INSERT INTO parents VALUES (?,?,?,?)', (pid, doc_id, p['text'], _j(payload)))
            for i, ch in enumerate(chunks):
                body = ch['text'].strip()
                if not body:
                    continue
                cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f'{doc_id}:{VERSION}:child:{i}:{body}'))
                payload = {**base, **{k:v for k,v in ch.items() if k!='text'}}
                payload.update(chunk_id=cid, chunk_index=i, document_id=doc_id)
                con.execute('INSERT INTO chunks VALUES (?,?,?,?)', (cid, doc_id, body, _j(payload)))
        return {'document_id': doc_id, 'duplicate': False, 'chunks': len(chunks), 'reindexed': reindexed}

    def count(self):
        with self._connect() as con:
            return con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]

    def list_chunks(self):
        with self._connect() as con:
            rows = con.execute('SELECT chunk_id,text,metadata_json FROM chunks ORDER BY chunk_id').fetchall()
        return [{'id': cid, 'text': text, 'payload': json.loads(meta)} for cid, text, meta in rows]

    def get_parent(self, parent_id):
        with self._connect() as con:
            row = con.execute('SELECT text,metadata_json FROM parents WHERE parent_id=?', (str(parent_id),)).fetchone()
        if not row:
            return None
        return {'text': row[0], 'metadata': json.loads(row[1])}

    def list_documents(self):
        with self._connect() as con:
            rows = con.execute('SELECT document_id,filename,sha256,metadata_json FROM documents ORDER BY filename').fetchall()
        return [{'document_id': did, 'filename': fn, 'sha256': sha,
                 'chunker_version': json.loads(meta).get('chunker_version','legacy'),
                 'has_original': self.archive_path(sha, fn).exists()} for did,fn,sha,meta in rows]

    def remove_document(self, document_id):
        with self._connect() as con:
            row=con.execute('SELECT sha256,filename FROM documents WHERE document_id=?',(document_id,)).fetchone()
            if not row:
                return False
            con.execute('DELETE FROM chunks WHERE document_id=?',(document_id,))
            con.execute('DELETE FROM parents WHERE document_id=?',(document_id,))
            con.execute('DELETE FROM documents WHERE document_id=?',(document_id,))
        self.archive_path(row[0],row[1]).unlink(missing_ok=True)
        return True

    def rebuild_archived(self, *, tokenizer=None, progress=None):
        """Upgrade any legacy documents with archived originals; do not guess old PDF structure."""
        from document_ingestion import ingest_document
        results=[]
        for doc in self.list_documents():
            if doc['chunker_version']==VERSION:
                continue
            if not doc['has_original']:
                results.append({'filename':doc['filename'],'status':'original_missing'})
                continue
            raw=self.archive_path(doc['sha256'],doc['filename']).read_bytes()
            res=ingest_document(filename=doc['filename'], file_bytes=raw, store=self,
                                tokenizer=tokenizer, progress=progress)
            results.append({'filename':doc['filename'],'status':'rebuilt','result':res})
        return results
