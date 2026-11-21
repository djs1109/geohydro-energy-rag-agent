"""Offline regression smoke tests; run: python -m unittest test_chunking.py"""
import tempfile
import unittest
from pathlib import Path
from document_store import DocumentStore
from document_ingestion import ingest_document
from parent_retrieval import expand_parents


class TestFormalChunking(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=DocumentStore(Path(self.temp.name)/'data'/'chunks.sqlite')

    def tearDown(self):
        self.temp.cleanup()

    def test_markdown_parent_child_and_dedup(self):
        raw=(('# Study\n## Inputs\nIMERG V07 uses satellite inputs.\n\n'
             '## Fusion\nSatellite precipitation estimates undergo processing.\n\n')*30).encode()
        result=ingest_document(filename='NASA.md',file_bytes=raw,store=self.store)
        self.assertGreater(result['parent_count'],0)
        self.assertGreater(result['chunk_count'],0)
        chunks=self.store.list_chunks()
        self.assertTrue(all('parent_id' in r['payload'] for r in chunks))
        self.assertIsNotNone(self.store.get_parent(chunks[0]['payload']['parent_id']))
        result2=ingest_document(filename='NASA.md',file_bytes=raw,store=self.store)
        self.assertTrue(result2['duplicate'])
        self.assertEqual(self.store.count(),result['chunk_count'])
        hit={'point_id':chunks[0]['id'],'text':chunks[0]['text'],
             'metadata':chunks[0]['payload'],'rerank_score':0.91}
        expanded=expand_parents([hit,hit],self.store,top_k=5)
        self.assertEqual(len(expanded),1)
        self.assertGreaterEqual(len(expanded[0]['text']),len(hit['text']))

    def test_pdf(self):
        try:
            import fitz
        except ImportError:
            self.skipTest('pymupdf missing')
        pdf=fitz.open()
        for k in range(4):
            page=pdf.new_page()
            page.insert_text((75,70), f'IMERG V07 technical manual section {k+1}',fontsize=18)
            page.insert_text((75,110), ('Satellite precipitation information fusion. '*3),fontsize=10)
        raw=pdf.tobytes();pdf.close()
        result=ingest_document(filename='manual.pdf',file_bytes=raw,store=self.store)
        self.assertEqual(result['page_count'],4)
        self.assertTrue(all(r['payload']['page'] is not None for r in self.store.list_chunks()))

    def test_docx_txt(self):
        import io
        from docx import Document
        doc=Document();doc.add_heading('Hydrology',1);doc.add_paragraph('Precipitation and evaporation balance.')
        buff=io.BytesIO();doc.save(buff)
        a=ingest_document(filename='readme.docx',file_bytes=buff.getvalue(),store=self.store)
        b=ingest_document(filename='readme.txt',file_bytes=b'Precipitation.\n\nEvaporation.',store=self.store)
        self.assertGreater(a['chunk_count'],0)
        self.assertGreater(b['chunk_count'],0)
        self.assertIsNone(self.store.list_chunks()[-1]['payload']['page'])

    def test_legacy_upgrade_on_reupload(self):
        from document_store import _j
        import hashlib
        import uuid
        raw=b'# Old title\n\nNew full structure.'
        sha=hashlib.sha256(raw).hexdigest()
        did=str(uuid.uuid5(uuid.NAMESPACE_URL,f'document:{sha}'))
        with self.store._connect() as con:
            con.execute('INSERT INTO documents VALUES(?,?,?,?)',(did,'old.md',sha,_j({'source':'old.md'})))
            con.execute('INSERT INTO chunks VALUES(?,?,?,?)',('old-point',did,'legacy chopped string',_j({'source':'old.md'})))
        result=ingest_document(filename='old.md',file_bytes=raw,store=self.store)
        self.assertTrue(result['reindexed'])
        self.assertFalse(result['duplicate'])
        self.assertNotIn('old-point',{r['id'] for r in self.store.list_chunks()})
        self.assertTrue(self.store.list_documents()[0]['has_original'])

if __name__=='__main__':
    unittest.main()
