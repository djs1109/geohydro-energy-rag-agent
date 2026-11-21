"""Structure-aware ingestion; preserves current FastAPI and batch-task interface."""
from __future__ import annotations
import hashlib
from pathlib import Path
from structured_chunking import (
    structured_parse, build_parent_child, parse_pdf as _parse_pdf,
    decode_text, PARENT_TOKENS, CHILD_TOKENS,
)

SUPPORTED_EXTENSIONS={'.pdf','.docx','.md','.txt'}
MAX_CHUNKS=12000


def safe_filename(filename):
    return Path((filename or '').replace('\\','/')).name


def parse_pdf(data, progress=None):
    units,_=_parse_pdf(data,progress)
    return [{'page':u.page,'text':u.text,'heading':u.heading,'kind':u.kind} for u in units]


def ingest_document(*, filename, file_bytes, store, chunk_size=None, overlap=None,
                    tokenizer=None, parent_tokens=PARENT_TOKENS,
                    child_tokens=CHILD_TOKENS, progress=None):
    """Persist original PDF + structured parents/children. Legacy size args ignored.

    `tokenizer` should be `bundle.embedding_model.tokenizer` for exact BGE token counts.
    `chunk_size`/`overlap` only retained to avoid breaking old call sites.
    """
    filename=safe_filename(filename)
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f'不支持文件类型: {filename}')
    if not file_bytes:
        raise ValueError('文件为空')
    if progress: progress('parsing',0,1,'分析文档结构和章节')
    units,page_count=structured_parse(filename,file_bytes,progress)
    if not units:
        raise ValueError('没有抽取到文本；扫描 PDF 需要 OCR')
    if progress: progress('chunking',0,1,'创建 Parent/Child 章节索引')
    sha=hashlib.sha256(file_bytes).hexdigest()
    parents,children=build_parent_child(
        units,file_hash=sha,tokenizer=tokenizer,
        parent_tokens=parent_tokens,child_tokens=child_tokens)
    if len(children)>MAX_CHUNKS:
        raise ValueError(f'子切片超过 {MAX_CHUNKS} 条限制')
    if progress: progress('storing',0,1,f'持久化 {len(parents)} 个父块、{len(children)} 个子块及原始文件')
    result=store.add_document(filename,file_bytes,children,
        metadata={'file_type':Path(filename).suffix.lower(),'ingestion_method':'structured_parent_child_v3'},
        parents=parents)
    return {
        'filename':filename,'document_id':result['document_id'],
        'duplicate':result['duplicate'],'reindexed':result.get('reindexed',False),
        'chunk_count':result['chunks'],'parent_count':len(parents),
        'page_count':page_count,'chunker_version':'structure_parent_child_v3',
        'message':'文件已保存，等待向量同步' if not result['duplicate'] else '相同版本的文件已经存在',
    }
