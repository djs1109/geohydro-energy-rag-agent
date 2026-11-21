"""Structure-aware Parent/Child chunking for scientific PDF/DOCX/MD/TXT.

PDF: PyMuPDF blocks/spans (page/order/font-size heading heuristic);
DOCX: heading styles and tables; MD: ATX headings and fenced code;
TXT: paragraphs. No LLM invents structure or content. Physical PDF page
numbers are preserved. Non-PDF formats do not pretend to have physical pages.
"""
from __future__ import annotations
from dataclasses import dataclass
import io
import re
import uuid
from collections import Counter

PARENT_TOKENS = 1200
CHILD_TOKENS = 320
MAX_UNITS = 15000


def normalize(text):
    text = str(text or '').replace('\x00','').replace('\r\n','\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


@dataclass
class Unit:
    text: str
    page: int | None
    heading: str = ''
    kind: str = 'paragraph'


class TokenCounter:
    def __init__(self, tokenizer=None):
        self.tokenizer=tokenizer

    def count(self, text):
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Fallback is approximate; production should pass BGE tokenizer.
        return len(re.findall(r'[\u4e00-\u9fff]|\w+|[^\w\s]', text))

    def split_long(self, text, limit):
        if self.count(text)<=limit:
            return [text]
        if self.tokenizer is not None:
            ids=self.tokenizer.encode(text, add_special_tokens=False)
            pieces=[normalize(self.tokenizer.decode(ids[i:i+limit], skip_special_tokens=True))
                    for i in range(0,len(ids),limit)]
            return [p for p in pieces if p]
        # Fallback: progressive character slicing by token estimate.
        out=[]
        while text:
            lo,hi=1,len(text)
            while lo<hi:
                mid=(lo+hi+1)//2
                if self.count(text[:mid])<=limit:
                    lo=mid
                else:
                    hi=mid-1
            out.append(text[:lo]); text=text[lo:]
        return [normalize(p) for p in out if p.strip()]


def _unit(text, page=None, heading='', kind='paragraph'):
    txt=normalize(text)
    return Unit(txt, page, heading, kind) if txt else None


def parse_pdf(data, progress=None):
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError('结构化 PDF 解析需要安装 pymupdf') from exc
    document=fitz.open(stream=data, filetype='pdf')
    try:
        if document.needs_pass and not document.authenticate(''):
            raise ValueError('PDF 需要密码；当前不支持该文档')
        if len(document)>500:
            raise ValueError('PDF 超过 500 页；请分卷或提升限制')
        units=[]; current_heading=''
        for index, page in enumerate(document):
            info=page.get_text('dict', sort=True)
            sizes=[]
            for block in info.get('blocks',[]):
                for line in block.get('lines',[]):
                    for span in line.get('spans',[]):
                        if span.get('text','').strip():
                            sizes.append(round(float(span.get('size',0)),1))
            body_size=Counter(sizes).most_common(1)[0][0] if sizes else 10.0
            for block in info.get('blocks',[]):
                if block.get('type',0)!=0:  # image block
                    continue
                lines=[]; max_font=0.0
                for line in block.get('lines',[]):
                    spans=line.get('spans',[])
                    lines.append(''.join(s.get('text','') for s in spans))
                    max_font=max(max_font,*(float(s.get('size',0)) for s in spans),0.0)
                content=normalize('\n'.join(lines))
                if not content:
                    continue
                # Common uppercase/numeric headings or visually large font. Heuristic only.
                is_heading=(len(content)<155 and '\n' not in content and
                            ((max_font>=body_size*1.22 and max_font>=body_size+1.2)
                             or bool(re.match(r'^(?:\d+(?:\.\d+){0,4}\s+)[A-Z][A-Za-z -]{3,}$',content))))
                if is_heading:
                    current_heading=content
                    kind='heading'
                else:
                    kind='paragraph'
                item=_unit(content,index+1,current_heading,kind)
                if item: units.append(item)
            if progress and (index==0 or (index+1)%5==0 or index+1==len(document)):
                progress('parsing',index+1,len(document),f'结构解析 PDF {index+1}/{len(document)} 页')
            if len(units)>MAX_UNITS:
                raise ValueError('文档结构块数量过多')
        return units, len(document)
    finally:
        document.close()


def parse_docx(data):
    from docx import Document
    doc=Document(io.BytesIO(data))
    heading=''; units=[]
    for block in doc.iter_inner_content():
        if hasattr(block,'rows'):
            rows=[[normalize(c.text) for c in row.cells] for row in block.rows]
            # A table is an indivisible logical block unless it exceeds token budget.
            text='\n'.join(' | '.join(row) for row in rows)
            item=_unit(text,None,heading,'table')
        else:
            text=normalize(block.text)
            style=getattr(getattr(block,'style',None),'name','') or ''
            kind='heading' if style.lower().startswith('heading') or style.startswith('标题') else 'paragraph'
            if kind=='heading' and text: heading=text
            item=_unit(text,None,heading,kind)
        if item: units.append(item)
    return units, None


def parse_markdown(data):
    text=decode_text(data); units=[]; heading_stack=[]; paragraph=[]; fence=None
    def flush():
        if paragraph:
            block='\n'.join(paragraph).strip()
            if block:
                units.append(Unit(block,None,' / '.join(x[1] for x in heading_stack),'paragraph'))
            paragraph.clear()
    for line in text.splitlines():
        f=re.match(r'^\s*(```|~~~)',line)
        if f:
            if fence is None: fence=f.group(1)
            elif fence==f.group(1): fence=None
        m=re.match(r'^\s*(#{1,6})\s+(.+?)\s*#*\s*$',line) if fence is None else None
        if m:
            flush(); level=len(m.group(1)); title=m.group(2).strip()
            while heading_stack and heading_stack[-1][0]>=level:
                heading_stack.pop()
            heading_stack.append((level,title))
            units.append(Unit(title,None,' / '.join(x[1] for x in heading_stack),'heading'))
        elif not line.strip() and fence is None:
            flush()
        else:
            paragraph.append(line)
    flush()
    return units,None


def decode_text(data):
    for enc in ('utf-8-sig','gb18030'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    raise ValueError('无法解码 TXT/Markdown')


def parse_txt(data):
    content=decode_text(data)
    return [Unit(normalize(part),None,'','paragraph') for part in re.split(r'\n\s*\n',content) if normalize(part)],None


def _pages(items):
    ps=[u.page for u in items if u.page is not None]
    return (min(ps),max(ps)) if ps else (None,None)


def _split_units(units, counter, limit):
    # Enforce token bound without truncating or inventing missing material.
    for item in units:
        for text in counter.split_long(item.text,limit):
            yield Unit(text,item.page,item.heading,item.kind)


def _group(items, count, limit):
    groups=[]; current=[]; size=0
    for item in items:
        n=count.count(item.text)
        heading_changed=bool(current and item.heading and item.heading!=current[0].heading)
        if current and (size+n>limit or heading_changed):
            groups.append(current);current=[];size=0
        current.append(item); size+=n
    if current: groups.append(current)
    return groups


def build_parent_child(units, *, file_hash, tokenizer=None, parent_tokens=PARENT_TOKENS, child_tokens=CHILD_TOKENS):
    if not 64<=child_tokens<parent_tokens<=4096:
        raise ValueError('要求 64<=child_tokens<parent_tokens<=4096')
    count=TokenCounter(tokenizer)
    expanded=list(_split_units(units,count,child_tokens))
    parent_groups=_group(expanded,count,parent_tokens)
    parents=[];children=[]
    for pi,grp in enumerate(parent_groups):
        parent_text='\n\n'.join(x.text for x in grp)
        parent_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f'{file_hash}:parent_child_v3:{pi}'))
        start,end=_pages(grp)
        head=next((u.heading for u in grp if u.heading),'')
        parents.append({'parent_id':parent_id,'text':parent_text,'heading':head,
                        'page':start,'page_start':start,'page_end':end,'parent_index':pi,
                        'kind':'parent'})
        for ci,child_group in enumerate(_group(grp,count,child_tokens)):
            child_text='\n\n'.join(x.text for x in child_group)
            page_start,page_end=_pages(child_group)
            # Heading included as metadata, not repeated in each parent body.
            indexed_text=(head+'\n'+child_text) if head and head not in child_text[:120] else child_text
            children.append({'text':indexed_text,'parent_id':parent_id,'child_index':ci,
                             'heading':head,'page':page_start,'page_start':page_start,
                             'page_end':page_end,'kind':'child'})
    return parents,children


def structured_parse(filename, data, progress=None):
    suffix=filename.lower().rsplit('.',1)[-1]
    if suffix=='pdf':return parse_pdf(data,progress)
    if suffix=='docx':return parse_docx(data)
    if suffix=='md':return parse_markdown(data)
    if suffix=='txt':return parse_txt(data)
    raise ValueError(f'不支持的文件类型: {suffix}')
