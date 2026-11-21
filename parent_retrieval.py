"""Rank child fragments first; then expand to parent and deduplicate parents."""
from __future__ import annotations
from copy import deepcopy


def expand_parents(ranked, store, *, top_k=4, max_chars=9000):
    selected=[]; seen=set()
    for hit in ranked:
        meta=hit.get('metadata') or {}
        parent_id=meta.get('parent_id')
        # Older chunks without parent_id still remain retrievable until reindexed.
        key=('parent',parent_id) if parent_id else ('legacy',str(hit.get('point_id')))
        if key in seen:
            continue
        item=deepcopy(hit)
        if parent_id:
            parent=store.get_parent(parent_id)
            if parent:
                full=parent['text']
                item['text']=full[:max_chars]
                item['context_truncated']=len(full)>max_chars
                item['metadata']={**meta,
                                  'heading':parent['metadata'].get('heading') or meta.get('heading'),
                                  'page':parent['metadata'].get('page_start'),
                                  'page_start':parent['metadata'].get('page_start'),
                                  'page_end':parent['metadata'].get('page_end'),
                                  'parent_id':parent_id}
                item['matched_child_text']=hit.get('text','')
            # If parent missing, continue with child text; never invent parent.
        selected.append(item)
        seen.add(key)
        if len(selected)>=top_k:
            break
    return selected
