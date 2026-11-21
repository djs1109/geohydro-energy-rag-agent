#!/usr/bin/env python3
"""Evaluate FastAPI POST /search with manually curated relevance labels.

Workflow:
  python evaluate_retrieval.py collect --api http://localhost:8000 --questions questions.json --out runs/bge_m3.json
  # Edit qrels.json with human-verified source/page labels.
  python evaluate_retrieval.py score --run runs/bge_m3.json --qrels qrels.json --k 1 3 5 10
  python evaluate_retrieval.py compare --runs runs/bge_m3.json runs/bge_small.json --qrels qrels.json --k 1 3 5 10
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, obj):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def post_json(url, payload, timeout=120):
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def normalize_source(x):
    # Match by file basename to avoid differences in absolute upload paths.
    return str(x or '').replace('\\', '/').split('/')[-1].casefold().strip()


def source_key(item):
    return normalize_source(item.get('source'))


def load_questions(path):
    q = read_json(path)
    if not isinstance(q, list) or not q:
        raise ValueError('questions.json must be non-empty list')
    ids = set()
    for x in q:
        if not isinstance(x, dict) or not x.get('id') or not x.get('query'):
            raise ValueError('Each question must have id and query')
        if x['id'] in ids:
            raise ValueError(f'duplicate question id {x["id"]}')
        ids.add(x['id'])
    return q


def collect(args):
    questions = load_questions(args.questions)
    results = []
    for i, item in enumerate(questions, 1):
        try:
            response = post_json(args.api.rstrip('/') + '/search', {'query': item['query'], 'top_k': args.top_k}, args.timeout)
            hits = []
            for rank, hit in enumerate(response.get('results', []), 1):
                hits.append({'rank': rank, 'source': hit.get('source'), 'page': hit.get('page'),
                             'point_id': str(hit.get('point_id')), 'score': hit.get('score'),
                             'text': str(hit.get('text') or '')[:args.max_chars]})
            results.append({'id': item['id'], 'query': item['query'], 'hits': hits})
            print(f'[{i}/{len(questions)}] {item["id"]}: {len(hits)} hits')
        except (urllib.error.URLError, ValueError) as exc:
            print(f'ERROR {item["id"]}: {exc}', file=sys.stderr)
            raise
    write_json(args.out, {'api': args.api, 'name': args.name, 'top_k': args.top_k, 'results': results})
    print('Saved:', args.out)
    print('NEXT: Open each hit text and make qrels.json with independently verified relevance (0,1,2).')


def parse_qrels(path):
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError('qrels must be JSON object indexed by question ID')
    cleaned = {}
    for qid, rows in raw.items():
        if not isinstance(rows, dict) or not rows:
            raise ValueError(f'{qid}: must provide non-empty source->grade mapping')
        cleaned[qid] = {}
        for source, grade in rows.items():
            key = normalize_source(source)
            if not key:
                raise ValueError(f'{qid}: empty filename')
            if type(grade) is not int or grade not in (0, 1, 2):
                raise ValueError(f'{qid}/{source}: grade must be 0,1,2')
            cleaned[qid][key] = grade
    return cleaned


def rank_sources(hits):
    """Unique source ranking; source-level evaluation cannot validate particular pages or facts."""
    ordered = []
    seen = set()
    for hit in hits:
        src = source_key(hit)
        if src and src not in seen:
            ordered.append(src)
            seen.add(src)
    return ordered


def measure_one(ranked, grades, k):
    if not any(g > 0 for g in grades.values()):
        raise ValueError('qrels has no relevant sources; metrics would be misleading')
    arr = [grades.get(src, 0) for src in ranked[:k]]
    relevant = sum(g > 0 for g in grades.values())
    found = sum(g > 0 for g in arr)
    reciprocal = next((1 / (i+1) for i, grade in enumerate(arr) if grade > 0), 0.0)
    dcg = sum((2**grade-1)/math.log2(i+2) for i, grade in enumerate(arr))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**grade-1)/math.log2(i+2) for i, grade in enumerate(ideal))
    return {'recall': found/relevant, 'mrr': reciprocal, 'ndcg': dcg/idcg if idcg else 0.0,
            'hits': found, 'relevant': relevant}


def score_run(run, qrels, ks):
    measures = {k: [] for k in ks}
    skipped = []
    rows = run.get('results', [])
    for row in rows:
        qid = row['id']
        if qid not in qrels:
            skipped.append(qid)
            continue
        ranked = rank_sources(row.get('hits', []))
        # Source-level recall is valid only when labels are exhaustive over all corpus sources.
        for k in ks:
            measures[k].append(measure_one(ranked, qrels[qid], k))
    if not any(measures.values()):
        raise ValueError('No questions have independently annotated qrels')
    output = {'name': run.get('name') or run.get('api'), 'evaluated_questions': len(measures[ks[0]]),
              'skipped_unlabeled': skipped, 'metrics': {}}
    for k in ks:
        parts = measures[k]
        output['metrics'][f'@{k}'] = {m: round(mean(row[m] for row in parts), 4)
                                     for m in ('recall','mrr','ndcg')}
    return output


def print_scores(score):
    print(f'\nRun: {score["name"]} | annotated questions: {score["evaluated_questions"]}')
    if score['skipped_unlabeled']:
        print('Skipped unlabeled:', ', '.join(score['skipped_unlabeled']))
    print(f'{"K":>4} {"Recall@K":>12} {"MRR@K":>12} {"nDCG@K":>12}')
    for k, v in score['metrics'].items():
        print(f'{k:>4} {v["recall"]:>12.4f} {v["mrr"]:>12.4f} {v["ndcg"]:>12.4f}')


def main():
    ap = argparse.ArgumentParser(description='Source-level offline RAG retrieval evaluation')
    subs = ap.add_subparsers(dest='action', required=True)
    co = subs.add_parser('collect')
    co.add_argument('--api', default='http://localhost:8000')
    co.add_argument('--questions', default='questions.json')
    co.add_argument('--out', default='runs/bge_m3.json')
    co.add_argument('--name', default='bge-m3')
    co.add_argument('--top-k', type=int, default=10)
    co.add_argument('--max-chars', type=int, default=1000)
    co.add_argument('--timeout', type=int, default=120)
    sc = subs.add_parser('score')
    sc.add_argument('--run', required=True)
    sc.add_argument('--qrels', required=True)
    sc.add_argument('--k', nargs='+', type=int, default=[1,3,5,10])
    cp = subs.add_parser('compare')
    cp.add_argument('--runs', nargs='+', required=True)
    cp.add_argument('--qrels', required=True)
    cp.add_argument('--k', nargs='+', type=int, default=[1,3,5,10])
    args = ap.parse_args()
    if args.action == 'collect':
        if not 1 <= args.top_k <= 20:
            ap.error('top-k must be 1 to 20 for current /search API')
        collect(args)
    else:
        if not args.k or min(args.k) < 1:
            ap.error('all k values must be >= 1')
        qrels = parse_qrels(args.qrels)
        for path in ([args.run] if args.action == 'score' else args.runs):
            run = read_json(path)
            print_scores(score_run(run, qrels, args.k))
        print('\nImportant: source-level metrics require exhaustive judgments for all sources;')
        print('filename match does not demonstrate factual correctness or page-level grounding.')

if __name__ == '__main__':
    main()
