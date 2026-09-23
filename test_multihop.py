#!/usr/bin/env python3
"""
test_multihop.py — Multi-hop RAG 검색 테스트
인덱스: koneps/open/model/ 사용 (nara 인덱스 빌드 전 임시)

실행:
  python3 test_multihop.py
  python3 test_multihop.py --query "실적제한 1배수" --depth 2
"""

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

# ── 인덱스 경로 (koneps 것 임시 사용) ──────────────────────────
_HERE       = Path(__file__).parent
KONEPS_DIR  = _HERE.parent / "koneps" / "open" / "model"
NARA_DIR    = _HERE / "model"
MODEL_DIR   = NARA_DIR if NARA_DIR.exists() and (NARA_DIR / "law_index.faiss").exists() else KONEPS_DIR

CHUNK_FILE  = MODEL_DIR / "law_chunks.json"
INDEX_FILE  = MODEL_DIR / "law_index.faiss"

# ── 참조 패턴 ──────────────────────────────────────────────────
# 연속 항: "제76조 제5항·제6항·제9항"
PARA_SEQ_PAT = re.compile(
    r'(제\d+조(?:의\d+)?)'
    r'((?:\s*제\d+항[\s·,및]*)+)'
)
# 단일: "제21조", "제21조 제3항"
REF_PAT = re.compile(
    r'(제\d+조(?:의\d+)?)'
    r'(?:\s*제(\d+)항)?'
)


def extract_refs(text: str) -> list[dict]:
    refs = []
    seen = set()

    # 연속 항 먼저
    for m in PARA_SEQ_PAT.finditer(text):
        article   = m.group(1)
        para_part = m.group(2)
        for p in re.findall(r'제(\d+)항', para_part):
            key = (article, p)
            if key not in seen:
                seen.add(key)
                refs.append({"article": article, "paragraph": p})

    # 단일
    for m in REF_PAT.finditer(text):
        article = m.group(1)
        para    = m.group(2)
        key     = (article, para)
        if key not in seen:
            seen.add(key)
            refs.append({"article": article, "paragraph": para})

    return refs


def fetch_by_ref(chunks_db: list[dict], ref: dict) -> list[dict]:
    results = []
    for c in chunks_db:
        if c["article"] != ref["article"]:
            continue
        if ref["paragraph"] and c.get("paragraph", "") != ref["paragraph"]:
            continue
        results.append(c)
    return results


def embed_query(text: str, embed_dir: str):
    """bge-m3 또는 sentence-transformers로 쿼리 임베딩"""
    import numpy as np
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from FlagEmbedding import BGEM3FlagModel
        model = BGEM3FlagModel(embed_dir, use_fp16=False, device=device)
        out   = model.encode([text], batch_size=1, max_length=512,
                             return_dense=True, return_sparse=False, return_colbert_vecs=False)
        vec   = np.array(out["dense_vecs"], dtype="float32")
    except ImportError:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embed_dir, device=device)
        vec   = model.encode([text], normalize_embeddings=True,
                             convert_to_numpy=True).astype("float32")

    vec /= (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
    return vec  # (1, 1024)


def multi_hop_search(
    query: str,
    faiss_index,
    chunks_db: list[dict],
    embed_fn,
    top_k: int = 5,
    max_depth: int = 2,
) -> list[dict]:
    collected = OrderedDict()

    def add(c):
        key = (c.get("law",""), c["article"], c.get("paragraph",""))
        if key not in collected:
            collected[key] = c

    # 1단계: 시맨틱 검색
    q_vec = embed_fn(query)
    scores, ids = faiss_index.search(q_vec, top_k)
    frontier = [chunks_db[i] for i in ids[0] if i >= 0]
    for c in frontier:
        add(c)
    print(f"\n[1단계] 시맨틱 검색 → {len(frontier)}개")

    # 2~n단계: 참조 추적
    for depth in range(max_depth):
        refs = []
        for c in frontier:
            refs.extend(extract_refs(c["text"]))

        print(f"[{depth+2}단계] 참조 발견 {len(refs)}개: "
              + ", ".join(f"{r['article']}{'제'+r['paragraph']+'항' if r['paragraph'] else ''}"
                          for r in refs[:8])
              + ("..." if len(refs) > 8 else ""))

        new_frontier = []
        for ref in refs:
            for c in fetch_by_ref(chunks_db, ref):
                key = (c.get("law",""), c["article"], c.get("paragraph",""))
                if key not in collected:
                    add(c)
                    new_frontier.append(c)

        print(f"         → 새 청크 {len(new_frontier)}개 추가")
        frontier = new_frontier
        if not frontier:
            break

    return list(collected.values())


def print_result(chunks: list[dict]):
    print(f"\n{'='*60}")
    print(f"최종 수집 청크: {len(chunks)}개")
    print('='*60)
    for i, c in enumerate(chunks, 1):
        law  = c.get("law", c.get("source",""))[:20]
        art  = c.get("article_title", c["article"])
        para = c.get("paragraph", "")
        text = c["text"][:120].replace("\n", " ")
        print(f"\n[{i}] {law} / {art} {para}")
        print(f"    {text}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="실적 제한 1배수 초과 금지")
    ap.add_argument("--embed-dir",
                    default="/home/user1/.cache/huggingface/hub/"
                            "models--BAAI--bge-m3/snapshots/"
                            "5617a9f61b028005a4858fdac845db406aefb181")
    ap.add_argument("--top-k",  type=int, default=5)
    ap.add_argument("--depth",  type=int, default=2)
    args = ap.parse_args()

    # 인덱스 로드
    if not INDEX_FILE.exists():
        sys.exit(f"[ERROR] 인덱스 없음: {INDEX_FILE}\n"
                 f"       먼저 build_index.py 실행하세요.")

    print(f"[LOAD] {INDEX_FILE}")
    import faiss
    index      = faiss.read_index(str(INDEX_FILE))
    chunks_db  = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    print(f"       {index.ntotal}벡터 / {len(chunks_db)}청크")

    # 임베딩 함수
    print(f"\n[EMB] 모델 로딩: {args.embed_dir}")
    embed_fn = lambda q: embed_query(q, args.embed_dir)

    # 검색
    print(f"\n[검색] '{args.query}'  top_k={args.top_k}  depth={args.depth}")
    results = multi_hop_search(
        args.query, index, chunks_db, embed_fn,
        top_k=args.top_k, max_depth=args.depth,
    )

    print_result(results)

    # 총 컨텍스트 토큰 추정 (한글 1자 ≈ 1.5토큰)
    total_chars = sum(len(c["text"]) for c in results)
    print(f"\n총 텍스트: {total_chars}자 / 추정 토큰: ~{int(total_chars*1.5):,}")
    print("(32,768 컨텍스트 한도 내 여부 확인용)")


if __name__ == "__main__":
    main()
