#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""항목별 법령 조문 벡터 검색.

팀원이 build_index.py로 만드는 model/law_index.faiss(IndexFlatIP, dim=1024, 코사인 유사도)와
model/law_chunks.json({text, article, source, law_type})을 읽는다. 인덱스 파일이 아직 없으면
조용히 빈 결과를 반환한다 — --mock 검증이 이 파일 유무와 무관하게 항상 통과해야 하기 때문이다.

검색은 공고의 적용계약법(국가/지방)으로 항목표.json의 해당 필드만 골라, 그 안의 조문 참조를
개별 단위로 쪼갠 뒤 참조 하나당 검색을 따로 해서 top-1만 가져온다. 국가/지방 조문을 한 쿼리에
섞거나 여러 조문을 이어붙여 검색하면 결과가 흐려져서(실제 검토: docs/rag_search_review.md)
참조 단위로 쪼갰다. 결과가 공고 메타(적용계약법)에 따라 달라지므로 문서마다 다시 계산해야 한다.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

EMBED_MODEL_ID = "BAAI/bge-m3"   # 고정 임베딩 모델 (CLAUDE.md)

_embedder = None   # 지연 로드 — 인덱스가 없는 --mock 실행에서는 아예 안 불러온다

# "(계약예규)"·"(행안부예규)"는 뒤따르는 법령명의 출처 라벨일 뿐 새 참조의 시작이 아니므로 먼저 지운다.
_LABEL_RE = re.compile(r"\((?:계약예규|행안부예규)\)\s*")

# 항목표.json의 국가계약법/지방계약법 필드에 실제 등장하는 법령/예규 이름(라벨 제거 후 기준).
# 이 이름이 시작되는 지점마다 새 조문 참조로 쪼갠다. 긴 이름을 먼저 매칭하도록 길이 역순으로 정렬해서 쓴다.
LAW_NAME_ANCHORS = [
    "정부 입찰·계약 집행기준", "정부입찰계약집행기준", "공동계약운용요령",
    "지방자치단체 입찰 및 계약 집행기준", "지방자치단체 입찰 및 계약집행기준", "지방자치단체 입찰시 낙찰자 결정기준",
    "중소기업제품 구매촉진 및 판로지원에 관한 법률", "중소기업자간 경쟁제품 및 공사용자재 직접구매 대상 품목 지정 내역",
    "소프트웨어진흥법", "국가를 당사자로하는 계약에 관한 법률",
    "국가계약법", "지방계약 법", "지방계약법",
]
_ANCHOR_RE = re.compile("(?=(?:" + "|".join(re.escape(a) for a in
                         sorted(LAW_NAME_ANCHORS, key=len, reverse=True)) + "))")


def split_law_refs(raw: str) -> List[str]:
    """법령 참조 문자열을 조문 단위로 쪼갠다. '해당 없음'류(조문 불필요 항목)는 빈 리스트."""
    raw = (raw or "").strip()
    if not raw or raw.startswith("해당 없음"):
        return []
    raw = _LABEL_RE.sub("", raw)
    return [p.strip() for p in _ANCHOR_RE.split(raw) if p.strip()]


def select_scope(meta_val: Any) -> str:
    """공고 메타의 적용계약법 값을 항목표.json 필드명(국가계약법/지방계약법)으로 정규화한다."""
    return "국가계약법" if meta_val == "국가계약법" else "지방계약법"


def embed(text: str):
    """bge-m3로 쿼리를 임베딩하고 L2 정규화한다(인덱스가 코사인 유사도용 정규화 벡터라 맞춰야 함)."""
    global _embedder
    import numpy as np
    if _embedder is None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
        model = AutoModel.from_pretrained(EMBED_MODEL_ID, use_safetensors=True)
        model.eval()
        _embedder = (tok, model, torch)
    tok, model, torch = _embedder
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
        vec = model(**enc).last_hidden_state[0, 0].numpy()   # bge류 CLS 풀링
    return vec / (np.linalg.norm(vec) + 1e-12)


class VectorSearch:
    """FAISS 인덱스 + 청크 메타데이터. 인덱스 파일이 없으면 ready=False로 빈 결과만 반환한다."""

    def __init__(self, index_path: str, chunks_path: str):
        self.ready = os.path.exists(index_path) and os.path.exists(chunks_path)
        if not self.ready:
            return
        import faiss
        self.index = faiss.read_index(index_path)
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks = json.load(f)   # 순서 = 인덱스 행 순서

    def search(self, query_text: str, top_k: int = 1, law_type: str = None,
               candidate_k: int = 10) -> List[Dict[str, Any]]:
        """law_type이 주어지면 후보 candidate_k개 중 그 타입(또는 '기타')만 남겨 top_k를 고른다.

        FAISS 인덱스 자체엔 메타데이터 필터가 없어 후보를 넉넉히 뽑은 뒤 걸러내는 방식이다.
        걸러낸 결과가 하나도 없으면(후보 안에 해당 타입이 없음) 필터 없이 원래 순위를 그대로 쓴다.
        """
        if not self.ready:
            return []
        vec = embed(query_text).astype("float32").reshape(1, -1)
        k = candidate_k if law_type else top_k
        _, idxs = self.index.search(vec, k)
        hits = [self.chunks[i] for i in idxs[0] if 0 <= i < len(self.chunks)]
        if law_type:
            filtered = [c for c in hits if c.get("law_type") in (law_type, "기타")]
            hits = filtered or hits
        return hits[:top_k]


def format_law_block(chunks: List[Dict[str, Any]], char_budget: int) -> str:
    """청크들을 '출처 · 원문' 형태로 이어붙이고, 청크당 균등 예산으로 잘라 붙인다."""
    if not chunks:
        return ""
    per_chunk = max(1, char_budget // len(chunks))
    parts = []
    for c in chunks:
        head = f"[{c.get('source', '')} {c.get('article', '')}]".strip()
        parts.append(f"{head}\n{c.get('text', '')[:per_chunk]}")
    return "\n".join(parts)


def build_law_block(item_table: Dict[str, Dict[str, Any]], vs: VectorSearch,
                     items: List[str], scope: str, char_budget: int) -> str:
    """공고의 적용계약법(scope)에 맞는 조문참조만 골라, 참조 단위로 top-1 검색한 결과를 항목별로 묶는다.

    결과가 scope에 따라 달라지므로(국가/지방) 공고마다 다시 호출해야 한다. 인덱스가 없으면 빈 문자열.
    """
    if not vs.ready:
        return ""
    field = select_scope(scope)
    per_item_budget = max(1, char_budget // len(items))
    blocks = []
    for v in items:
        it = item_table[v]
        refs = split_law_refs(it.get(field))
        chunks = []
        for ref in refs:
            # ponytail: "제N조"가 명시된 참조도 순수 임베딩 유사도로만 찾음 — 부칙(다른 법령의 개정)
            # 조항이 조번호를 텍스트로 언급해 오검색을 유발하는 사례 다수 확인(docs/rag_search_review.md).
            # 조번호가 있으면 article 필드 정확매칭으로 우선 찾는 방식으로 업그레이드.
            chunks.extend(vs.search(f"{it['항목명']} {ref}", top_k=1, law_type=field))
        law = format_law_block(chunks, per_item_budget)
        if law:
            block = f"■ {v} {it['항목명']}\n{law}"[:per_item_budget]   # 항목 헤더 포함 예산 상한
            blocks.append(block)
    return "\n\n".join(blocks)
