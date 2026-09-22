#!/usr/bin/env python3
"""
build_index.py — 법령 조문 → FAISS 벡터 DB 구축

실행:
  # 실제 GPU 빌드 (서버 또는 로컬 GPU 환경)
  python3 build_index.py --embed-dir /path/to/BAAI/bge-m3

  # PPS_EMBED_DIR 환경변수로 지정된 경우
  python3 build_index.py

  # 구조 확인용 mock (임베딩 없이)
  python3 build_index.py --mock

출력:
  open/model/law_index.faiss   — FAISS IndexFlatIP (dim=1024, 코사인 유사도)
  open/model/law_chunks.json   — 청크 메타데이터 {text, article, source, law_type}
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────────
# 실행 위치에 관계없이 스크립트 위치 기준으로 경로 결정
_HERE = Path(__file__).parent
# 서버: /workspace/nara/build_index.py  → data/, model/ 이 같은 레벨
# 로컬: koneps/build_index.py           → open/data/, open/model/ 이 하위
_open_sub = _HERE / "open"
BASE_DIR   = _open_sub if _open_sub.exists() else _HERE
LAW_DIR    = BASE_DIR / "data" / "법령패키지" / "법령"
OUT_DIR    = BASE_DIR / "model"
CHUNK_FILE = OUT_DIR / "law_chunks.json"
INDEX_FILE = OUT_DIR / "law_index.faiss"

# ── 청킹 설정 ──────────────────────────────────────────────────────
MAX_CHUNK_CHARS = 512   # 조문 청크 최대 길이 (초과 시 항 단위 분할)
MIN_CHUNK_CHARS = 20    # 너무 짧은 조문 제외

ARTICLE_PAT = re.compile(r'^(제\d+조(?:의\d+)?(?:\([^)]+\))?)', re.MULTILINE)
PARA_PAT    = re.compile(r'[①②③④⑤⑥⑦⑧⑨⑩]')


# ── 법종 판별 ──────────────────────────────────────────────────────
def detect_law_type(filename: str) -> str:
    """파일명에서 국가/지방 계약법 여부 판별 (NFC 정규화 필수 — macOS NFD 대응)"""
    name = unicodedata.normalize("NFC", filename)
    if "지방자치단체" in name or "지방계약" in name:
        return "지방계약법"
    if "국가를 당사자" in name or "국가계약" in name:
        return "국가계약법"
    return "기타"


# ── 조문 단위 청킹 ─────────────────────────────────────────────────
def chunk_law_file(path: Path) -> list[dict]:
    """
    법령 텍스트 파일을 조문 단위로 분리.
    조문이 MAX_CHUNK_CHARS 초과 시 항(①②...) 단위로 재분할.
    """
    raw = path.read_text(encoding="utf-8")
    text = unicodedata.normalize("NFC", raw)
    law_type = detect_law_type(path.name)

    # 헤더(법령명, 공포일 등) 제거
    sep = text.find("=" * 10)
    body = text[sep:] if sep != -1 else text

    # 조문 위치 탐지
    positions = [(m.start(), m.group()) for m in ARTICLE_PAT.finditer(body)]

    if not positions:
        # 조문 구분이 없는 파일(고시 등) — 고정 크기로 분할
        return _fixed_split(body, path.name, law_type)

    chunks = []
    for i, (start, article_id) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(body)
        content = body[start:end].strip()

        if len(content) < MIN_CHUNK_CHARS:
            continue

        if len(content) <= MAX_CHUNK_CHARS:
            chunks.append(_make(content, article_id, path.name, law_type))
        else:
            # 항(①②③) 기준으로 재분할
            chunks.extend(_para_split(content, article_id, path.name, law_type))

    return chunks


def _para_split(text: str, article_id: str, src: str, law_type: str) -> list[dict]:
    """조문 내 항(①②...) 단위 분할"""
    parts = PARA_PAT.split(text)
    markers = PARA_PAT.findall(text)

    buf = parts[0]  # 조문 제목 (항 이전 텍스트)
    result = []

    for marker, part in zip(markers, parts[1:]):
        segment = marker + part
        if len(buf) + len(segment) > MAX_CHUNK_CHARS and len(buf) >= MIN_CHUNK_CHARS:
            result.append(_make(buf.strip(), article_id, src, law_type))
            buf = segment
        else:
            buf += segment

    if buf.strip() and len(buf.strip()) >= MIN_CHUNK_CHARS:
        result.append(_make(buf.strip(), article_id, src, law_type))

    return result or [_make(text[:MAX_CHUNK_CHARS], article_id, src, law_type)]


def _fixed_split(text: str, src: str, law_type: str, size: int = 400) -> list[dict]:
    """조문 구분 없는 파일용 고정 크기 분할"""
    step = size - 50  # 50자 overlap
    return [
        _make(text[i:i + size].strip(), "기타", src, law_type)
        for i in range(0, len(text), step)
        if len(text[i:i + size].strip()) >= MIN_CHUNK_CHARS
    ]


def _make(text: str, article_id: str, src: str, law_type: str) -> dict:
    return {
        "text": text,
        "article": article_id,
        "source": src,
        "law_type": law_type,
    }


def load_all_chunks() -> list[dict]:
    """법령 디렉토리 전체 파일 청킹"""
    if not LAW_DIR.exists():
        sys.exit(f"[ERROR] 법령 디렉토리 없음: {LAW_DIR}")

    all_chunks = []
    files = sorted(f for f in LAW_DIR.iterdir() if f.suffix == ".txt")

    for f in files:
        chunks = chunk_law_file(f)
        print(f"  {f.name[:52]:<52}  {len(chunks):>4}청크")
        all_chunks.extend(chunks)

    from collections import Counter
    dist = Counter(c["law_type"] for c in all_chunks)
    print(f"\n총 {len(all_chunks)}청크  |  {dict(dist)}")
    return all_chunks


# ── GPU 임베딩 ─────────────────────────────────────────────────────
def embed_chunks(chunks: list[dict], embed_dir: str, mock: bool = False,
                 batch_size: int = 64, fp16: bool = True) -> "np.ndarray":
    """
    bge-m3로 청크 텍스트 임베딩.
    GPU 사용 가능 시 자동으로 CUDA 사용.
    """
    import numpy as np

    texts = [c["text"] for c in chunks]
    dim = 1024  # bge-m3 dense 차원

    if mock:
        print("[MOCK] 랜덤 임베딩 (실제 모델 없음)")
        embs = np.random.randn(len(texts), dim).astype("float32")
        # L2 정규화
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return (embs / norms).astype("float32")

    # GPU 가용 여부 확인
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EMB] device: {device}")
    if device == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")
        print(f"      VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    try:
        # FlagEmbedding 사용 (bge-m3 공식 라이브러리, GPU fp16 최적화)
        from FlagEmbedding import BGEM3FlagModel
        print(f"[EMB] FlagEmbedding으로 bge-m3 로딩: {embed_dir}")
        model = BGEM3FlagModel(
            embed_dir,
            use_fp16=(fp16 and device == "cuda"),
            device=device,
        )
        t0 = time.time()
        out = model.encode(
            texts,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        embs = np.array(out["dense_vecs"], dtype="float32")
        print(f"[EMB] 완료: {len(texts)}건 / {time.time()-t0:.1f}s")

    except ImportError:
        # fallback: sentence-transformers
        print("[EMB] FlagEmbedding 없음 → sentence-transformers 사용")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(embed_dir, device=device)
        if fp16 and device == "cuda":
            model.half()
        t0 = time.time()
        embs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        print(f"[EMB] 완료: {len(texts)}건 / {time.time()-t0:.1f}s")

    # L2 정규화 (내적 = 코사인 유사도)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-9, None)
    return embs.astype("float32")


# ── FAISS 인덱스 저장 ──────────────────────────────────────────────
def save_faiss(embs: "np.ndarray", chunks: list[dict]):
    """
    IndexFlatIP (내적 기반 코사인 유사도 검색)으로 저장.
    4000~5000벡터 규모에서 IVF 없이 Flat이 충분히 빠름 (<1ms/쿼리).
    """
    import faiss
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dim = embs.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, str(INDEX_FILE))
    CHUNK_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8"
    )

    idx_mb = INDEX_FILE.stat().st_size / 1024 / 1024
    print(f"\n[SAVE] {INDEX_FILE.name}  {index.ntotal}벡터 / dim={dim} / {idx_mb:.1f}MB")
    print(f"[SAVE] {CHUNK_FILE.name}  {len(chunks)}청크")


# ── 검색 테스트 ────────────────────────────────────────────────────
def test_search(query: str = "제한경쟁입찰 실적제한 1배수 사업예산", k: int = 3):
    """저장된 인덱스에서 쿼리 검색 (mock 벡터는 의미 없음, 구조 확인용)"""
    import faiss, numpy as np

    if not INDEX_FILE.exists():
        print("[TEST] 인덱스 없음 — 건너뜀")
        return

    index = faiss.read_index(str(INDEX_FILE))
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))

    # mock: 랜덤 쿼리 벡터
    dim = index.d
    q = np.random.randn(1, dim).astype("float32")
    q /= np.linalg.norm(q)

    scores, ids = index.search(q, k)
    print(f"\n[TEST] 쿼리: '{query}'")
    print(f"       (mock 인덱스라 결과는 의미 없음 — 구조 확인용)")
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), 1):
        c = chunks[idx]
        print(f"  {rank}. [{c['law_type']}] {c['source'][:30]} / {c['article']}")
        print(f"     {c['text'][:80]}...")


# ── main ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="법령 → FAISS 벡터 DB 구축")
    ap.add_argument(
        "--embed-dir",
        default=os.environ.get("PPS_EMBED_DIR", "BAAI/bge-m3"),
        help="bge-m3 모델 경로 (기본: PPS_EMBED_DIR 환경변수 또는 'BAAI/bge-m3')",
    )
    ap.add_argument("--batch-size", type=int, default=64,
                    help="임베딩 배치 크기 (GPU VRAM에 따라 조정, 기본: 64)")
    ap.add_argument("--no-fp16", action="store_true",
                    help="fp16 비활성화 (CPU 또는 fp16 미지원 GPU용)")
    ap.add_argument("--mock", action="store_true",
                    help="랜덤 임베딩으로 구조만 확인 (모델 없이 테스트)")
    ap.add_argument("--test", action="store_true",
                    help="저장된 인덱스 검색 테스트")
    args = ap.parse_args()

    if args.test:
        test_search()
        return

    print("=" * 60)
    print("  법령 벡터 DB 구축")
    print("=" * 60)

    # 1. 청킹
    print("\n[1/3] 법령 조문 청킹")
    chunks = load_all_chunks()

    # 2. 임베딩
    print("\n[2/3] bge-m3 임베딩")
    embs = embed_chunks(
        chunks,
        embed_dir=args.embed_dir,
        mock=args.mock,
        batch_size=args.batch_size,
        fp16=not args.no_fp16,
    )

    # 3. FAISS 저장
    print("\n[3/3] FAISS 인덱스 저장")
    save_faiss(embs, chunks)

    print("\n완료. open/model/ 폴더를 submit.zip에 포함하세요.")
    print(f"  포함 파일: {INDEX_FILE.name}, {CHUNK_FILE.name}")


if __name__ == "__main__":
    main()
