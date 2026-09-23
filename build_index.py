#!/usr/bin/env python3
"""
build_index.py — 법령 조문 → FAISS 벡터 DB 구축

실행:
  # 실제 GPU 빌드 (서버 또는 로컬 GPU 환경)
  python3 build_index.py --embed-dir /home/user1/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181

  # PPS_EMBED_DIR 환경변수로 지정된 경우
  python3 build_index.py

  # 청킹만 확인 (임베딩·저장 없음)
  python3 build_index.py --dry-run

  # 구조 확인용 mock (임베딩 없이)
  python3 build_index.py --mock

출력:
  open/model/law_index.faiss   — FAISS IndexFlatIP (dim=1024, 코사인 유사도)
  open/model/law_chunks.json   — 청크 메타데이터 {text, article, article_no, article_title, paragraph,
                                 items, chapter, deleted, seq, seq_total, prefix_len,
                                 chunk_id, source, law_type}
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
MIN_CHUNK_CHARS = 20    # (고정 크기 분할용) 너무 짧은 조각 제외

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


# ── 항 단위 청킹 ───────────────────────────────────────────────────
# 조문(제N조)으로 먼저 나누고, 각 조문을 항(①②…) → 호(1. 2. …) 단위 청크로 만든다.
# 삭제된 조·항·호, [본조신설 …] 같은 주석도 원문 그대로 남긴다 (deleted 메타데이터로 구분).
#  - 항이 없는 조문은 조문 전체가 1청크 (paragraph="")
#  - 항(또는 항 없는 조문)이 MAX_CHUNK_CHARS 초과 시 호(1. 2. …) 단위로 묶어 분할,
#    각 조각 앞에 "조문 제목 + 항 첫 문장"을 붙여 문맥 유지 (잘라서 버리는 내용 없음)
#  - 같은 조문의 청크는 article / article_no 메타데이터로 묶인다
INCLUDE_BUCHIK = False  # 부칙(개정 이력·경과조치) 제외

PARA_CHARS   = "①-⑳"                              # ①~⑳
PARA_START   = re.compile(rf"^[ \t]*(?=[{PARA_CHARS}])", re.MULTILINE)  # 줄 맨 앞 항 기호만 (문장 중간 ①②는 열거라 제외)
ITEM_START   = re.compile(r"^[ \t]*(\d+)(?:의\d+)?\.\s", re.MULTILINE)  # 호: "1. ", "2의2. "
HEADING_LINE = re.compile(r"^제\d+(?:장|절|관)(?:의\d+)?\s.*$", re.MULTILINE)
BUCHIK_LINE  = re.compile(r"^부칙(?:\s|<|\(|$)", re.MULTILINE)
BLANK_LINES  = re.compile(r"\n(?:[ \t　]*\n)+")
ARTICLE_HEAD = re.compile(r"^(제\d+조(?:의\d+)?)\s*(?:\(([^)]*)\))?")


def _clean(s: str) -> str:
    s = s.replace("<![CDATA[", "").replace("]]>", "")
    s = BLANK_LINES.sub("\n", s)          # 빈 줄 / 공백만 있는 줄 제거
    return s.strip()


ITEM_LABEL = re.compile(r"^[ \t]*(\d+(?:의\d+)?)\.\s")                  # "8의2. " → "8의2"
MOK_START  = re.compile(r"^[ \t]*([가-하])\.\s", re.MULTILINE)          # 목: "가. "
AMEND_TAG  = re.compile(r"\s*<(?:개정|신설|전문개정|제목개정|본문개정|단서신설|타법개정)[^>\n]*>")


def iter_articles(path: Path):
    """(정제된 조문 텍스트, 장 제목) 을 원문 순서대로 반환. 조문이 없으면 아무것도 반환 안 함."""
    text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
    sep = text.find("=" * 10)
    body = text[sep:] if sep != -1 else text
    body = body.lstrip("=").lstrip("\n")
    if not INCLUDE_BUCHIK:
        m = BUCHIK_LINE.search(body)
        if m:
            body = body[:m.start()]
    positions = [m.start() for m in ARTICLE_PAT.finditer(body)]
    headings = [(m.start(), m.group().strip()) for m in HEADING_LINE.finditer(body)]
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(body)
        chapter = next((h for p, h in reversed(headings) if p < start), "")
        yield _clean(HEADING_LINE.sub("", body[start:end])), chapter
    if not positions:
        yield None, _clean(body)


def chunk_law_file(path: Path) -> list[dict]:
    law_type = detect_law_type(path.name)
    chunks = []
    for article_text, chapter in iter_articles(path):
        if article_text is None:                       # 조문 구분 없는 파일
            return _fixed_split(chapter, path.name, law_type)
        chunks.extend(_split_article(article_text, chapter, path.name, law_type))
    return chunks


def _split_article(text: str, chapter: str, src: str, law_type: str) -> list[dict]:
    m = ARTICLE_HEAD.match(text)
    article_no = m.group(1)
    article_title = (m.group(2) or "").strip()
    header = f"{article_no}({article_title})" if article_title else article_no
    rest = text[m.end():].strip()

    meta = dict(article=header, article_no=article_no, article_title=article_title,
                chapter=chapter, source=src, law_type=law_type)

    # 항 경계: 줄 맨 앞의 ①② (조문 제목 바로 뒤에 붙은 ①도 rest 맨 앞이라 포함)
    # 삭제된 조문·항도 그대로 청크로 남긴다 (deleted=True 로 표시)
    starts = [p.end() for p in PARA_START.finditer(rest)]
    units = []                                 # (항 기호, 본문)
    if not starts:
        units.append(("", rest))
    else:
        lead = rest[:starts[0]].strip()        # 제목과 ① 사이 본문 (드묾)
        if lead:
            units.append(("", lead))
        for j, s in enumerate(starts):
            e = starts[j + 1] if j + 1 < len(starts) else len(rest)
            para = rest[s:e].strip()
            units.append((para[0], para))

    out = []
    for paragraph, body in units:
        out += _split_unit(header, body, paragraph, meta)
    for seq, c in enumerate(out, 1):
        c["seq"] = seq                         # 조문 안 순서 (1부터)
        c["seq_total"] = len(out)
    return out


def _split_lead(body: str, pat: re.Pattern):
    """본문을 [머리말, 하위단위1, 하위단위2, ...] 로 순서대로 자름 (내용 누락 없음)."""
    pos = [m.start() for m in pat.finditer(body)]
    if not pos:
        return body.strip(), []
    lead = body[:pos[0]].strip()
    subs = [body[p:(pos[k + 1] if k + 1 < len(pos) else len(body))].strip()
            for k, p in enumerate(pos)]
    return lead, subs


def _pack(pieces: list[tuple[str, str]], first_prefix: str, cont_prefix: str) -> list[tuple[str, list]]:
    """(라벨, 텍스트) 조각들을 순서대로 MAX_CHUNK_CHARS 이내로 묶음 (목 분할에만 사용)."""
    groups, buf, prefix = [], [], first_prefix
    for p in pieces:
        size = len(prefix) + sum(len(t) + 1 for _, t in buf) + len(p[1]) + 1
        if buf and size > MAX_CHUNK_CHARS:
            groups.append((prefix, buf))
            buf, prefix = [], cont_prefix
        buf.append(p)
    if buf:
        groups.append((prefix, buf))
    return groups


def _range(labels: list[str]) -> str:
    labels = [l for l in labels if l]
    if not labels:
        return ""
    return labels[0] if len(labels) == 1 else f"{labels[0]}~{labels[-1]}"


DELETED_ONLY = re.compile(
    rf"^(?:[{PARA_CHARS}]|\d+(?:의\d+)?\.|[가-하]\.)?\s*삭제\s*<[^>\n]*>(?:\s*\[[^\]\n]*\])*\s*$")


def _split_unit(header: str, body: str, paragraph: str, meta: dict) -> list[dict]:
    """조문 제목 + 항(또는 항 없는 조문) 본문을 청크로.
      - 항이 MAX_CHUNK_CHARS 이내 → 항 1개 = 청크 1개
      - 항이 길면 → 호 1개 = 청크 1개 (첫 호 청크에 항 첫 문장 원문 포함)
      - 호 하나도 길면 → 그 호만 목(가. 나.) 단위로 순서대로 묶어 분할
    각 청크 text = 문맥 prefix + 원문 조각. prefix_len 이후가 원문."""
    def make(prefix, text_parts, items):
        content = "\n".join(text_parts)
        return _make(prefix + "\n" + content, paragraph=paragraph, items=items,
                     deleted=bool(DELETED_ONLY.match(content)),
                     prefix_len=len(prefix) + 1, **meta)

    if len(header) + 1 + len(body) <= MAX_CHUNK_CHARS:
        return [make(header, [body], "")]

    lead, items = _split_lead(body, ITEM_START)
    if not items:                               # 호가 없으면 더 쪼갤 기준 없음
        return [make(header, [body], "")]

    # 두 번째 호부터 앞에 붙일 문맥: 제목 + 항 첫 문장(개정 이력 태그 뺀 짧은 버전)
    short_lead = AMEND_TAG.sub("", lead).strip()
    cont_prefix = f"{header}\n{short_lead} (계속)" if short_lead else header

    out = []
    for k, it in enumerate(items):
        label = ITEM_LABEL.match(it).group(1)
        prefix = header if k == 0 else cont_prefix
        parts = ([lead] if k == 0 and lead else []) + [it]
        if len(prefix) + 1 + len("\n".join(parts)) <= MAX_CHUNK_CHARS:
            out.append(make(prefix, parts, label))
            continue
        # 호 하나가 너무 김 → 목 단위로 분할
        it_lead, moks = _split_lead(it, MOK_START)
        if not moks:
            out.append(make(prefix, parts, label))
            continue
        mok_prefix = f"{cont_prefix}\n{AMEND_TAG.sub('', it_lead).strip()} (계속)"
        pieces = ([("", lead)] if k == 0 and lead else []) + [(label, it_lead)] + [
            (f"{label}{MOK_START.match(x).group(1)}", x) for x in moks]
        for pfx, grp in _pack(pieces, prefix, mok_prefix):
            out.append(make(pfx, [t for _, t in grp], _range([l for l, _ in grp])))
    return out


def _fixed_split(text: str, src: str, law_type: str, size: int = 400) -> list[dict]:
    """조문 구분 없는 파일용 고정 크기 분할"""
    step = size - 50  # 50자 overlap
    return [
        _make(text[i:i + size].strip(), article="기타", source=src, law_type=law_type)
        for i in range(0, len(text), step)
        if len(text[i:i + size].strip()) >= MIN_CHUNK_CHARS
    ]


def _make(text: str, article: str, source: str, law_type: str, **extra) -> dict:
    d = {"text": text, "article": article, "source": source, "law_type": law_type}
    d.update(extra)
    return d


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
    empty = [i for i, c in enumerate(all_chunks) if not c["text"].strip()]
    if empty:
        sys.exit(f"[ERROR] 빈 청크 {len(empty)}개 (idx 예: {empty[:5]})")
    for i, c in enumerate(all_chunks):
        c["chunk_id"] = i                      # FAISS 벡터 번호와 동일

    dist = Counter(c["law_type"] for c in all_chunks)
    lens = [len(c["text"]) for c in all_chunks]
    print(f"\n총 {len(all_chunks)}청크  |  {dict(dist)}")
    print(f"  길이 평균 {sum(lens)/len(lens):.0f}자 / 최대 {max(lens)}자 / "
          f"{MAX_CHUNK_CHARS}자 초과 {sum(l > MAX_CHUNK_CHARS for l in lens)}개")
    print(f"  삭제 조항 청크(deleted=True) {sum(1 for c in all_chunks if c.get('deleted'))}개")
    return all_chunks


# ── GPU 임베딩 ─────────────────────────────────────────────────────
def embed_chunks(chunks: list[dict], embed_dir: str, mock: bool = False,
                 batch_size: int = 64, fp16: bool = True, max_length: int = 1024) -> "np.ndarray":
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
            max_length=max_length,
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
        model.max_seq_length = max_length
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
    ap.add_argument("--max-length", type=int, default=1024,
                    help="임베딩 최대 토큰 길이 (bge-m3 최대 8192, 기본: 1024 — 긴 청크 잘림 방지)")
    ap.add_argument("--dry-run", action="store_true",
                    help="청킹만 하고 통계 출력 (임베딩·저장 안 함)")
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
    if args.dry_run:
        print("\n[DRY-RUN] 청킹만 확인하고 종료")
        return

    # 2. 임베딩
    print("\n[2/3] bge-m3 임베딩")
    embs = embed_chunks(
        chunks,
        embed_dir=args.embed_dir,
        mock=args.mock,
        batch_size=args.batch_size,
        fp16=not args.no_fp16,
        max_length=args.max_length,
    )

    # 3. FAISS 저장
    print("\n[3/3] FAISS 인덱스 저장")
    save_faiss(embs, chunks)

    print("\n완료. open/model/ 폴더를 submit.zip에 포함하세요.")
    print(f"  포함 파일: {INDEX_FILE.name}, {CHUNK_FILE.name}")


if __name__ == "__main__":
    main()
