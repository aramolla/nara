"""참조 표현(expression) → 조·항 번호 규칙 파서.

LLM이 원문 그대로 복사한 참조 표현을 받아 (article, paragraphs) 목록으로 바꾼다.
- 조 번호가 없으면 현재 조, "같은 조"/"동조"면 직전 참조의 조
- 호·목은 항에 포함되므로 버린다
- "ㆍ", "및", "또는" 나열과 "부터~까지", "내지" 범위를 펼친다
- 앞에 붙은 법령 표시(「X」, 법, 영 등)는 떼고 번호 부분만 읽는다

실행: python prep/ref_parse.py  → model/law_chunks.json 전체 참조 표현으로 파서 검증
"""
import re
from typing import List, Optional, Tuple

TOKEN = re.compile(
    r"(?P<same_jo>같은\s?조|동\s?조)"
    r"|(?P<same_hang>같은\s?항|동\s?항)"
    r"|(?P<range>부터|내지|에서)"
    r"|제?\s?(?P<n>\d+)(?:의(?P<sub1>\d+))?\s?(?P<unit>조|항|호)(?:의\s?(?P<sub2>\d+))?"
    r"|(?P<mok>[가-하]목)"
)
# 토큰 사이에 올 수 있는 연결어·수식어. 이 외의 글자가 남으면 파싱 실패로 본다.
# "같은 법"·"어느 하나"는 LLM이 표현에 붙여 복사하는 경우가 있어 허용한다.
FILLER = re.compile(r"[ㆍ·,\s]|및|또는|와|과|까지|각\s?호|각\s?목|외의\s?부분|본문|단서|전단|후단|같은\s?법|어느\s?하나|의")

Ref = Tuple[str, List[int]]   # (조 번호, 항 번호 목록)

# expression 앞에 붙은 법령 표시(「X」, 법, 영, 같은 법 등)와 번호 부분의 경계
REF_START = re.compile(r"같은\s?조|같은\s?항|동\s?조|동\s?항|제\s?\d")


def split_prefix(expression: str) -> Tuple[str, str]:
    """expression → (법령 표시, 번호 부분). 괄호(예: (이하 "법"이라 한다))는 표시 쪽에서 지운다."""
    s = re.sub(r"\([^()]*\)", "", expression)
    m = REF_START.search(s)
    if not m:
        return s.strip(), ""
    return s[:m.start()].strip(), s[m.start():].strip()


def _article(n: str, sub: Optional[str]) -> str:
    return f"제{n}조" + (f"의{sub}" if sub else "")


def parse(expression: str, current_article: str,
          prev: Optional[Ref] = None) -> Tuple[List[Ref], bool]:
    """expression → [(article, paragraphs)], 파싱 성공 여부.

    current_article: 입력 항이 속한 조 (예: "제12조")
    prev: 직전 참조 — "같은 조"/"같은 항" 해석용
    """
    out: dict = {}                             # article → 항 번호 목록(순서 유지)
    cur: Optional[str] = None                  # 지금 가리키는 조
    last: Optional[Tuple[str, int, Optional[str]]] = None   # 직전 토큰 (단위, 번호, 조의N)
    pending_range = False
    ok = True

    def add(article: str, para: Optional[int] = None) -> None:
        paras = out.setdefault(article, [])
        if para is not None and para not in paras:
            paras.append(para)

    expression = split_prefix(expression)[1]
    for m in TOKEN.finditer(expression):
        if m.group("same_jo"):
            cur = prev[0] if prev else current_article
        elif m.group("same_hang"):
            cur = prev[0] if prev else current_article
            if prev and prev[1]:
                add(cur, prev[1][-1])
            last = ("항", prev[1][-1], None) if prev and prev[1] else last
        elif m.group("range"):
            pending_range = True
        elif m.group("unit"):
            n, unit, sub = int(m.group("n")), m.group("unit"), m.group("sub2")
            if unit == "조":
                if pending_range and last and last[0] == "조":
                    if last[2] or sub:         # 조의N이 섞인 범위는 조 목록 없이 펼칠 수 없음
                        ok = False
                    else:
                        for k in range(last[1] + 1, n):
                            add(_article(str(k), None))
                cur = _article(m.group("n"), sub)
                add(cur)
            elif unit == "항":
                if m.group("sub1"):            # "제4의2항" 같은 형태는 정수 항 번호로 표현 불가
                    ok = False
                    continue
                cur = cur or current_article
                if pending_range and last and last[0] == "항":
                    for k in range(last[1] + 1, n):
                        add(cur, k)
                add(cur, n)
            else:                              # 호: 항에 포함되므로 버리되, 조는 기록
                cur = cur or current_article
                add(cur)
            last = (unit, n, sub)
            pending_range = False
    residue = FILLER.sub("", TOKEN.sub("", expression))
    if residue or not out:
        ok = False
    return list(out.items()), ok


def _verify() -> None:
    """model/law_chunks.json 전체에서 번호 참조 표현을 모아 파서가 실패하는 것만 출력."""
    import collections
    import json
    import os
    import unicodedata
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "law_chunks.json")
    chunks = json.load(open(path, encoding="utf-8"))
    unit = r"(?:같은\s?조\s?|같은\s?항\s?|동\s?조\s?|동\s?항\s?)?제\s?\d+(?:의\d+)?\s?(?:조|항|호)(?:의\s?\d+)?(?:\s?[가-하]목)?"
    conn = r"(?:\s*[ㆍ·,]\s*|\s+및\s+|\s+또는\s+|\s*부터\s+|\s*에서\s+|\s*내지\s+|\s*와\s+|\s*과\s+|\s*)"
    span = re.compile(rf"{unit}(?:{conn}(?:제?\s?\d+(?:의\d+)?\s?(?:조|항|호)(?:의\s?\d+)?(?:\s?[가-하]목)?))*(?:\s*까지)?")
    head = re.compile(r"^제\d+조(?:의\d+)?\s*\(")
    seen, total, fails = set(), 0, collections.Counter()
    for c in chunks:
        for ln in unicodedata.normalize("NFC", c["text"]).split("\n"):
            ln = ln.strip()
            if (c["source"], ln) in seen:
                continue
            seen.add((c["source"], ln))
            for m in span.finditer(ln):
                if m.start() == 0 and head.match(ln):
                    continue
                total += 1
                _, ok = parse(m.group(), "제0조")
                if not ok:
                    fails[m.group()] += 1
    print(f"표현 {total}건 · 실패 {sum(fails.values())}건")
    for s, n in fails.most_common():
        print(n, "|", s)


if __name__ == "__main__":
    _verify()
