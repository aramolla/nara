"""문서별 약칭 사전 생성: 원문에 명시된 정의 「X」(이하 "Y"라 한다)만 추출한다.

- 범위가 제한된 정의("이하 이 조에서", "이 장에서" 등)는 제외
- 같은 문서에서 같은 약칭을 다르게 정의하면 제외
- 정식 이름은 법령패키지 파일명과 띄어쓰기만 다르면 파일명으로 맞춘다

실행: python prep/build_abbrev.py  → model/abbrev.json
"""
import collections
import json
import os
import re
import unicodedata

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CHUNKS = os.path.join(ROOT, "model", "law_chunks.json")
OUT = os.path.join(ROOT, "model", "abbrev.json")

DEF = re.compile(
    r"「(?P<full>[^」]+)」\s*"
    r"\(이하\s*(?P<scope>이\s*(?:조|항|호|장|절|관|편)(?:에서)?\s*)?(?:이\s*\S+에서\s*)?"
    r"[\"“](?P<abbr>[^\"”]+)[\"”]\s*(?:이?라고?|으로)\s*한다\)"
)


def main() -> None:
    chunks = json.load(open(CHUNKS, encoding="utf-8"))
    names = {c["source"].replace(".txt", "") for c in chunks}
    canon = {n.replace(" ", ""): n for n in names}

    found = collections.defaultdict(lambda: collections.defaultdict(set))   # 문서 → 약칭 → 정식 이름들
    scoped = collections.Counter()
    for c in chunks:
        doc = c["source"].replace(".txt", "")
        for m in DEF.finditer(unicodedata.normalize("NFC", c["text"])):
            if m.group("scope"):
                scoped[(doc, m.group("abbr"))] += 1
                continue
            full = m.group("full").strip()
            found[doc][m.group("abbr").strip()].add(canon.get(full.replace(" ", ""), full))

    abbrev, conflicts = {}, []
    for doc in sorted(found):
        for abbr, fulls in sorted(found[doc].items()):
            if len(fulls) > 1:
                conflicts.append((doc, abbr, sorted(fulls)))
                continue
            abbrev.setdefault(doc, {})[abbr] = next(iter(fulls))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(abbrev, f, ensure_ascii=False, indent=2)
    print(f"저장: {OUT} · 문서 {len(abbrev)}개 · 약칭 {sum(len(v) for v in abbrev.values())}개")
    for doc, d in abbrev.items():
        print(f"\n[{doc}]")
        for a, full in d.items():
            print(f"  {a} → {full}" + ("" if full in names else "   (패키지 밖)"))
    print("\n제외 — 같은 약칭 다른 정의:")
    for doc, a, fulls in conflicts:
        print(f"  [{doc}] {a} → {fulls}")
    print("제외 — 범위 제한 정의:")
    for (doc, a), n in scoped.items():
        print(f"  [{doc}] {a} ({n}회)")


if __name__ == "__main__":
    main()
