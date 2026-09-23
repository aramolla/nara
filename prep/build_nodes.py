"""법령 청크 → 항 단위 노드(참조 조문 포함) JSON.

model/law_chunks/NN_*.json 1개 → model/node/NN_*.json 1개 (ref_exclude.json의 파일은 건너뜀)
- 같은 조·항의 청크를 seq 순서로 이어 항 원문을 복원 (각 청크 text[prefix_len:])
- 항 원문 → Gemma(참조 표현·법령 추출) → ref_parse(조·항 번호) : ref_extract_test.py의 프롬프트·조립 로직 그대로 사용
- 모든 파일의 항을 한 번에 배치 호출한 뒤 파일별로 나눠 저장

실행: python3 prep/build_nodes.py              # 전체
      python3 prep/build_nodes.py --dry-run    # LLM 없이 항 복원·파일 수만 확인
      python3 prep/build_nodes.py --limit 50   # 앞에서 50개 항만 LLM 호출 (저장 안 함)
      python3 prep/build_nodes.py --only 05:제26조:1 --max-tokens 16384 --max-model-len 32768
                                               # 지정 항만 재실행해 기존 노드 파일에서 교체
"""
import argparse
import copy
import glob
import json
import os
import time

import jsonschema

import ref_extract_test as rx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CHUNK_DIR = os.path.join(ROOT, "model", "law_chunks")
NODE_DIR = os.path.join(ROOT, "model", "node")
LOG_PATH = os.path.join(HERE, "build_nodes_log.txt")
ERR_PATH = os.path.join(HERE, "build_nodes_errors.txt")   # 예외·미처리 사유

errors: list = []                             # 예외 사유 (멈추지 않고 모아서 ERR_PATH에 저장)


def record(where: str, reason: str) -> None:
    errors.append(f"[{where}] {reason}")


with open(os.path.join(HERE, "ref_exclude.json"), encoding="utf-8") as f:
    EXCLUDE = json.load(f)["exclude"]

# 노드 스키마: ref_schema.json + 항 없는 조(paragraph=null) + 원문·메타
NODE_SCHEMA = copy.deepcopy(rx.FINAL_SCHEMA)
NODE_SCHEMA["properties"]["paragraph"] = {"type": ["integer", "null"]}
NODE_SCHEMA["properties"].update({
    "chapter": {"type": "string"},
    "deleted": {"type": "boolean"},
    "chunk_ids": {"type": "array", "items": {"type": "integer"}},
    "text": {"type": "string"},
    "error": {"type": "string"},
})
NODE_SCHEMA["required"] += ["chapter", "deleted", "chunk_ids", "text"]


def load_units(path: str) -> list:
    """청크 파일 → 항 단위 목록 [{article, chapter, paragraph(기호), deleted, chunk_ids, text}]"""
    chunks = json.load(open(path, encoding="utf-8"))
    units = []
    for c in chunks:
        if c["seq"] == 1 or units[-1]["_art"] != c["article_no"] or units[-1]["_para"] != c["paragraph"]:
            units.append({"_art": c["article_no"], "_para": c["paragraph"], "article": c["article"],
                          "chapter": c["chapter"], "parts": [], "chunk_ids": [], "deleted": True})
        u = units[-1]
        u["parts"].append(c["text"][c["prefix_len"]:])
        u["chunk_ids"].append(c["chunk_id"])
        u["deleted"] = u["deleted"] and c["deleted"]
    for u in units:
        u["text"] = "\n".join(u.pop("parts"))
    return units


def paragraph_no(symbol: str):
    return ord(symbol) - ord("①") + 1 if symbol else None


def to_node(law: str, u: dict, llm_text) -> dict:
    """LLM 출력(없으면 삭제 항) → 노드. 실패 시 refs=[] + error."""
    err = None
    n_fail = len(rx.log_fail)
    try:
        llm_out = json.loads(llm_text) if llm_text is not None else {"refs": []}
        node = rx.assemble(law, u["article"], u["text"], llm_out)
    except Exception as e:                    # JSON 파싱 실패(토큰 한도 등)·조립 실패
        err = f"{type(e).__name__}: {e}"
        node = {"article": u["_art"], "article_title": u["article"], "law": law, "refs": []}
    node["paragraph"] = paragraph_no(u["_para"])   # assemble은 원문 첫 글자로 계산 → 메타로 덮어씀
    node.update(chapter=u["chapter"], deleted=u["deleted"], chunk_ids=u["chunk_ids"], text=u["text"])
    if len(rx.log_fail) > n_fail:
        err = "; ".join(filter(None, [err, "파서 실패: " + " | ".join(rx.log_fail[n_fail:])]))
    try:
        jsonschema.validate(node, NODE_SCHEMA)
    except jsonschema.ValidationError as e:
        err = "; ".join(filter(None, [err, f"스키마 위반: {e.message}"]))
    if err:
        node["error"] = err
        record(f"{law} {u['article']} {u['_para'] or '-'}", err)
    return node


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None, help="파일번호:조:항 (항 없으면 0)")
    ap.add_argument("--max-tokens", type=int, default=rx.MAX_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=rx.MAX_MODEL_LEN)
    args = ap.parse_args()

    files = []                                # (파일명, 법령명, law_type, units)
    for path in sorted(glob.glob(os.path.join(CHUNK_DIR, "[0-9]*.json"))):
        head = None
        try:
            head = json.load(open(path, encoding="utf-8"))[0]
            if head["source"] in EXCLUDE:
                continue
            units = load_units(path)
        except Exception as e:                # 읽기·항 복원 실패 → 빈 노드 파일로 저장해 파일 수 유지
            record(os.path.basename(path), f"항 복원 실패 {type(e).__name__}: {e}")
            head = head or {"source": os.path.basename(path), "law_type": ""}
            units = []
        files.append((os.path.basename(path), head["source"].removesuffix(".txt"), head["law_type"], units))
    jobs = [(fi, ui) for fi, f in enumerate(files) for ui, u in enumerate(f[3]) if not u["deleted"]]
    n_units = sum(len(f[3]) for f in files)
    print(f"파일 {len(files)}개 · 항 {n_units}개 · LLM 대상 {len(jobs)}개 (삭제 항 제외)")
    if args.dry_run:
        return
    if args.limit:
        jobs = jobs[:args.limit]
    if args.only:                             # 지정 항만: (파일번호, 조, 항번호) 일치
        want = {tuple(o.split(":")) for o in args.only}
        jobs = [(fi, ui) for fi, ui in jobs
                if (files[fi][0][:2], files[fi][3][ui]["_art"], str(paragraph_no(files[fi][3][ui]["_para"]) or 0))
                in want]
        print(f"재실행 대상 {len(jobs)}개")

    if not os.path.isdir(rx.MODEL_DIR):
        raise SystemExit(f"모델 디렉터리 없음: {rx.MODEL_DIR}")
    t0 = time.time()
    llm = rx.LLM(model=rx.MODEL_DIR, tokenizer=rx.MODEL_DIR, max_model_len=args.max_model_len,
                 gpu_memory_utilization=0.92, seed=rx.SEED, dtype="auto", quantization=rx.QUANT or None)
    sp = rx.SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=rx.SEED,
                           structured_outputs=rx.StructuredOutputsParams(json=rx.LLM_SCHEMA))
    load_s = time.time() - t0

    t1 = time.time()
    batch = [[{"role": "system", "content": rx.SYSTEM_PROMPT},
              {"role": "user", "content": rx.build_user_prompt(files[fi][1], files[fi][3][ui]["article"],
                                                                files[fi][3][ui]["text"])}]
             for fi, ui in jobs]
    try:
        outs = [r.outputs[0] for r in llm.chat(batch, sampling_params=sp, use_tqdm=True)]
    except Exception as e:                    # 배치 전체 실패 → 한 건씩 재시도, 실패 건은 기록
        record("LLM 배치", f"{type(e).__name__}: {e} → 한 건씩 재시도")
        outs = []
        for (fi, ui), msgs in zip(jobs, batch):
            try:
                outs.append(llm.chat([msgs], sampling_params=sp, use_tqdm=False)[0].outputs[0])
            except Exception as e2:
                outs.append(None)
                record(f"LLM {files[fi][0]} #{ui}", f"{type(e2).__name__}: {e2}")
    infer_s = time.time() - t1
    llm_text = {job: o.text for job, o in zip(jobs, outs) if o is not None and o.finish_reason == "stop"}
    truncated = {job for job, o in zip(jobs, outs) if o is None or o.finish_reason != "stop"}
    for (fi, ui), o in zip(jobs, outs):
        if o is not None and o.finish_reason != "stop":
            u = files[fi][3][ui]
            record(f"{files[fi][1]} {u['article']} {u['_para'] or '-'}",
                   f"LLM 출력 잘림 finish={o.finish_reason} tokens={len(o.token_ids)}")

    log = [f"load={load_s:.1f}s infer={infer_s:.1f}s jobs={len(jobs)} 잘림={len(truncated)}"]
    if args.limit:                            # 일부만 돌린 경우 저장하지 않고 결과만 출력
        for fi, ui in jobs:
            node = to_node(files[fi][1], files[fi][3][ui], llm_text.get((fi, ui), "{"))
            print(json.dumps({k: node[k] for k in node if k != "text"}, ensure_ascii=False))
        print(log[0])
        print("\n".join(errors))
        return

    if args.only:                             # 기존 노드 파일에서 해당 항만 교체, 예외 기록도 해당 항만 갱신
        old_err = open(ERR_PATH, encoding="utf-8").read().splitlines()[1:] if os.path.exists(ERR_PATH) else []
        for fi, ui in jobs:
            fname, law, _, units = files[fi]
            u = units[ui]
            where = f"[{law} {u['article']} {u['_para'] or '-'}]"
            old_err = [e for e in old_err if not e.startswith(where)]
            node = to_node(law, u, llm_text.get((fi, ui), "{"))
            path = os.path.join(NODE_DIR, fname)
            data = json.load(open(path, encoding="utf-8"))
            data["nodes"][ui] = node
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            print(f"{fname} {u['article']} {u['_para']}: 참조 {len(node['refs'])}개 · 오류 {node.get('error', '없음')}")
        with open(ERR_PATH, "w", encoding="utf-8") as f:
            all_err = old_err + errors
            f.write(f"예외 {len(all_err)}건\n" + "\n".join(all_err) + "\n")
        print(log[0])
        return

    os.makedirs(NODE_DIR, exist_ok=True)
    for fi, (fname, law, law_type, units) in enumerate(files):
        nodes = []
        for ui, u in enumerate(units):
            if u["deleted"]:
                text = None                   # 삭제 항: LLM 호출 없이 refs=[]
            elif (fi, ui) in truncated:
                text = "{"                    # 잘린 출력·호출 실패 → JSON 실패로 기록
            else:
                text = llm_text[(fi, ui)]
            try:
                nodes.append(to_node(law, u, text))
            except Exception as e:            # 예상 못한 실패 → 최소 노드로 남김
                record(f"{law} {u['article']} {u['_para'] or '-'}", f"노드 생성 실패 {type(e).__name__}: {e}")
                nodes.append({"article": u["_art"], "article_title": u["article"], "law": law,
                              "paragraph": paragraph_no(u["_para"]), "refs": [], "chapter": u["chapter"],
                              "deleted": u["deleted"], "chunk_ids": u["chunk_ids"], "text": u["text"],
                              "error": f"{type(e).__name__}: {e}"})
        try:
            with open(os.path.join(NODE_DIR, fname), "w", encoding="utf-8") as f:
                json.dump({"source": law + ".txt", "law": law, "law_type": law_type, "nodes": nodes},
                          f, ensure_ascii=False, indent=1)
        except Exception as e:
            record(fname, f"저장 실패 {type(e).__name__}: {e}")
        errs = [n for n in nodes if "error" in n]
        log.append(f"{fname}: 항 {len(nodes)} · 참조 {sum(len(n['refs']) for n in nodes)} · 오류 {len(errs)}")
        log += [f"    {n['article']} {n['paragraph']}: {n['error']}" for n in errs]
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    with open(ERR_PATH, "w", encoding="utf-8") as f:
        f.write(f"예외 {len(errors)}건\n" + "\n".join(errors) + "\n")
    print("\n".join(log))
    print(f"저장: {NODE_DIR} ({len(files)}개) · 로그: {LOG_PATH} · 예외 {len(errors)}건: {ERR_PATH}")


if __name__ == "__main__":
    main()
