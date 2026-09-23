"""조문 1개 → Gemma → 참조 조문 목록(고정 스키마) 추출 단일 테스트.

실행: python prep/ref_extract_test.py
결과: 이 파일과 같은 디렉터리의 ref_extract_result.txt
"""
import json
import os
import time

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

MODEL_DIR = os.environ.get("PPS_MODEL_DIR", "/workspace/models/gemma-4-26B-A4B-it")
QUANT = os.environ.get("PPS_QUANT", "int8_per_channel_weight_only")   # script.py와 동일
MAX_MODEL_LEN = 16384
MAX_TOKENS = 4096
SEED = 20260826
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_extract_result.txt")

# ===== 입력: 테스트용 조문 원문(하드코딩) =====
LAW_NAME = "국가를 당사자로 하는 계약에 관한 법률 시행령"
ARTICLE_TEXT = """제12조(경쟁입찰의 참가자격)
  ①각 중앙관서의 장 또는 계약담당공무원은 다음 각호의 요건을 갖춘 자에 한하여 경쟁입찰에 참가하게 하여야 한다. <개정 1996.12.31, 1999.9.9, 2008.2.29, 2025.12.30>
    1. 삭제<1999.9.9>
    2. 다른 법령의 규정에 의하여 허가ㆍ인가ㆍ면허ㆍ등록ㆍ신고등을 요하거나 자격요건을 갖추어야 할 경우에는 당해 허가ㆍ인가ㆍ면허ㆍ등록ㆍ신고등을 받았거나 당해 자격요건에 적합할 것
    3. 보안측정등의 조사가 필요한 경우에는 관계기관으로부터 적합판정을 받을 것
    4. 기타 재정경제부령이 정하는 요건에 적합할 것
  ②「중소기업협동조합법」에 따른 중소기업협동조합이 물품의 제조ㆍ구매ㆍ임차 또는 용역에 관한 경쟁입찰에 참가하는 경우(제1항제2호에 따른 요건을 갖춘 조합원으로 하여금 해당 물품을 제조ㆍ공급 또는 임대하게 하거나 용역을 수행하게 하는 경우로 한정한다)에는 제1항제2호를 적용하지 아니한다. <신설 1999.9.9, 2005.9.8, 2006.5.30, 2007.10.10, 2013.6.17, 2024.12.24>
    1. 삭제<2007.10.10>
    2. 삭제<2007.10.10>
  ③ 법 제27조의5제1항에서 "대통령령으로 정하는 조세포탈 등을 한 자"란 다음 각 호의 어느 하나에 해당하는 자를 말한다. <신설 2013.12.30, 2017.3.27, 2018.12.4, 2019.9.17, 2021.2.17>
    1. 「조세범 처벌법」 제3조에 따른 조세 포탈세액이나 환급ㆍ공제받은 세액이 5억원 이상인 자
    2. 「관세법」 제270조에 따른 부정한 방법으로 관세를 감면받거나 면탈하거나 환급받은 세액이 5억원 이상인 자
    3. 「지방세기본법」 제102조에 따른 지방세 포탈세액이나 환급ㆍ공제 세액이 5억원 이상인 자
    4. 「국제조세조정에 관한 법률」 제53조에 따른 해외금융계좌의 신고의무를 위반하고, 그 신고의무 위반금액이 「조세범 처벌법」 제16조제1항에 따른 금액을 초과하는 자
    5. 「외국환거래법」 제18조에 따른 자본거래의 신고의무를 위반하고, 그 신고의무 위반금액이 같은 법 제29조제1항제3호에 해당하는 자
  ④ 각 중앙관서의 장 또는 계약담당공무원은 「형의 실효 등에 관한 법률」 제2조제5호에 따른 범죄경력자료의 회보서나 판결문 등의 입증서류를 제출하게 하는 등의 방법으로 계약상대방이 제3항 각 호의 어느 하나에 해당하는지를 계약 체결 전까지 확인하여야 한다. <신설 2013.12.30>
  ⑤ 각 중앙관서의 장 또는 계약담당공무원은 계약상대방이 입찰에 참가할 때에 제4항에 따른 입증서류를 제출하기 어려운 경우에는 제3항 각 호의 어느 하나에 해당하지 아니한다는 사실을 적은 서약서를 제출하게 할 수 있다. 이 경우 서약서에는 서약서에 적은 내용과 다른 사실이 발견된 때에는 계약을 해제ㆍ해지할 수 있고, 부정당업자제재처분을 받을 수 있다는 내용이 포함되어야 한다. <신설 2013.12.30>
  ⑥ 제3항에 해당하는 자에 대한 입찰 참가자격 제한에 관하여는 제76조제5항ㆍ제6항ㆍ제9항 및 제10항을 준용한다. <신설 2013.12.30, 2016.9.2, 2021.7.6, 2024.9.20>"""

# ===== 출력 스키마 =====
NULLABLE_STR = {"type": ["string", "null"]}
SCHEMA = {
    "type": "object",
    "properties": {
        "source": {
            "type": "object",
            "properties": {
                "law": {"type": "string"},
                "article": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["law", "article", "title"],
            "additionalProperties": False,
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_location": {"type": "string"},
                    "raw_expression": {"type": "string"},
                    "target_law": {"type": "string"},
                    "target_article": NULLABLE_STR,
                    "target_paragraph": NULLABLE_STR,
                    "target_item": NULLABLE_STR,
                },
                "required": ["source_location", "raw_expression", "target_law",
                             "target_article", "target_paragraph", "target_item"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["source", "references"],
    "additionalProperties": False,
}

# ===== 프롬프트 =====
SYSTEM_PROMPT = """당신은 한국 법령 조문에서 다른 조문을 가리키는 참조 표현을 빠짐없이 추출한다.
개정 이력 표기(<개정 ...>, <신설 ...>, 삭제<...>)는 참조가 아니므로 무시한다.
하나의 참조 표현이 여러 항 또는 여러 호를 나열하면(예: "제76조제5항ㆍ제6항") 항·호마다 reference를 하나씩 따로 만든다.

출력은 반드시 아래 스키마를 따른다.

source
- law: 현재 입력된 법령의 이름
- article: 현재 입력된 조 번호
- title: 현재 입력된 조의 제목

references
- 현재 조문 안에서 발견된 참조 조문 목록

각 reference 객체:
- source_location:
  현재 입력 조문 안에서 참조 표현이 등장한 위치
  예: "제6항", "제3항제4호"

- raw_expression:
  원문에 실제로 등장한 참조 표현
  예: "제76조제5항", "같은 법 제29조제1항제3호"

- target_law:
  참조 대상 법령의 정확한 이름
  "법"은 현재 시행령의 모법,
  "같은 법"은 직전에 명시된 법령을 문맥에 따라 해석한다.

- target_article:
  참조 대상의 조
  예: "제76조", "제27조의5"
  조가 특정되지 않으면 null

- target_paragraph:
  참조 대상의 항
  예: "제1항", "제10항"
  항이 특정되지 않으면 null

- target_item:
  참조 대상의 호
  예: "제2호", "제3호"
  호가 특정되지 않으면 null"""

USER_PROMPT = f"[법령명]\n{LAW_NAME}\n\n[조문]\n{ARTICLE_TEXT}"


def main() -> None:
    if not os.path.isdir(MODEL_DIR):          # 로컬 경로가 없으면 HF 다운로드로 넘어가지 않도록 즉시 중단
        raise SystemExit(f"모델 디렉터리 없음: {MODEL_DIR}")
    t0 = time.time()
    llm = LLM(model=MODEL_DIR, tokenizer=MODEL_DIR, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.92, seed=SEED, dtype="auto", quantization=QUANT or None)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=SEED,
                        structured_outputs=StructuredOutputsParams(json=SCHEMA))
    load_s = time.time() - t0

    t1 = time.time()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}]
    out = llm.chat([messages], sampling_params=sp, use_tqdm=False)[0].outputs[0]
    infer_s = time.time() - t1

    try:
        body = json.dumps(json.loads(out.text), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        body = out.text                       # 토큰 한도 등으로 잘린 경우 원문 그대로 저장

    header = (f"model={MODEL_DIR} quant={QUANT} load={load_s:.1f}s infer={infer_s:.1f}s "
              f"out_tokens={len(out.token_ids)} finish={out.finish_reason}\n")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(header + "\n" + body + "\n")
    print(header + body)
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
