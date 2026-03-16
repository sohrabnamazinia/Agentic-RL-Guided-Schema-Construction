"""
Concept (schema-attribute) candidate extractor for maritime forms.

Goal
----
Given a CSV where each row is a form, use an LLM to propose a set of common,
useful schema *attributes* (aka concepts/candidates) from the most important
blocks, then scan the full dataset to estimate how many forms each attribute
appears in.

This is intentionally NOT "concept extraction per-row" (too expensive); instead:
1) Sample rows -> LLM proposes ~50 candidate attributes + presence indicators.
2) Full scan -> count per attribute using simple indicator matching + known cols.

Output
------
Writes a single CSV report with:
attribute, forms_with_attribute, percent_of_scanned, value_type, extractable_from,
description, indicators

Usage
-----
python3 preprocessing/concept_exrtacter.py \
  --csv_path "data/2-Kilo-Data7-8-2025_obfuscated(Sheet1).csv" \
  --output_path "preprocessing/concept_candidates_report.csv"

Environment
-----------
Requires OPENAI_API_KEY set in environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


BLK35_COLS = (
    "BLK_35_PROBLEM_DESC",
    "BLK_35_RECOMMEND_SOLUTION",
    "BLK_35_ACTUAL_SOLUTION",
    "BLK_35_CLOSING_REMARKS",
)


BASE_ATTRIBUTES = [
    {
        "attribute": "problem_description",
        "description": "Narrative describing the observed problem/defect.",
        "value_type": "string",
        "extractable_from": ["BLK_35_PROBLEM_DESC"],
        "presence_indicators": [],
    },
    {
        "attribute": "recommended_solution",
        "description": "Suggested corrective action/plan before execution.",
        "value_type": "string",
        "extractable_from": ["BLK_35_RECOMMEND_SOLUTION"],
        "presence_indicators": [],
    },
    {
        "attribute": "actual_solution",
        "description": "What was actually done to resolve the issue.",
        "value_type": "string",
        "extractable_from": ["BLK_35_ACTUAL_SOLUTION"],
        "presence_indicators": [],
    },
    {
        "attribute": "closing_remarks",
        "description": "Closing notes/remarks about completion, status, or follow-up.",
        "value_type": "string",
        "extractable_from": ["BLK_35_CLOSING_REMARKS"],
        "presence_indicators": [],
    },
    {
        "attribute": "equipment",
        "description": "Equipment / component referenced by the form.",
        "value_type": "string",
        "extractable_from": ["BLK_5_Equip*"],
        "presence_indicators": [],
    },
]


def _snake_case(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "attribute"


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    # Keep newlines minimal; DictReader already gives raw values.
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 12)] + " …(truncated)"


def _best_effort_json_load(text: str) -> dict:
    """
    Try parsing model output as JSON.
    Supports: pure JSON, or JSON embedded in markdown fences / extra text.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Extract the first JSON object/array substring.
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not m:
        raise ValueError("Could not find JSON object/array in model output.")
    return json.loads(m.group(1))


def _iter_csv_rows(csv_path: str, *, limit: Optional[int] = None) -> Iterable[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                return
            yield row


def _discover_columns(csv_path: str) -> List[str]:
    with open(csv_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [h.strip() for h in header]


def _find_equipment_column(columns: Sequence[str]) -> Optional[str]:
    # The dataset currently has "BLK_5_Equip Noun Name", but we support prefix matching.
    for c in columns:
        if c.startswith("BLK_5_Equip"):
            return c
    return None


def _row_to_llm_example(row: Dict[str, str], *, equip_col: Optional[str], max_chars_per_field: int) -> str:
    parts: List[str] = []
    if equip_col:
        equip = _clean_text(row.get(equip_col))
        if equip:
            parts.append(f"[{equip_col}]\n{_truncate(equip, max_chars_per_field)}")
    for col in BLK35_COLS:
        val = _clean_text(row.get(col))
        if val:
            parts.append(f"[{col}]\n{_truncate(val, max_chars_per_field)}")
    return "\n\n".join(parts).strip()


def _build_prompt(top_k: int) -> ChatPromptTemplate:
    # NOTE: LangChain templates use Python `.format()` under the hood for *all* messages.
    # That means any literal `{` / `}` in the template must be escaped as `{{` / `}}`.
    sys_msg = f"""You are a schema engineer for maritime maintenance / sailor forms.
You will be shown multiple form rows (each row includes equipment and Block 35 narratives).
Your job: propose a list of COMMON, USEFUL schema attributes (candidates) that could be
consistently extracted across many forms, to help build a structured schema.

Hard requirements:
- Return ONLY valid JSON.
- Include these base attributes: problem_description, recommended_solution, actual_solution, closing_remarks, equipment.
- Output up to {top_k} total attributes.
- Attribute names MUST be snake_case.
- Provide presence_indicators: a short list of lowercase keywords/phrases that suggest the attribute is present in text.
  These will be used for approximate counting via substring matching (no regex needed).

Return JSON in this exact shape:
{{
  "attributes": [
    {{
      "attribute": "snake_case_name",
      "description": "what it represents",
      "value_type": "string|enum|number|date|boolean|list|string_list",
      "extractable_from": ["BLK_35_PROBLEM_DESC", "BLK_35_RECOMMEND_SOLUTION", "..."],
      "presence_indicators": ["keyword1", "keyword2", "phrase3"]
    }}
  ]
}}
"""
    # Escape braces so the above JSON example is treated literally during templating.
    sys_msg = sys_msg.replace("{", "{{").replace("}", "}}")

    human_msg = (
        "Here are sample rows:\n\n"
        "{rows}\n\n"
        "Propose the candidate schema attributes now."
    )
    # Use templated messages so `{rows}` is properly formatted at runtime.
    # (Passing concrete HumanMessage/SystemMessage instances would not substitute variables.)
    return ChatPromptTemplate.from_messages([("system", sys_msg), ("human", human_msg)])


@dataclass
class AttributeCandidate:
    attribute: str
    description: str = ""
    value_type: str = "string"
    extractable_from: List[str] = field(default_factory=list)
    presence_indicators: List[str] = field(default_factory=list)

    def merge_from(self, other: "AttributeCandidate") -> None:
        if not self.description and other.description:
            self.description = other.description
        if self.value_type == "string" and other.value_type and other.value_type != "string":
            self.value_type = other.value_type
        if other.extractable_from:
            existing = set(self.extractable_from)
            for x in other.extractable_from:
                if x not in existing:
                    self.extractable_from.append(x)
                    existing.add(x)
        if other.presence_indicators:
            existing_i = set(self.presence_indicators)
            for i in other.presence_indicators:
                i = _snake_case(i).replace("_", " ") if "_" in i else i.strip().lower()
                i = re.sub(r"\s+", " ", i).strip()
                if i and i not in existing_i:
                    self.presence_indicators.append(i)
                    existing_i.add(i)


def _ensure_base_attributes(cands: Dict[str, AttributeCandidate]) -> None:
    for base in BASE_ATTRIBUTES:
        name = base["attribute"]
        if name not in cands:
            cands[name] = AttributeCandidate(
                attribute=name,
                description=base["description"],
                value_type=base["value_type"],
                extractable_from=list(base["extractable_from"]),
                presence_indicators=list(base["presence_indicators"]),
            )


def propose_attributes_with_llm(
    *,
    csv_path: str,
    model: str,
    temperature: float,
    sample_forms: int,
    sample_scan_limit: int,
    llm_rows_per_call: int,
    max_chars_per_field: int,
    top_k: int,
    seed: int,
) -> Dict[str, AttributeCandidate]:
    columns = _discover_columns(csv_path)
    equip_col = _find_equipment_column(columns)

    # Sample rows (reservoir sampling) with a hard cap on how many rows we scan,
    # so the "LLM proposal" stage doesn't spend minutes just reading the CSV.
    rng = random.Random(seed)
    buffer: List[Dict[str, str]] = []
    for idx, row in enumerate(_iter_csv_rows(csv_path, limit=None)):
        if len(buffer) < sample_forms:
            buffer.append(row)
        else:
            j = rng.randint(0, idx)
            if j < sample_forms:
                buffer[j] = row
        if (idx + 1) % 1000 == 0:
            print(f"Sampling rows for LLM: scanned={idx+1}, reservoir={len(buffer)}", file=sys.stderr)
        if idx + 1 >= max(sample_forms, sample_scan_limit):
            break

    examples: List[str] = []
    for r in buffer:
        ex = _row_to_llm_example(r, equip_col=equip_col, max_chars_per_field=max_chars_per_field)
        if ex:
            examples.append(ex)
    if not examples:
        raise RuntimeError("No non-empty sample examples found in the specified columns.")

    llm = ChatOpenAI(model=model, temperature=temperature)
    prompt = _build_prompt(top_k=top_k)

    cands: Dict[str, AttributeCandidate] = {}
    _ensure_base_attributes(cands)

    # Call the LLM on chunks of examples.
    for start in range(0, len(examples), llm_rows_per_call):
        chunk = examples[start : start + llm_rows_per_call]
        rows_blob = "\n\n---\n\n".join(
            f"ROW_{start + i + 1}\n{chunk[i]}" for i in range(len(chunk))
        )
        messages = prompt.format_messages(rows=rows_blob)
        resp = llm.invoke(messages)
        data = _best_effort_json_load(resp.content)
        attrs = data.get("attributes", [])
        if not isinstance(attrs, list):
            continue
        for a in attrs:
            if not isinstance(a, dict):
                continue
            name = _snake_case(str(a.get("attribute", "")).strip())
            if not name:
                continue
            cand = AttributeCandidate(
                attribute=name,
                description=str(a.get("description", "") or "").strip(),
                value_type=str(a.get("value_type", "") or "string").strip() or "string",
                extractable_from=[str(x).strip() for x in (a.get("extractable_from") or []) if str(x).strip()],
                presence_indicators=[
                    re.sub(r"\s+", " ", str(x).strip().lower())
                    for x in (a.get("presence_indicators") or [])
                    if str(x).strip()
                ],
            )
            if name in cands:
                cands[name].merge_from(cand)
            else:
                cands[name] = cand

        _ensure_base_attributes(cands)

    # Trim to top_k while always keeping base attributes.
    base_names = {b["attribute"] for b in BASE_ATTRIBUTES}
    if len(cands) > top_k:
        keep: Dict[str, AttributeCandidate] = {k: v for k, v in cands.items() if k in base_names}
        others = [v for k, v in cands.items() if k not in base_names]
        # Prefer attributes with more indicators (easier to count later).
        others.sort(key=lambda x: len(x.presence_indicators), reverse=True)
        for v in others:
            if len(keep) >= top_k:
                break
            keep[v.attribute] = v
        cands = keep

    return cands


def _build_presence_regex(indicators: Sequence[str]) -> Optional[re.Pattern]:
    # Simple, safe substring search using regex alternation.
    toks = [t.strip().lower() for t in indicators if t and t.strip()]
    toks = [t for t in toks if len(t) >= 3]  # drop too-short noisy indicators
    if not toks:
        return None
    toks = list(dict.fromkeys(toks))  # stable unique
    # Escape for literal matching; keep as "contains" match.
    pattern = "(" + "|".join(re.escape(t) for t in toks[:50]) + ")"  # cap alternation size
    return re.compile(pattern, flags=re.IGNORECASE)


def count_attribute_presence(
    *,
    csv_path: str,
    candidates: Dict[str, AttributeCandidate],
    max_forms_to_scan: Optional[int],
) -> Tuple[Dict[str, int], int]:
    columns = _discover_columns(csv_path)
    equip_col = _find_equipment_column(columns)

    # Precompute detectors.
    detectors: Dict[str, Optional[re.Pattern]] = {}
    for name, cand in candidates.items():
        detectors[name] = _build_presence_regex(cand.presence_indicators)

    counts: Dict[str, int] = {name: 0 for name in candidates.keys()}
    scanned = 0

    for row in _iter_csv_rows(csv_path, limit=max_forms_to_scan):
        scanned += 1

        # Base: present if the underlying column has any text.
        base_present: Dict[str, bool] = {}
        base_present["problem_description"] = bool(_clean_text(row.get("BLK_35_PROBLEM_DESC")))
        base_present["recommended_solution"] = bool(_clean_text(row.get("BLK_35_RECOMMEND_SOLUTION")))
        base_present["actual_solution"] = bool(_clean_text(row.get("BLK_35_ACTUAL_SOLUTION")))
        base_present["closing_remarks"] = bool(_clean_text(row.get("BLK_35_CLOSING_REMARKS")))
        base_present["equipment"] = bool(_clean_text(row.get(equip_col))) if equip_col else False

        # Text blob for approximate matching (Block 35 + equipment).
        parts: List[str] = []
        if equip_col:
            parts.append(_clean_text(row.get(equip_col)))
        for col in BLK35_COLS:
            parts.append(_clean_text(row.get(col)))
        blob = "\n".join(p for p in parts if p).lower()

        for name in candidates.keys():
            if name in base_present:
                if base_present[name]:
                    counts[name] += 1
                continue
            det = detectors.get(name)
            if det is None:
                continue
            if blob and det.search(blob):
                counts[name] += 1

        if scanned % 2000 == 0:
            print(f"Scanned {scanned} forms...", file=sys.stderr)

    return counts, scanned


def write_report_csv(
    *,
    output_path: str,
    candidates: Dict[str, AttributeCandidate],
    counts: Dict[str, int],
    scanned: int,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows: List[Dict[str, str]] = []
    for name, cand in candidates.items():
        n = int(counts.get(name, 0))
        pct = (100.0 * n / scanned) if scanned else 0.0
        rows.append(
            {
                "attribute": name,
                "forms_with_attribute": str(n),
                "percent_of_scanned": f"{pct:.2f}",
                "value_type": cand.value_type or "string",
                "extractable_from": ";".join(cand.extractable_from),
                "description": cand.description,
                "indicators": ";".join(cand.presence_indicators),
            }
        )

    rows.sort(key=lambda r: int(r["forms_with_attribute"]), reverse=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attribute",
                "forms_with_attribute",
                "percent_of_scanned",
                "value_type",
                "extractable_from",
                "description",
                "indicators",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_path",
        default="data/2-Kilo-Data7-8-2025_obfuscated(Sheet1).csv",
        help="Path to input CSV (each row is a form).",
    )
    ap.add_argument(
        "--output_path",
        default="preprocessing/concept_candidates_report.csv",
        help="Single output report file (CSV).",
    )
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--sample_forms", type=int, default=250, help="Rows sampled for LLM proposing candidates.")
    ap.add_argument(
        "--sample_scan_limit",
        type=int,
        default=2500,
        help="Max CSV rows to scan while building the LLM sample reservoir (speeds up LLM proposal stage).",
    )
    ap.add_argument("--llm_rows_per_call", type=int, default=20, help="How many sample rows to send per LLM call.")
    ap.add_argument("--max_chars_per_field", type=int, default=700)
    ap.add_argument(
        "--max_forms_to_scan",
        type=int,
        default=0,
        help="If >0, limit number of forms scanned for counting (useful for quick runs).",
    )
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY in environment.", file=sys.stderr)
        return 2

    max_forms = args.max_forms_to_scan if args.max_forms_to_scan and args.max_forms_to_scan > 0 else None

    print("Proposing candidate attributes with LLM...", file=sys.stderr)
    candidates = propose_attributes_with_llm(
        csv_path=args.csv_path,
        model=args.model,
        temperature=args.temperature,
        sample_forms=args.sample_forms,
        sample_scan_limit=args.sample_scan_limit,
        llm_rows_per_call=args.llm_rows_per_call,
        max_chars_per_field=args.max_chars_per_field,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(f"Proposed {len(candidates)} attributes. Scanning dataset for counts...", file=sys.stderr)

    counts, scanned = count_attribute_presence(
        csv_path=args.csv_path,
        candidates=candidates,
        max_forms_to_scan=max_forms,
    )

    write_report_csv(
        output_path=args.output_path,
        candidates=candidates,
        counts=counts,
        scanned=scanned,
    )

    print(
        f"Wrote report to: {args.output_path} (scanned_forms={scanned}, attributes={len(candidates)})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

