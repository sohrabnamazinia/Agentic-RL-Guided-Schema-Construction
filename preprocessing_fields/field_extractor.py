"""
Field (schema-attribute) candidate extractor for maritime maintenance forms.

This script is a preprocessing step for RL-based schema construction.

What it does
------------
1) **LLM-assisted field discovery**:
   - Samples multiple forms (rows) from a CSV.
   - Sends batches of sampled rows (Block 35 narratives + equipment) to an LLM.
   - The LLM proposes a list of common **fields** (name, description, type, source columns)
     and lightweight **presence indicators** (keywords/phrases).

2) **Deterministic coverage scan**:
   - Scans forms and estimates whether each field exists in each form:
     - For base fields tied to a specific column: present if that column is non-empty.
     - For other fields: present if any indicator keyword/phrase appears in the combined
       narrative text blob (Block 35 + equipment).

Outputs (2 CSV files)
---------------------
1) Field report (default: `preprocessing_outputs/field_candidates_report.csv`)
   - One row per field with coverage counts and metadata.

2) Coverage matrix (default: `preprocessing_outputs/coverage_fields_forms.csv`)
   - Rows are fields, columns are forms (form IDs), values are 0/1.
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

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


BLK35_COLS = (
    "BLK_35_PROBLEM_DESC",
    "BLK_35_RECOMMEND_SOLUTION",
    "BLK_35_ACTUAL_SOLUTION",
    "BLK_35_CLOSING_REMARKS",
)


BASE_FIELDS = [
    {
        "field": "problem_description",
        "description": "Narrative describing the observed problem/defect.",
        "value_type": "string",
        "extractable_from": ["BLK_35_PROBLEM_DESC"],
        "presence_indicators": [],
    },
    {
        "field": "recommended_solution",
        "description": "Suggested corrective action/plan before execution.",
        "value_type": "string",
        "extractable_from": ["BLK_35_RECOMMEND_SOLUTION"],
        "presence_indicators": [],
    },
    {
        "field": "actual_solution",
        "description": "What was actually done to resolve the issue.",
        "value_type": "string",
        "extractable_from": ["BLK_35_ACTUAL_SOLUTION"],
        "presence_indicators": [],
    },
    {
        "field": "closing_remarks",
        "description": "Closing notes/remarks about completion, status, or follow-up.",
        "value_type": "string",
        "extractable_from": ["BLK_35_CLOSING_REMARKS"],
        "presence_indicators": [],
    },
    {
        "field": "equipment",
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
    return s or "field"


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 12)] + " …(truncated)"


def _best_effort_json_load(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
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
    for c in columns:
        if c.startswith("BLK_5_Equip"):
            return c
    return None


def _find_form_id_column(columns: Sequence[str]) -> Optional[str]:
    for name in ("Record Num", "RECORD_NUM", "record_num"):
        if name in columns:
            return name
    if columns and columns[0] == "":
        return ""
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
    sys_msg = f"""You are a schema engineer for maritime maintenance / sailor forms.
You will be shown multiple form rows (each row includes equipment and Block 35 narratives).
Your job: propose a list of COMMON, USEFUL schema fields that could be consistently extracted
across many forms, to help build a structured schema.

Hard requirements:
- Return ONLY valid JSON.
- Include these base fields: problem_description, recommended_solution, actual_solution, closing_remarks, equipment.
- Output up to {top_k} total fields.
- Field names MUST be snake_case.
- Provide presence_indicators: a short list of lowercase keywords/phrases that suggest the field is present in text.
  These will be used for approximate coverage counting via substring matching.

Return JSON in this exact shape:
{{
  "fields": [
    {{
      "field": "snake_case_name",
      "description": "what it represents",
      "value_type": "string|enum|number|date|boolean|list|string_list",
      "extractable_from": ["BLK_35_PROBLEM_DESC", "BLK_35_RECOMMEND_SOLUTION", "..."],
      "presence_indicators": ["keyword1", "keyword2", "phrase3"]
    }}
  ]
}}
"""
    sys_msg = sys_msg.replace("{", "{{").replace("}", "}}")

    human_msg = (
        "Here are sample rows:\n\n"
        "{rows}\n\n"
        "Propose the candidate schema fields now."
    )
    return ChatPromptTemplate.from_messages([("system", sys_msg), ("human", human_msg)])


@dataclass
class FieldCandidate:
    field: str
    description: str = ""
    value_type: str = "string"
    extractable_from: List[str] = field(default_factory=list)
    presence_indicators: List[str] = field(default_factory=list)

    def merge_from(self, other: "FieldCandidate") -> None:
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
                i = re.sub(r"\s+", " ", i.strip().lower()).strip()
                if i and i not in existing_i:
                    self.presence_indicators.append(i)
                    existing_i.add(i)


def _ensure_base_fields(fields: Dict[str, FieldCandidate]) -> None:
    for base in BASE_FIELDS:
        name = base["field"]
        if name not in fields:
            fields[name] = FieldCandidate(
                field=name,
                description=base["description"],
                value_type=base["value_type"],
                extractable_from=list(base["extractable_from"]),
                presence_indicators=list(base["presence_indicators"]),
            )


def propose_fields_with_llm(
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
    llm_debug_output_path: Optional[str] = None,
) -> Dict[str, FieldCandidate]:
    columns = _discover_columns(csv_path)
    equip_col = _find_equipment_column(columns)

    rng = random.Random(seed)
    reservoir: List[Dict[str, str]] = []
    for idx, row in enumerate(_iter_csv_rows(csv_path, limit=None)):
        if len(reservoir) < sample_forms:
            reservoir.append(row)
        else:
            j = rng.randint(0, idx)
            if j < sample_forms:
                reservoir[j] = row
        if idx + 1 >= max(sample_forms, sample_scan_limit):
            break

    examples: List[str] = []
    for r in reservoir:
        ex = _row_to_llm_example(r, equip_col=equip_col, max_chars_per_field=max_chars_per_field)
        if ex:
            examples.append(ex)
    if not examples:
        raise RuntimeError("No non-empty sample examples found in the specified columns.")

    llm = ChatOpenAI(model=model, temperature=temperature)
    prompt = _build_prompt(top_k=top_k)

    fields: Dict[str, FieldCandidate] = {}
    _ensure_base_fields(fields)

    for start in range(0, len(examples), llm_rows_per_call):
        chunk = examples[start : start + llm_rows_per_call]
        rows_blob = "\n\n---\n\n".join(f"ROW_{start + i + 1}\n{chunk[i]}" for i in range(len(chunk)))
        resp = llm.invoke(prompt.format_messages(rows=rows_blob))

        if llm_debug_output_path and start == 0:
            os.makedirs(os.path.dirname(llm_debug_output_path) or ".", exist_ok=True)
            with open(llm_debug_output_path, "w", encoding="utf-8") as f:
                f.write(resp.content)

        data = _best_effort_json_load(resp.content)
        proposed = data.get("fields") or data.get("attributes") or []
        if not isinstance(proposed, list):
            continue
        for item in proposed:
            if not isinstance(item, dict):
                continue
            name = _snake_case(str(item.get("field") or item.get("attribute") or "").strip())
            if not name:
                continue
            cand = FieldCandidate(
                field=name,
                description=str(item.get("description", "") or "").strip(),
                value_type=str(item.get("value_type", "") or "string").strip() or "string",
                extractable_from=[str(x).strip() for x in (item.get("extractable_from") or []) if str(x).strip()],
                presence_indicators=[
                    re.sub(r"\s+", " ", str(x).strip().lower())
                    for x in (item.get("presence_indicators") or [])
                    if str(x).strip()
                ],
            )
            if name in fields:
                fields[name].merge_from(cand)
            else:
                fields[name] = cand

        _ensure_base_fields(fields)

    base_names = {b["field"] for b in BASE_FIELDS}
    if len(fields) > top_k:
        keep: Dict[str, FieldCandidate] = {k: v for k, v in fields.items() if k in base_names}
        others = [v for k, v in fields.items() if k not in base_names]
        others.sort(key=lambda x: len(x.presence_indicators), reverse=True)
        for v in others:
            if len(keep) >= top_k:
                break
            keep[v.field] = v
        fields = keep

    return fields


def _build_presence_regex(indicators: Sequence[str]) -> Optional[re.Pattern]:
    toks = [t.strip().lower() for t in indicators if t and t.strip()]
    toks = [t for t in toks if len(t) >= 3]
    if not toks:
        return None
    toks = list(dict.fromkeys(toks))
    pattern = "(" + "|".join(re.escape(t) for t in toks[:50]) + ")"
    return re.compile(pattern, flags=re.IGNORECASE)


def _compute_field_presence_for_row(
    *,
    row: Dict[str, str],
    fields: Dict[str, FieldCandidate],
    detectors: Dict[str, Optional[re.Pattern]],
    equip_col: Optional[str],
) -> Dict[str, bool]:
    base_present: Dict[str, bool] = {
        "problem_description": bool(_clean_text(row.get("BLK_35_PROBLEM_DESC"))),
        "recommended_solution": bool(_clean_text(row.get("BLK_35_RECOMMEND_SOLUTION"))),
        "actual_solution": bool(_clean_text(row.get("BLK_35_ACTUAL_SOLUTION"))),
        "closing_remarks": bool(_clean_text(row.get("BLK_35_CLOSING_REMARKS"))),
        "equipment": bool(_clean_text(row.get(equip_col))) if equip_col else False,
    }

    parts: List[str] = []
    if equip_col:
        parts.append(_clean_text(row.get(equip_col)))
    for col in BLK35_COLS:
        parts.append(_clean_text(row.get(col)))
    blob = "\n".join(p for p in parts if p).lower()

    present: Dict[str, bool] = {}
    for name in fields.keys():
        if name in base_present:
            present[name] = base_present[name]
            continue
        det = detectors.get(name)
        present[name] = bool(blob and det and det.search(blob))
    return present


def scan_coverage_and_counts(
    *,
    csv_path: str,
    fields: Dict[str, FieldCandidate],
    max_forms_to_scan: Optional[int],
) -> Tuple[Dict[str, int], int, List[str], Dict[str, bytearray]]:
    columns = _discover_columns(csv_path)
    equip_col = _find_equipment_column(columns)
    form_id_col = _find_form_id_column(columns)

    detectors: Dict[str, Optional[re.Pattern]] = {k: _build_presence_regex(v.presence_indicators) for k, v in fields.items()}
    counts: Dict[str, int] = {k: 0 for k in fields.keys()}

    form_ids: List[str] = []
    matrix: Dict[str, bytearray] = {k: bytearray() for k in fields.keys()}

    scanned = 0
    for row_idx, row in enumerate(_iter_csv_rows(csv_path, limit=max_forms_to_scan)):
        scanned += 1

        raw_id = row.get(form_id_col) if form_id_col is not None else None
        raw_id = _clean_text(raw_id) if raw_id is not None else ""
        form_id = raw_id if raw_id else str(row_idx)
        form_ids.append(f"form_{form_id}")

        present = _compute_field_presence_for_row(
            row=row,
            fields=fields,
            detectors=detectors,
            equip_col=equip_col,
        )

        for name, is_present in present.items():
            if is_present:
                counts[name] += 1
                matrix[name].append(1)
            else:
                matrix[name].append(0)

        if scanned % 2000 == 0:
            print(f"Scanned {scanned} forms...", file=sys.stderr)

    return counts, scanned, form_ids, matrix


def write_field_report_csv(
    *,
    output_path: str,
    fields: Dict[str, FieldCandidate],
    counts: Dict[str, int],
    scanned: int,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows: List[Dict[str, str]] = []
    for name, cand in fields.items():
        n = int(counts.get(name, 0))
        pct = (100.0 * n / scanned) if scanned else 0.0
        rows.append(
            {
                "field": name,
                "forms_with_field": str(n),
                "percent_of_scanned": f"{pct:.2f}",
                "value_type": cand.value_type or "string",
                "extractable_from": ";".join(cand.extractable_from),
                "description": cand.description,
                "indicators": ";".join(cand.presence_indicators),
            }
        )

    rows.sort(key=lambda r: int(r["forms_with_field"]), reverse=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "field",
                "forms_with_field",
                "percent_of_scanned",
                "value_type",
                "extractable_from",
                "description",
                "indicators",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_coverage_matrix_csv(
    *,
    output_path: str,
    fields: Dict[str, FieldCandidate],
    form_ids: List[str],
    matrix: Dict[str, bytearray],
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    header = ["field", *form_ids]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name in fields.keys():
            writer.writerow([name, *matrix[name]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_path",
        default="data/2-Kilo-Data7-8-2025_obfuscated(Sheet1).csv",
        help="Path to input CSV (each row is a form).",
    )
    ap.add_argument(
        "--output_path",
        default="preprocessing_outputs/field_candidates_report.csv",
        help="Output field report CSV.",
    )
    ap.add_argument(
        "--coverage_output_path",
        default="preprocessing_outputs/coverage_fields_forms.csv",
        help="Output coverage matrix CSV (fields x forms with 0/1).",
    )
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--sample_forms", type=int, default=250)
    ap.add_argument("--sample_scan_limit", type=int, default=2500)
    ap.add_argument("--llm_rows_per_call", type=int, default=20)
    ap.add_argument("--max_chars_per_field", type=int, default=700)
    ap.add_argument(
        "--max_forms_to_scan",
        type=int,
        default=0,
        help="If >0, limit number of forms scanned for coverage/counting (useful for quick runs).",
    )
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--llm_debug_output_path",
        default="",
        help="If set, writes the raw LLM output for the first batch to this path.",
    )
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY in environment.", file=sys.stderr)
        return 2

    max_forms = args.max_forms_to_scan if args.max_forms_to_scan and args.max_forms_to_scan > 0 else None

    print("Proposing candidate fields with LLM...", file=sys.stderr)
    fields = propose_fields_with_llm(
        csv_path=args.csv_path,
        model=args.model,
        temperature=args.temperature,
        sample_forms=args.sample_forms,
        sample_scan_limit=args.sample_scan_limit,
        llm_rows_per_call=args.llm_rows_per_call,
        max_chars_per_field=args.max_chars_per_field,
        top_k=args.top_k,
        seed=args.seed,
        llm_debug_output_path=(args.llm_debug_output_path.strip() or None),
    )
    print(f"Proposed {len(fields)} fields. Scanning dataset for coverage...", file=sys.stderr)

    counts, scanned, form_ids, matrix = scan_coverage_and_counts(
        csv_path=args.csv_path,
        fields=fields,
        max_forms_to_scan=max_forms,
    )

    write_field_report_csv(output_path=args.output_path, fields=fields, counts=counts, scanned=scanned)
    write_coverage_matrix_csv(output_path=args.coverage_output_path, fields=fields, form_ids=form_ids, matrix=matrix)

    print(
        f"Wrote field report: {args.output_path} | coverage matrix: {args.coverage_output_path} "
        f"(scanned_forms={scanned}, fields={len(fields)})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

