"""
LLM-based pairwise redundancy evaluator for schema fields.

Purpose
-------
Precompute a *redundancy matrix* between fields, to be used as a component of
the RL reward signal during schema construction.

Inputs
------
- coverage_fields_forms.csv (required):
  rows: fields
  columns: form IDs
  values: 0/1 indicating whether the field exists in that form

- forms CSV (required):
  the original dataset; used to pull narrative context for sampled forms

- field_candidates_report.csv (optional):
  used to enrich prompts with field descriptions (not required)

Output
------
Writes a symmetric CSV matrix:
  redundancy[field_a][field_b] in [0.0, 1.0] (one decimal)
Diagonal is 1.0.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


BLK35_COLS = (
    "BLK_35_PROBLEM_DESC",
    "BLK_35_RECOMMEND_SOLUTION",
    "BLK_35_ACTUAL_SOLUTION",
    "BLK_35_CLOSING_REMARKS",
)


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
    m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        raise ValueError("Could not find JSON object in model output.")
    return json.loads(m.group(1))


def _discover_columns(csv_path: str) -> List[str]:
    with open(csv_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [h.strip() for h in header]


def _iter_csv_rows(csv_path: str) -> Iterable[Dict[str, str]]:
    with open(csv_path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


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


def _row_to_context(row: Dict[str, str], *, equip_col: Optional[str], max_chars_per_field: int) -> str:
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


def _build_prompt(batch_size: int) -> ChatPromptTemplate:
    sys_msg = f"""You are helping build an automatic schema for maritime maintenance forms.
We have candidate schema *fields*. A component of the RL reward is **redundancy** between fields:
two fields are redundant if they capture the same underlying information.

You will be given up to {batch_size} field-pairs. For each pair:
- You will see a few example forms that contain Field A and a few that contain Field B.
- Return a redundancy score in [0.0, 1.0] with **one decimal**.
  - 0.0 = completely different information
  - 1.0 = essentially the same information

Return ONLY valid JSON:
{{
  "scores": {{
    "field_a||field_b": 0.0
  }}
}}
"""
    sys_msg = sys_msg.replace("{", "{{").replace("}", "}}")

    human_msg = """Evaluate the following field pairs.

{pairs_blob}
"""
    return ChatPromptTemplate.from_messages([("system", sys_msg), ("human", human_msg)])


@dataclass(frozen=True)
class Pair:
    a: str
    b: str

    def key(self) -> str:
        x, y = sorted((self.a, self.b))
        return f"{x}||{y}"


class RedundancyEvaluator:
    def __init__(
        self,
        *,
        coverage_csv_path: str,
        forms_csv_path: str = "data/2-Kilo-Data7-8-2025_obfuscated(Sheet1).csv",
        field_report_csv_path: Optional[str] = None,
        batch_size: int = 5,
        sample_forms_per_field: int = 2,
        seed: int = 13,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_chars_per_field: int = 700,
        output_csv_path: str = "preprocessing_outputs/redundancy_fields_matrix.csv",
        debug_print_llm_response: bool = False,
    ) -> None:
        self.coverage_csv_path = coverage_csv_path
        self.forms_csv_path = forms_csv_path
        self.field_report_csv_path = field_report_csv_path
        self.batch_size = max(1, int(batch_size))
        self.sample_forms_per_field = max(1, int(sample_forms_per_field))
        self.seed = int(seed)
        self.model = model
        self.temperature = float(temperature)
        self.max_chars_per_field = int(max_chars_per_field)
        self.output_csv_path = output_csv_path
        self.debug_print_llm_response = bool(debug_print_llm_response)

        self.rng = random.Random(self.seed)

        self.field_to_forms: Dict[str, List[str]] = self._load_coverage_mapping(self.coverage_csv_path)
        self.fields: List[str] = list(self.field_to_forms.keys())

        self.field_descriptions: Dict[str, str] = (
            self._load_field_descriptions(field_report_csv_path) if field_report_csv_path else {}
        )

        self.field_to_sampled_forms: Dict[str, List[str]] = self._sample_forms_per_field()

        needed_form_ids = {fid for fids in self.field_to_sampled_forms.values() for fid in fids}
        self.form_context: Dict[str, str] = self._load_form_contexts(needed_form_ids)

        self.llm = ChatOpenAI(model=self.model, temperature=self.temperature)
        self.prompt = _build_prompt(self.batch_size)

    def _load_coverage_mapping(self, path: str) -> Dict[str, List[str]]:
        with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            form_ids = header[1:]
            mapping: Dict[str, List[str]] = {}
            for row in reader:
                if not row:
                    continue
                field_name = row[0].strip()
                if not field_name:
                    continue
                covered: List[str] = []
                for j, v in enumerate(row[1:]):
                    if v.strip() == "1":
                        covered.append(form_ids[j])
                mapping[field_name] = covered
        return mapping

    def _load_field_descriptions(self, path: str) -> Dict[str, str]:
        desc: Dict[str, str] = {}
        try:
            with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("field") or "").strip()
                    if not name:
                        continue
                    d = (row.get("description") or "").strip()
                    if d:
                        desc[name] = d
        except FileNotFoundError:
            return {}
        return desc

    def _sample_forms_per_field(self) -> Dict[str, List[str]]:
        sampled: Dict[str, List[str]] = {}
        for field_name, form_ids in self.field_to_forms.items():
            if not form_ids:
                sampled[field_name] = []
                continue
            if len(form_ids) <= self.sample_forms_per_field:
                sampled[field_name] = list(form_ids)
            else:
                sampled[field_name] = self.rng.sample(form_ids, k=self.sample_forms_per_field)
        return sampled

    def _load_form_contexts(self, needed_form_ids: set) -> Dict[str, str]:
        if not needed_form_ids:
            return {}

        columns = _discover_columns(self.forms_csv_path)
        equip_col = _find_equipment_column(columns)
        form_id_col = _find_form_id_column(columns)

        found: Dict[str, str] = {}
        for idx, row in enumerate(_iter_csv_rows(self.forms_csv_path)):
            raw_id = row.get(form_id_col) if form_id_col is not None else None
            raw_id = _clean_text(raw_id) if raw_id is not None else ""
            form_id = raw_id if raw_id else str(idx)
            form_key = f"form_{form_id}"
            if form_key not in needed_form_ids:
                continue
            found[form_key] = _row_to_context(row, equip_col=equip_col, max_chars_per_field=self.max_chars_per_field)
            if len(found) >= len(needed_form_ids):
                break

        missing = needed_form_ids.difference(found.keys())
        if missing:
            print(f"WARNING: could not find {len(missing)} sampled forms in CSV.", file=sys.stderr)
        return found

    def _pairs(self, *, max_pairs: Optional[int] = None) -> List[Pair]:
        pairs: List[Pair] = []
        for i in range(len(self.fields)):
            for j in range(i + 1, len(self.fields)):
                pairs.append(Pair(self.fields[i], self.fields[j]))
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
        return pairs

    def _pair_blob(self, pair: Pair) -> str:
        a = pair.a
        b = pair.b
        a_desc = self.field_descriptions.get(a, "")
        b_desc = self.field_descriptions.get(b, "")
        a_forms = self.field_to_sampled_forms.get(a, [])
        b_forms = self.field_to_sampled_forms.get(b, [])

        def fmt_examples(field_name: str, desc: str, form_ids: List[str]) -> str:
            header = f"FIELD: {field_name}"
            if desc:
                header += f"\nDESCRIPTION: {desc}"
            exs: List[str] = []
            for k, fid in enumerate(form_ids[:2]):
                ctx = self.form_context.get(fid, "")
                exs.append(f"EXAMPLE_{k+1} ({fid})\n{ctx or '[missing context]'}")
            return header + "\n\n" + "\n\n---\n\n".join(exs)

        return (
            f"PAIR_KEY: {pair.key()}\n\n"
            + fmt_examples(a, a_desc, a_forms)
            + "\n\n====\n\n"
            + fmt_examples(b, b_desc, b_forms)
        )

    def _call_llm_for_batch(self, batch: List[Pair]) -> Dict[str, float]:
        pairs_blob = "\n\n##########\n\n".join(self._pair_blob(p) for p in batch)
        resp = self.llm.invoke(self.prompt.format_messages(pairs_blob=pairs_blob))
        if self.debug_print_llm_response:
            print("\n=== RAW LLM RESPONSE (redundancy evaluator) ===", file=sys.stderr)
            print(resp.content, file=sys.stderr)
            print("=== END RAW LLM RESPONSE ===\n", file=sys.stderr)
        data = _best_effort_json_load(resp.content)
        scores_obj = data.get("scores", data)
        if not isinstance(scores_obj, dict):
            return {}

        out: Dict[str, float] = {}
        for k, v in scores_obj.items():
            if not isinstance(k, str):
                continue
            try:
                score = float(v)
            except Exception:
                continue
            score = max(0.0, min(1.0, score))
            score = round(score, 1)
            out[k.strip()] = score
        return out

    def compute_all(self, *, max_pairs: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        pairs = self._pairs(max_pairs=max_pairs)
        results: Dict[str, Dict[str, float]] = {f: {} for f in self.fields}

        for f in self.fields:
            results[f][f] = 1.0

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            print(
                f"Evaluating redundancy batch {start//self.batch_size + 1} / {((len(pairs)-1)//self.batch_size)+1}",
                file=sys.stderr,
            )
            batch_scores = self._call_llm_for_batch(batch)

            for p in batch:
                key = p.key()
                score = batch_scores.get(key)
                if score is None:
                    score = 0.0
                a, b = sorted((p.a, p.b))
                results[a][b] = score
                results[b][a] = score

        self._write_matrix_csv(results)
        return results

    def _write_matrix_csv(self, matrix: Dict[str, Dict[str, float]]) -> None:
        os.makedirs(os.path.dirname(self.output_csv_path) or ".", exist_ok=True)
        fields_sorted = list(self.fields)
        with open(self.output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["field", *fields_sorted])
            for row_field in fields_sorted:
                row_vals = []
                for col_field in fields_sorted:
                    v = matrix.get(row_field, {}).get(col_field, 0.0)
                    row_vals.append(f"{float(v):.1f}")
                writer.writerow([row_field, *row_vals])

