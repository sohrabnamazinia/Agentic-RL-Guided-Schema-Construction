"""
Small runnable entrypoint for testing redundancy evaluation.

Example (quick test on a few pairs):
python3 preprocessing_reward/run_redundancy_eval.py --max_pairs 10
"""

from __future__ import annotations

import argparse
import os
import sys


def _ensure_repo_root_on_syspath() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_syspath()

from preprocessing_reward.redundancy_evaluator import RedundancyEvaluator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coverage_csv_path",
        default="preprocessing_outputs/coverage_fields_forms.csv",
        help="Path to coverage matrix from field_extractor.",
    )
    ap.add_argument(
        "--forms_csv_path",
        default="data/2-Kilo-Data7-8-2025_obfuscated(Sheet1).csv",
        help="Path to the original forms CSV.",
    )
    ap.add_argument(
        "--field_report_csv_path",
        default="",
        help="Optional path to field_candidates_report.csv for descriptions.",
    )
    ap.add_argument("--output_csv_path", default="preprocessing_outputs/redundancy_fields_matrix.csv")
    ap.add_argument("--batch_size", type=int, default=5)
    ap.add_argument("--max_pairs", type=int, default=0, help="If >0, only score this many pairs (quick test).")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_chars_per_field", type=int, default=700)
    ap.add_argument(
        "--debug_print_llm_response",
        action="store_true",
        help="Print raw LLM response content to stderr (per batch).",
    )
    args = ap.parse_args()

    evaluator = RedundancyEvaluator(
        coverage_csv_path=args.coverage_csv_path,
        forms_csv_path=args.forms_csv_path,
        field_report_csv_path=(args.field_report_csv_path.strip() or None),
        batch_size=args.batch_size,
        seed=args.seed,
        model=args.model,
        temperature=args.temperature,
        max_chars_per_field=args.max_chars_per_field,
        output_csv_path=args.output_csv_path,
        debug_print_llm_response=args.debug_print_llm_response,
    )

    max_pairs = args.max_pairs if args.max_pairs and args.max_pairs > 0 else None
    evaluator.compute_all(max_pairs=max_pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

