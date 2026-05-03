"""Inspect Parquet dataset structure and per-column statistics.

Typical usage:
    python analyze_parquet.py --data demo_1000.parquet
    python analyze_parquet.py --data ./data_dir --schema ./data_dir/schema.json

Outputs:
- dataset-level summary (files / row groups / rows / schema)
- per-column logical type, null ratio, and list-length stats for list columns
- optional schema cross-check against the training pipeline's schema.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


@dataclass
class ColumnSummary:
    name: str
    logical_type: str
    rows: int
    null_count: int
    null_ratio: float
    sample_values: List[str]
    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    length_min: Optional[int] = None
    length_mean: Optional[float] = None
    length_p50: Optional[float] = None
    length_p95: Optional[float] = None
    length_max: Optional[int] = None
    empty_count: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect parquet dataset columns and list-length distributions")
    parser.add_argument("--data", required=True, help="Parquet file path or directory containing *.parquet")
    parser.add_argument("--schema", default=None, help="Optional schema.json used by the training pipeline")
    parser.add_argument("--sample-rows", type=int, default=3, help="How many non-null sample values to show per column")
    parser.add_argument("--list-scan-limit", type=int, default=200000, help="Max non-null list rows scanned per column for length stats")
    parser.add_argument("--output-json", default=None, help="Optional path to save the full report as JSON")
    return parser.parse_args()


def resolve_parquet_files(path_str: str) -> List[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {path}")
        return files
    raise FileNotFoundError(f"Path not found: {path}")


def format_arrow_type(tp: pa.DataType) -> str:
    if pa.types.is_list(tp):
        return f"list<{format_arrow_type(tp.value_type)}>"
    if pa.types.is_large_list(tp):
        return f"large_list<{format_arrow_type(tp.value_type)}>"
    return str(tp)


def sample_non_null_values(arr: pa.Array, limit: int) -> List[str]:
    values: List[str] = []
    for i in range(len(arr)):
        value = arr[i].as_py()
        if value is None:
            continue
        values.append(repr(value))
        if len(values) >= limit:
            break
    return values


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def summarize_scalar_column(name: str, arr: pa.Array, sample_rows: int) -> ColumnSummary:
    logical_type = format_arrow_type(arr.type)
    rows = len(arr)
    null_count = arr.null_count
    null_ratio = null_count / rows if rows else 0.0
    sample_values = sample_non_null_values(arr, sample_rows)

    numeric_min: Optional[float] = None
    numeric_max: Optional[float] = None
    if pa.types.is_integer(arr.type) or pa.types.is_floating(arr.type) or pa.types.is_timestamp(arr.type):
        try:
            numeric_min = _to_float(pc.min(arr).as_py())
            numeric_max = _to_float(pc.max(arr).as_py())
        except (pa.ArrowInvalid, NotImplementedError):
            numeric_min = None
            numeric_max = None

    return ColumnSummary(
        name=name,
        logical_type=logical_type,
        rows=rows,
        null_count=null_count,
        null_ratio=null_ratio,
        sample_values=sample_values,
        numeric_min=numeric_min,
        numeric_max=numeric_max,
    )


def summarize_list_column(name: str, arr: pa.Array, sample_rows: int, list_scan_limit: int) -> ColumnSummary:
    logical_type = format_arrow_type(arr.type)
    rows = len(arr)
    null_count = arr.null_count
    null_ratio = null_count / rows if rows else 0.0
    sample_values = sample_non_null_values(arr, sample_rows)

    non_null = arr.drop_null()
    if len(non_null) == 0:
        return ColumnSummary(
            name=name,
            logical_type=logical_type,
            rows=rows,
            null_count=null_count,
            null_ratio=null_ratio,
            sample_values=sample_values,
            length_min=0,
            length_mean=0.0,
            length_p50=0.0,
            length_p95=0.0,
            length_max=0,
            empty_count=0,
        )

    if len(non_null) > list_scan_limit:
        non_null = non_null.slice(0, list_scan_limit)
    lengths_arr = pc.list_value_length(non_null)
    lengths = np.asarray(lengths_arr.to_numpy(zero_copy_only=False), dtype=np.int64)
    empty_count = int((lengths == 0).sum())

    return ColumnSummary(
        name=name,
        logical_type=logical_type,
        rows=rows,
        null_count=null_count,
        null_ratio=null_ratio,
        sample_values=sample_values,
        length_min=int(lengths.min()) if len(lengths) else 0,
        length_mean=float(lengths.mean()) if len(lengths) else 0.0,
        length_p50=float(np.percentile(lengths, 50)) if len(lengths) else 0.0,
        length_p95=float(np.percentile(lengths, 95)) if len(lengths) else 0.0,
        length_max=int(lengths.max()) if len(lengths) else 0,
        empty_count=empty_count,
    )


def summarize_table(table: pa.Table, sample_rows: int, list_scan_limit: int) -> List[ColumnSummary]:
    summaries: List[ColumnSummary] = []
    for name in table.column_names:
        arr = table[name].combine_chunks()
        if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
            summaries.append(summarize_list_column(name, arr, sample_rows, list_scan_limit))
        else:
            summaries.append(summarize_scalar_column(name, arr, sample_rows))
    return summaries


def load_schema_groups(schema_path: str) -> Dict[str, List[str]]:
    with open(schema_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    groups: Dict[str, List[str]] = {
        "user_int": [f"user_int_feats_{fid}" for fid, _, _ in raw.get("user_int", [])],
        "item_int": [f"item_int_feats_{fid}" for fid, _, _ in raw.get("item_int", [])],
        "user_dense": [f"user_dense_feats_{fid}" for fid, _ in raw.get("user_dense", [])],
        "meta": ["user_id", "item_id", "label_type", "label_time", "timestamp"],
    }
    seq_cols: List[str] = []
    for domain_cfg in raw.get("seq", {}).values():
        prefix = domain_cfg["prefix"]
        seq_cols.extend(f"{prefix}_{fid}" for fid, _ in domain_cfg["features"])
    groups["seq"] = sorted(seq_cols)
    return groups


def build_schema_check(table_columns: Sequence[str], schema_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not schema_path:
        return None
    groups = load_schema_groups(schema_path)
    table_set = set(table_columns)
    group_counts = {k: len(v) for k, v in groups.items()}
    missing = {k: [c for c in v if c not in table_set] for k, v in groups.items()}
    unexpected = sorted(c for c in table_columns if c not in set(sum(groups.values(), [])))
    return {
        "schema_path": schema_path,
        "group_counts": group_counts,
        "missing_columns": {k: v for k, v in missing.items() if v},
        "unexpected_columns": unexpected,
    }


def report_dataset(files: Sequence[Path]) -> Dict[str, Any]:
    total_rows = 0
    total_row_groups = 0
    schemas: List[List[Tuple[str, str]]] = []
    for file in files:
        pf = pq.ParquetFile(file)
        total_rows += pf.metadata.num_rows
        total_row_groups += pf.metadata.num_row_groups
        schemas.append([(field.name, format_arrow_type(field.type)) for field in pf.schema_arrow])
    schema_consistent = all(s == schemas[0] for s in schemas[1:]) if schemas else True
    return {
        "files": [str(f) for f in files],
        "num_files": len(files),
        "total_rows": total_rows,
        "total_row_groups": total_row_groups,
        "schema_consistent": schema_consistent,
    }


def print_human_report(dataset_info: Dict[str, Any], summaries: Sequence[ColumnSummary], schema_check: Optional[Dict[str, Any]]) -> None:
    print("=== Dataset Summary ===")
    print(f"files: {dataset_info['num_files']}")
    print(f"rows: {dataset_info['total_rows']}")
    print(f"row_groups: {dataset_info['total_row_groups']}")
    print(f"schema_consistent: {dataset_info['schema_consistent']}")
    print(f"columns: {len(summaries)}")

    if schema_check:
        print("\n=== Schema Cross Check ===")
        print(f"schema_path: {schema_check['schema_path']}")
        print(f"group_counts: {schema_check['group_counts']}")
        if schema_check["missing_columns"]:
            print(f"missing_columns: {schema_check['missing_columns']}")
        else:
            print("missing_columns: {}")
        if schema_check["unexpected_columns"]:
            print(f"unexpected_columns: {schema_check['unexpected_columns']}")
        else:
            print("unexpected_columns: []")

    print("\n=== Column Summary ===")
    for col in summaries:
        parts = [
            f"{col.name}",
            f"type={col.logical_type}",
            f"null={col.null_count}/{col.rows} ({col.null_ratio:.2%})",
        ]
        if col.numeric_min is not None or col.numeric_max is not None:
            parts.append(f"range=[{col.numeric_min}, {col.numeric_max}]")
        if col.length_min is not None:
            parts.append(
                "len="
                f"min{col.length_min}/mean{col.length_mean:.2f}/p50{col.length_p50:.1f}/"
                f"p95{col.length_p95:.1f}/max{col.length_max}"
            )
            parts.append(f"empty={col.empty_count}")
        if col.sample_values:
            parts.append(f"samples={col.sample_values}")
        print(" | ".join(parts))


def main() -> None:
    args = parse_args()
    files = resolve_parquet_files(args.data)
    dataset_info = report_dataset(files)
    table = pq.read_table([str(f) for f in files])
    summaries = summarize_table(table, sample_rows=args.sample_rows, list_scan_limit=args.list_scan_limit)
    schema_check = build_schema_check(table.column_names, args.schema)

    print_human_report(dataset_info, summaries, schema_check)

    if args.output_json:
        payload = {
            "dataset": dataset_info,
            "schema_check": schema_check,
            "columns": [col.__dict__ for col in summaries],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nJSON report saved to {args.output_json}")


if __name__ == "__main__":
    main()
