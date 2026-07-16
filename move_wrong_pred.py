"""
move_wrong_predictions.py
==========================
Standalone script (no Streamlit -- run from the command line) for batch-moving
misclassified documents into Category/Subcategory folders named after the model's
PREDICTED class. Browsing each predicted-class folder afterward lets you see exactly
what's confusing the model for that class.

Folder structure created under --base-dir:
    <base_dir>/<category>/<subcategory>/<filename>   e.g. financial/financial_receipt/doc123.jpg
    <base_dir>/eform/<filename>                       eform and unrelated_document are flat --
    <base_dir>/unrelated_document/<filename>          they have no further subcategory split.

category is derived from the predicted label's prefix (medical_ / financial_ / id_),
so it automatically matches whatever leaves exist in cascade_pipeline_v11.py without
needing a hardcoded list here.

Usage:
    # ALWAYS dry-run first -- prints exactly what would happen, moves nothing.
    python move_wrong_predictions.py results.csv --base-dir ./Dataset

    # Once the dry-run output looks right, actually move the files:
    python move_wrong_predictions.py results.csv --base-dir ./Dataset --execute

    # Check triage-level or router-level mistakes instead of final-leaf mistakes:
    python move_wrong_predictions.py results.csv --stage triage --execute
    python move_wrong_predictions.py results.csv --stage router --execute

Every successful move is appended to move_log.csv -- same file/format the Streamlit
app's move log used, so both tools share one audit trail.
"""

import argparse
import os
import shutil
import csv
import sys
from datetime import datetime

import pandas as pd

MOVE_LOG_FILENAME = "move_log.csv"


def get_category_subcategory(label: str):
    """Maps a predicted label to (category, subcategory). eform/unrelated_document are
    flat -- no subcategory nesting, since they have no further split downstream."""
    if not isinstance(label, str):
        return "unknown", str(label)
    if label.startswith("medical_"):
        return "medical", label
    if label.startswith("financial_"):
        return "financial", label
    if label.startswith("id_"):
        return "identification", label
    if label in ("eform", "unrelated_document"):
        return label, None
    return "unknown", label


def resolve_dest_dir(base_dir: str, label: str) -> str:
    category, subcategory = get_category_subcategory(label)
    if subcategory is None:
        return os.path.join(base_dir, category)
    return os.path.join(base_dir, category, subcategory)


def move_one(source_path: str, dest_dir: str, overwrite: bool, dry_run: bool):
    """Returns (status, message). status is one of:
    'moved', 'would_move', 'missing_source', 'exists_skip', 'error'."""
    if not os.path.isfile(source_path):
        return "missing_source", f"Source not found: {source_path}"

    filename = os.path.basename(source_path)
    dest_path = os.path.join(dest_dir, filename)

    if os.path.exists(dest_path) and not overwrite:
        return "exists_skip", f"Already exists at destination (use --overwrite to replace): {dest_path}"

    if dry_run:
        return "would_move", dest_path

    try:
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(source_path, dest_path)
    except Exception as e:
        return "error", f"Move failed: {e}"

    return "moved", dest_path


def log_move(source_path: str, dest_path: str):
    log_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "filename": os.path.basename(source_path),
        "source_path": source_path,
        "dest_path": dest_path,
    }
    log_exists = os.path.isfile(MOVE_LOG_FILENAME)
    with open(MOVE_LOG_FILENAME, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_row.keys()))
        if not log_exists:
            writer.writeheader()
        writer.writerow(log_row)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-move wrongly-predicted documents into Category/Subcategory folders named after the predicted class."
    )
    parser.add_argument("csv_path", help="Results CSV with filepath/filename, ground_truth, final_subcategory (and optionally triage_gt/triage_decision, router_gt/router_decision).")
    parser.add_argument("--base-dir", default="./Dataset", help="Base folder to move files into (default: ./Dataset). Created automatically.")
    parser.add_argument("--execute", action="store_true", help="Actually move files. Without this flag, runs a dry run only -- nothing is moved or deleted.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files at the destination instead of skipping them.")
    parser.add_argument("--stage", choices=["leaf", "triage", "router"], default="leaf",
                         help="Which stage defines 'wrong'. leaf = ground_truth vs final_subcategory (default, most granular). "
                              "triage = triage_gt vs triage_decision. router = router_gt vs router_decision.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    path_col = "filepath" if "filepath" in df.columns else "filename"
    if path_col not in df.columns:
        sys.exit("CSV must have a 'filepath' or 'filename' column.")

    stage_cols = {
        "leaf": ("ground_truth", "final_subcategory"),
        "triage": ("triage_gt", "triage_decision"),
        "router": ("router_gt", "router_decision"),
    }
    true_col, pred_col = stage_cols[args.stage]
    for col in (true_col, pred_col):
        if col not in df.columns:
            sys.exit(f"CSV is missing required column for --stage {args.stage}: '{col}'")

    wrong = df[
        df[true_col].notna() & df[pred_col].notna()
        & (df[true_col].astype(str) != df[pred_col].astype(str))
    ].copy()

    print(f"Found {len(wrong)} wrongly-predicted document(s) out of {len(df)} total (stage: {args.stage}).")
    if len(wrong) == 0:
        return

    wrong["_dest_dir"] = wrong[pred_col].apply(lambda v: resolve_dest_dir(args.base_dir, v))

    print("\nPlanned moves by destination folder:")
    for dest_dir, count in wrong["_dest_dir"].value_counts().sort_index().items():
        print(f"  {dest_dir}: {count} file(s)")

    if not args.execute:
        print("\nDRY RUN -- no files were moved or created. Re-run with --execute once this looks right.")

    counts = {"moved": 0, "would_move": 0, "missing_source": 0, "exists_skip": 0, "error": 0}
    for _, row in wrong.iterrows():
        source_path = str(row[path_col])
        dest_dir = row["_dest_dir"]
        status, message = move_one(source_path, dest_dir, args.overwrite, dry_run=not args.execute)
        counts[status] += 1
        if status == "moved":
            log_move(source_path, message)
        if status in ("missing_source", "exists_skip", "error"):
            print(f"  [{status}] {source_path} -> {message}")

    print("\nSummary:")
    for k, v in counts.items():
        if v:
            print(f"  {k}: {v}")
    if args.execute and counts["moved"]:
        print(f"\n{counts['moved']} file(s) moved. Logged to {MOVE_LOG_FILENAME}.")


if __name__ == "__main__":
    main()