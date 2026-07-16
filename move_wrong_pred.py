"""
move_wrong_predictions.py (v2)
================================
Standalone CLI script for flagging misclassified documents in place, plus reverting
that later if needed.

CHANGED FROM v1: instead of relocating a wrong prediction out of its true
Category/Subcategory into a folder named after the prediction, it now NESTS a
prediction-named folder inside the file's existing true Category/Subcategory. The
dataset's ground-truth organization stays intact; you just get a visible "these get
predicted as X" pocket inside each true class folder.

    BEFORE (source):  ./Dataset/<category>/<subcategory>/<filename>
    AFTER  (dest):    ./Dataset/<category>/<subcategory>/<predicted_class>/<filename>

<category>/<subcategory> is derived from the TRUE label (ground_truth, or triage_gt /
router_gt depending on --stage) using the same prefix rules as before (medical_ ->
medical, financial_ -> financial, id_ -> identification; eform/unrelated_document are
flat, no subcategory nesting).

Usage:
    # Dry run first -- always the default, moves nothing.
    python move_wrong_predictions.py results.csv --base-dir ./Dataset

    # Only handle wrong predictions within specific true classes:
    python move_wrong_predictions.py results.csv --base-dir ./Dataset --classes financial_receipt id_passport

    # Once it looks right:
    python move_wrong_predictions.py results.csv --base-dir ./Dataset --classes financial_receipt id_passport --execute

    # Revert every move currently logged (dry run first, as always):
    python move_wrong_predictions.py --revert
    python move_wrong_predictions.py --revert --execute

    # Only revert the 10 most recently logged moves:
    python move_wrong_predictions.py --revert --last 10 --execute

Every successful move is appended to move_log.csv. --revert reads that file, moves
files back to their original source_path, and removes/archives the reverted entries so
the same batch can't accidentally be reverted twice.
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
    """Maps a label to (category, subcategory). eform/unrelated_document are flat --
    no subcategory nesting, since they have no further split downstream."""
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


def resolve_category_dir(base_dir: str, label: str) -> str:
    category, subcategory = get_category_subcategory(label)
    if subcategory is None:
        return os.path.join(base_dir, category)
    return os.path.join(base_dir, category, subcategory)


def move_one(source_path: str, dest_path: str, overwrite: bool, dry_run: bool):
    """Returns (status, message). status is one of:
    'moved', 'would_move', 'missing_source', 'exists_skip', 'error'."""
    if not os.path.isfile(source_path):
        return "missing_source", f"Source not found: {source_path}"

    if os.path.exists(dest_path) and not overwrite:
        return "exists_skip", f"Already exists at destination (use --overwrite to replace): {dest_path}"

    if dry_run:
        return "would_move", dest_path

    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
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


def do_move(args):
    if not args.csv_path:
        sys.exit("csv_path is required unless --revert is given.")

    df = pd.read_csv(args.csv_path)

    path_col = "filepath" if "filepath" in df.columns else "filename"
    if path_col not in df.columns:
        sys.exit("CSV must have a 'filepath' or 'filename' column (used for the filename only -- the folder is derived from the true label, not this column's directory).")

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

    if args.classes:
        before = len(wrong)
        wrong = wrong[wrong[true_col].astype(str).isin(args.classes)]
        print(f"--classes filter: {before} -> {len(wrong)} row(s) (kept true class in {args.classes}).")

    print(f"Found {len(wrong)} wrongly-predicted document(s) to flag (stage: {args.stage}).")
    if len(wrong) == 0:
        return

    wrong["_filename"] = wrong[path_col].astype(str).apply(os.path.basename)
    wrong["_category_dir"] = wrong[true_col].apply(lambda v: resolve_category_dir(args.base_dir, v))
    wrong["_source_path"] = wrong.apply(lambda r: os.path.join(r["_category_dir"], r["_filename"]), axis=1)
    wrong["_dest_path"] = wrong.apply(lambda r: os.path.join(r["_category_dir"], str(r[pred_col]), r["_filename"]), axis=1)

    print("\nPlanned moves by predicted-class subfolder:")
    dest_dirs = wrong["_dest_path"].apply(os.path.dirname)
    for dest_dir, count in dest_dirs.value_counts().sort_index().items():
        print(f"  {dest_dir}: {count} file(s)")

    if not args.execute:
        print("\nDRY RUN -- no files were moved or created. Re-run with --execute once this looks right.")

    counts = {"moved": 0, "would_move": 0, "missing_source": 0, "exists_skip": 0, "error": 0}
    for _, row in wrong.iterrows():
        status, message = move_one(row["_source_path"], row["_dest_path"], args.overwrite, dry_run=not args.execute)
        counts[status] += 1
        if status == "moved":
            log_move(row["_source_path"], message)
        if status in ("missing_source", "exists_skip", "error"):
            print(f"  [{status}] {row['_source_path']} -> {message}")

    print("\nSummary:")
    for k, v in counts.items():
        if v:
            print(f"  {k}: {v}")
    if args.execute and counts["moved"]:
        print(f"\n{counts['moved']} file(s) moved. Logged to {MOVE_LOG_FILENAME}.")


def do_revert(args):
    if not os.path.isfile(MOVE_LOG_FILENAME):
        print(f"No {MOVE_LOG_FILENAME} found -- nothing to revert.")
        return

    log_df = pd.read_csv(MOVE_LOG_FILENAME)
    if len(log_df) == 0:
        print(f"{MOVE_LOG_FILENAME} is empty -- nothing to revert.")
        return

    to_revert = log_df.tail(args.last) if args.last else log_df
    print(f"Reverting {len(to_revert)} move(s) (out of {len(log_df)} logged total).")
    if not args.execute:
        print("DRY RUN -- no files will be moved or logs changed. Re-run with --execute once this looks right.\n")

    counts = {"reverted": 0, "would_revert": 0, "missing_dest": 0, "exists_skip": 0, "error": 0}
    reverted_indices = []

    for idx, row in to_revert.iterrows():
        current_path = row["dest_path"]   # where the file currently is
        original_path = row["source_path"]  # where it should go back to

        if not os.path.isfile(current_path):
            counts["missing_dest"] += 1
            print(f"  [missing_dest] {current_path} not found -- already moved elsewhere or already reverted?")
            continue
        if os.path.exists(original_path) and not args.overwrite:
            counts["exists_skip"] += 1
            print(f"  [exists_skip] {original_path} already exists (use --overwrite to replace)")
            continue

        if not args.execute:
            counts["would_revert"] += 1
            continue

        try:
            os.makedirs(os.path.dirname(original_path), exist_ok=True)
            shutil.move(current_path, original_path)
            counts["reverted"] += 1
            reverted_indices.append(idx)
        except Exception as e:
            counts["error"] += 1
            print(f"  [error] {current_path} -> {original_path}: {e}")

    if args.execute and reverted_indices:
        remaining = log_df.drop(index=reverted_indices)
        remaining.to_csv(MOVE_LOG_FILENAME, index=False)
        archive_path = f"move_log.reverted.{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_df.loc[reverted_indices].to_csv(archive_path, index=False)
        print(f"\nRemoved {len(reverted_indices)} reverted entries from {MOVE_LOG_FILENAME}; archived them to {archive_path} (so this batch can't be double-reverted).")

    print("\nSummary:")
    for k, v in counts.items():
        if v:
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Flag wrongly-predicted documents by nesting a predicted-class folder inside their true Category/Subcategory. Supports reverting later."
    )
    parser.add_argument("csv_path", nargs="?", default=None,
                         help="Results CSV (required unless --revert). Needs filepath/filename, ground_truth, final_subcategory (and optionally triage_gt/triage_decision, router_gt/router_decision).")
    parser.add_argument("--base-dir", default="./Dataset", help="Base dataset folder (default: ./Dataset).")
    parser.add_argument("--execute", action="store_true", help="Actually move/revert files. Without this flag, runs a dry run only.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files at the destination instead of skipping them.")
    parser.add_argument("--stage", choices=["leaf", "triage", "router"], default="leaf",
                         help="Which stage defines 'wrong' (default: leaf = ground_truth vs final_subcategory).")
    parser.add_argument("--classes", nargs="+", default=None,
                         help="Only flag rows whose TRUE class (per --stage) is in this list, e.g. --classes financial_receipt id_passport.")
    parser.add_argument("--revert", action="store_true", help="Revert previously logged moves instead of making new ones.")
    parser.add_argument("--last", type=int, default=None, help="With --revert, only revert the N most recently logged moves (default: revert all logged).")
    args = parser.parse_args()

    if args.revert:
        do_revert(args)
    else:
        do_move(args)


if __name__ == "__main__":
    main()