"""Print the shape and column names of every Beat AML raw file.

Run this FIRST, before writing any analysis code. The published column names
are not documented anywhere online, so this is how you learn the schema.
Paste its output into the team chat -- everyone then codes against the same names.

    python scripts/inspect_columns.py > reports/schema.txt
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")

# Columns likely to be label-bearing; we print their value counts when present.
LABEL_HINTS = [
    "eln", "risk", "vital", "status", "response", "timepoint", "specimen",
    "diagnosis", "sex", "gender", "race", "wave", "cohort", "tissue",
    "consensus", "inhibitor", "drug", "family", "treatment",
]


def preview_table(path: Path, sep: str) -> None:
    try:
        head = pd.read_csv(path, sep=sep, nrows=200, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! could not parse: {exc}")
        return

    # Full row count without loading the whole file into memory.
    with path.open("rb") as fh:
        n_rows = sum(1 for _ in fh) - 1

    print(f"  rows: {n_rows:,}   cols: {head.shape[1]:,}")
    print(f"  columns ({min(60, head.shape[1])} shown):")
    for col in list(head.columns)[:60]:
        print(f"    - {col!r}  [{head[col].dtype}]")
    if head.shape[1] > 60:
        print(f"    ... and {head.shape[1] - 60} more (likely sample IDs)")

    for col in head.columns[:60]:
        if any(h in str(col).lower() for h in LABEL_HINTS):
            vals = head[col].dropna().unique()[:12]
            print(f"  values of {col!r} (first 200 rows): {list(vals)}")


def preview_excel(path: Path) -> None:
    try:
        sheets = pd.read_excel(path, sheet_name=None, nrows=200)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! could not parse: {exc}")
        return

    for name, df in sheets.items():
        print(f"  sheet {name!r}: {df.shape[0]:,}+ rows x {df.shape[1]} cols")
        for col in df.columns:
            print(f"    - {col!r}  [{df[col].dtype}]")
        for col in df.columns:
            if any(h in str(col).lower() for h in LABEL_HINTS):
                vals = df[col].dropna().unique()[:12]
                print(f"    values of {col!r}: {list(vals)}")
        print()


def main() -> None:
    if not RAW.exists():
        raise SystemExit("data/raw/ not found -- run `bash scripts/get_data.sh` first.")

    files = sorted(p for p in RAW.rglob("*") if p.is_file())
    if not files:
        raise SystemExit("data/raw/ is empty -- run `bash scripts/get_data.sh` first.")

    for path in files:
        if path.name == "MANIFEST.sha256":
            continue
        size_mb = path.stat().st_size / 1e6
        print("=" * 78)
        print(f"{path.relative_to(RAW)}   ({size_mb:.1f} MB)")
        print("=" * 78)

        if path.suffix in {".txt", ".tsv", ".csv"}:
            preview_table(path, sep="\t" if path.suffix != ".csv" else ",")
        elif path.suffix in {".xlsx", ".xls"}:
            preview_excel(path)
        else:
            print("  (not a table -- skipped)")
        print()


if __name__ == "__main__":
    main()
