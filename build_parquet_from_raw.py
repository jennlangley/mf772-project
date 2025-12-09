from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PARQ_DIR = ROOT / "data" / "parquet_min"

PARQ_DIR.mkdir(parents=True, exist_ok=True)

# Build whatever quarters you need; you can extend this list.
QUARTERS = ["2022Q1", "2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2", "2023Q3", "2023Q4"]

# Mapping from pandas column index -> target column name in parquet
# Column index = (Field Position from glossary) - 1
COL_INDEX_TO_NAME = {
    1:  "LOAN_ID",        # Field  2: Loan Identifier
    2:  "ACT_PERIOD",     # Field  3: Monthly Reporting Period (MMYYYY)
    3:  "CHANNEL",        # Field  4: Channel
    7:  "ORIG_RATE",      # Field  8: Original Interest Rate
    9:  "ORIG_UPB",       # Field 10: Original UPB
    11: "CURRENT_UPB",    # Field 12: Current Actual UPB
    12: "ORIG_TERM",      # Field 13: Original Loan Term (months)
    19: "OLTV",           # Field 20: Original LTV
    22: "DTI",            # Field 23: Debt-To-Income
    23: "CSCORE_B",       # Field 24: Borrower Credit Score
    26: "PURPOSE",        # Field 27: Loan Purpose
    30: "STATE",          # Field 31: Property State
    34: "PRODUCT",        # Field 35: Amortization Type (FRM/ARM) -> used as PRODUCT
    43: "Zero_Bal_Code",  # Field 44: Zero Balance Code
    44: "ZB_DTE",         # Field 45: Zero Balance Effective Date (MMYYYY)
}

USECOLS = sorted(COL_INDEX_TO_NAME.keys())

for Q in QUARTERS:
    csv_path = RAW_DIR / f"{Q}.csv"
    if not csv_path.exists():
        print(f"⚠ {csv_path} not found, skipping.")
        continue

    out_path = PARQ_DIR / f"{Q}.parquet"
    print(f"\n=== Building {out_path} from {csv_path} ===")

    # Chunked read to avoid memory blowups
    chunks = pd.read_csv(
        csv_path,
        sep="|",
        header=None,
        dtype=str,        # keep IDs, dates, and codes exactly
        usecols=USECOLS,
        chunksize=200_000,
        engine="python",
        on_bad_lines="skip",
    )

    writer = None
    total_rows = 0

    for i, chunk in enumerate(chunks, start=1):
        # Rename numeric indices -> meaningful column names
        chunk = chunk.rename(columns=COL_INDEX_TO_NAME)

        # Optionally, add quarter label
        chunk["QUARTER"] = Q

        total_rows += len(chunk)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        print(f"  wrote chunk {i}, cumulative rows: {total_rows:,}")

    if writer is not None:
        writer.close()
        print(f"✓ Finished {Q}, parquet at {out_path}")
    else:
        print(f"⚠ No data written for {Q}")
