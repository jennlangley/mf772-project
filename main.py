from pathlib import Path
import re
import pandas as pd
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

CSV = Path("./data/2023Q1.csv")          # change per quarter
OUT = Path("./data/parquet_min/2023Q1")  # output dataset folder
OUT.mkdir(parents=True, exist_ok=True)

US_STATES = set("""
AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT
NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC
""".split())

ZB_CODES = {"01","02","03","06","09","15","16","96"}  # prepay/default/removal, etc.

def detect_columns(sample: pd.DataFrame):
    # sample has header=None, so columns are 0..N-1, values are strings
    cand = {i: sample.iloc[0, i] for i in sample.columns}

    # helper patterns
    def looks_loan_id(x):  return bool(re.fullmatch(r"\d{9,12}", str(x)))
    def looks_mmYYYY(x):   return bool(re.fullmatch(r"\d{6}", str(x))) and (1 <= int(str(x)[:2]) <= 12)
    def looks_rate(x):     return bool(re.fullmatch(r"\d{1,2}\.\d{3}", str(x)))  # e.g. 6.500
    def looks_money(x):    return bool(re.fullmatch(r"\d{1,3}(?:\d{3})*(?:\.\d{2})?", str(x))) and float(str(x).replace(',','')) > 1000
    def looks_term(x):     return str(x).isdigit() and int(x) in (180, 240, 360)
    def looks_state(x):    return str(x) in US_STATES
    def looks_ptype(x):    return str(x) in {"FRM"}  # dataset should be fixed-rate only
    def looks_zb(x):       return str(x) in ZB_CODES

    # try to find likely indexes by scanning first few rows for stability
    take = sample.head(200)

    def find_col(test_fn):
        score = {}
        for j in take.columns:
            vals = take[j].astype(str).tolist()
            hits = sum(test_fn(v) for v in vals)
            score[j] = hits
        # best column with at least a few hits
        j_best, hits_best = max(score.items(), key=lambda kv: kv[1])
        return j_best if hits_best >= max(5, len(take)//10) else None

    idx = {}
    idx['loan_id']                    = find_col(looks_loan_id)
    idx['monthly_reporting_period']   = find_col(looks_mmYYYY)
    idx['orig_rate']                  = find_col(looks_rate)
    idx['orig_upb']                   = find_col(looks_money)
    idx['orig_loan_term']             = find_col(looks_term)
    idx['state']                      = find_col(looks_state)
    idx['product_type']               = find_col(looks_ptype)
    idx['zero_balance_code']          = find_col(looks_zb)

    # current UPB is ambiguous vs original; pick a different money column than orig_upb
    money_cols = sorted([j for j in take.columns if take[j].astype(str).str.match(r"\d+(?:\.\d{2})?$").mean()>0.5])
    idx['current_upb'] = next((j for j in money_cols if j != idx['orig_upb']), None)

    # a second date-like column for zero_balance_effective_date
    date_like = sorted([j for j in take.columns if take[j].astype(str).str.fullmatch(r"\d{6}").mean()>0.5])
    idx['zero_balance_effective_date'] = next((j for j in date_like if j != idx['monthly_reporting_period']), None)

    return idx

# --- Pass 1: read a small chunk w/ no header to detect
sample = pd.read_csv(
    CSV, sep="|", header=None, nrows=2000,
    dtype=str, encoding="latin1", engine="python", on_bad_lines="skip"
)

col_idx = detect_columns(sample)
print("Detected column indices:", col_idx)

# Build names list for the selected columns
selected = {k: v for k, v in col_idx.items() if v is not None}
usecols = sorted(set(selected.values()))
name_map = {v: k for k, v in selected.items()}  # index -> logical name

# --- Pass 2: stream whole file keeping only needed columns, write Parquet
chunks = pd.read_csv(
    CSV, sep="|", header=None, usecols=usecols,
    chunksize=200_000, dtype=str, encoding="latin1",
    engine="python", on_bad_lines="skip"
)

for c in chunks:
    c = c.rename(columns=name_map)
    # basic cleaning and type prep
    if 'monthly_reporting_period' in c:
        # convert MMYYYY or YYYYMM to YYYY-MM-01
        s = c['monthly_reporting_period'].astype(str)
        mm = np.where(s.str.len()==6, s.str[:2], np.nan)
        yy = np.where(s.str.len()==6, s.str[2:], np.nan)
        c['mrp'] = pd.to_datetime(pd.Series(yy) + "-" + pd.Series(mm) + "-01", errors='coerce')
        c.drop(columns=['monthly_reporting_period'], inplace=True)
        c.rename(columns={'mrp':'monthly_reporting_period'}, inplace=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tbl = pa.Table.from_pandas(c, preserve_index=False)
    pq.write_to_dataset(tbl, root_path=str(OUT), compression="snappy")

print("Wrote minimal dataset to:", OUT)
