from pathlib import Path
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"          # 2023Q1.csv etc.
OUT_DIR = ROOT / "data" / "parquet_min"  # output
OUT_DIR.mkdir(parents=True, exist_ok=True)

QUARTERS = ["2023Q1", "2023Q2", "2023Q3", "2023Q4"]
SEP = "|"
CHUNK = 200_000

# 1) OFFICIAL FNMA LOAN PERFORMANCE HEADER (from their R code)
COL_NAMES = [
    "POOL_ID", "LOAN_ID", "ACT_PERIOD", "CHANNEL", "SELLER", "SERVICER",
    "MASTER_SERVICER", "ORIG_RATE", "CURR_RATE", "ORIG_UPB", "ISSUANCE_UPB",
    "CURRENT_UPB", "ORIG_TERM", "ORIG_DATE", "FIRST_PAY", "LOAN_AGE",
    "REM_MONTHS", "ADJ_REM_MONTHS", "MATR_DT", "OLTV", "OCLTV",
    "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG", "PURPOSE",
    "PROP", "NO_UNITS", "OCC_STAT", "STATE", "MSA", "ZIP", "MI_PCT",
    "PRODUCT", "PPMT_FLG", "IO", "FIRST_PAY_IO", "MNTHS_TO_AMTZ_IO",
    "DLQ_STATUS", "PMT_HISTORY", "MOD_FLAG", "MI_CANCEL_FLAG", "Zero_Bal_Code",
    "ZB_DTE", "LAST_UPB", "RPRCH_DTE", "CURR_SCHD_PRNCPL", "TOT_SCHD_PRNCPL",
    "UNSCHD_PRNCPL_CURR", "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE",
    "DISPOSITION_DATE", "FORECLOSURE_COSTS",
    "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
    "ASSET_RECOVERY_COSTS",
    "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
    "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", "NET_SALES_PROCEEDS",
    "CREDIT_ENHANCEMENT_PROCEEDS", "REPURCHASES_MAKE_WHOLE_PROCEEDS",
    "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB",
    "PRINCIPAL_FORGIVENESS_AMOUNT", "ORIGINAL_LIST_START_DATE",
    "ORIGINAL_LIST_PRICE", "CURRENT_LIST_START_DATE",
    "CURRENT_LIST_PRICE", "ISSUE_SCOREB", "ISSUE_SCOREC", "CURR_SCOREB",
    "CURR_SCOREC", "MI_TYPE", "SERV_IND",
    "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
    "CUMULATIVE_MODIFICATION_LOSS_AMOUNT",
    "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
    "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS",
    "HOMEREADY_PROGRAM_INDICATOR",
    "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT",
    "RELOCATION_MORTGAGE_INDICATOR",
    "ZERO_BALANCE_CODE_CHANGE_DATE", "LOAN_HOLDBACK_INDICATOR",
    "LOAN_HOLDBACK_EFFECTIVE_DATE", "DELINQUENT_ACCRUED_INTEREST",
    "PROPERTY_INSPECTION_WAIVER_INDICATOR",
    "HIGH_BALANCE_LOAN_INDICATOR", "ARM_5_YR_INDICATOR",
    "ARM_PRODUCT_TYPE", "MONTHS_UNTIL_FIRST_PAYMENT_RESET",
    "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET",
    "INTEREST_RATE_CHANGE_DATE", "PAYMENT_CHANGE_DATE", "ARM_INDEX",
    "ARM_CAP_STRUCTURE", "INITIAL_INTEREST_RATE_CAP",
    "PERIODIC_INTEREST_RATE_CAP", "LIFETIME_INTEREST_RATE_CAP", "MARGIN",
    "BALLOON_INDICATOR", "PLAN_NUMBER", "FORBEARANCE_INDICATOR",
    "HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR",
    "DEAL_NAME", "RE_PROCS_FLAG", "ADR_TYPE", "ADR_COUNT", "ADR_UPB",
    "PAYMENT_DEFERRAL_MOD_EVENT_FLAG", "INTEREST_BEARING_UPB",
]

print("Number of column names:", len(COL_NAMES))  # should be 110

# 2) Minimal modeling columns (subset of the above)
MIN_COLS = [
    "LOAN_ID",           # Loan Identifier
    "ACT_PERIOD",        # Monthly Reporting Period
    "CURRENT_UPB",       # Current Actual UPB
    "Zero_Bal_Code",     # reason loan ended
    "ZB_DTE",            # zero balance effective date

    "ORIG_RATE",         # Original Interest Rate
    "ORIG_UPB",          # Original UPB
    "ORIG_TERM",         # Original Loan Term

    "CSCORE_B",          # Credit score at origination (borrower)
    "OLTV",              # Original LTV
    "DTI",               # Debt-to-income

    "STATE",             # Property State
    "PURPOSE",           # Loan Purpose
    "CHANNEL",           # Channel
    "PRODUCT",           # Product type (FRM)
]

def convert_quarter_min(qname: str):
    csv_path = RAW_DIR / f"{qname}.csv"
    if not csv_path.exists():
        print("Missing CSV for", qname, "→ skipping")
        return

    dst_root = OUT_DIR / qname
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"Converting {csv_path} → {dst_root}")

    chunks = pd.read_csv(
        csv_path,
        sep=SEP,
        header=None,          # file has NO header row
        names=COL_NAMES,      # supply full 110 official names
        usecols=MIN_COLS,     # only keep minimal set
        dtype=str,
        engine="python",
        encoding="latin1",
        on_bad_lines="skip",
        chunksize=CHUNK,
    )

    for i, c in enumerate(chunks):
        tbl = pa.Table.from_pandas(c, preserve_index=False)
        pq.write_to_dataset(tbl, root_path=str(dst_root), compression="snappy")

    print("Done", qname)

for q in QUARTERS:
    convert_quarter_min(q)
