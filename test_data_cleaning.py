####################################################################################
# SAAM Project 2026 - Part I
# DATA CLEANING
# Group: North America + Europe, Scope 1
####################################################################################

from pathlib import Path
import numpy as np
import pandas as pd

####################################################################################
# 0. PATHS
####################################################################################

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"
CLEAN_DIR = BASE_DIR / "cleaned_data"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

####################################################################################
# 1. CONFIG
####################################################################################

REGIONS = ["AMER", "EUR"]

STATIC_FILE = "Static_2025.csv"
RI_FILE = "DS_RI_T_USD_M_2025.csv"
MV_FILE = "DS_MV_T_USD_M_2025.csv"
CO2_FILE = "DS_CO2_SCOPE_1_Y_2025.csv"
REV_FILE = "DS_REV_Y_2025.csv"

####################################################################################
# 2. HELPER FUNCTIONS
####################################################################################

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_id_cols(df: pd.DataFrame):
    upper_cols = {c.upper(): c for c in df.columns}
    name_col = upper_cols.get("NAME", None)
    isin_col = upper_cols.get("ISIN", None)
    return name_col, isin_col


def get_time_cols(df: pd.DataFrame):
    name_col, isin_col = get_id_cols(df)
    exclude = {c for c in [name_col, isin_col] if c is not None}
    return [c for c in df.columns if c not in exclude]


def sort_month_cols(cols):
    try:
        dt = pd.to_datetime(cols, errors="coerce")
        if dt.notna().all():
            return [c for _, c in sorted(zip(dt, cols))]
        return sorted(cols)
    except Exception:
        return sorted(cols)


def sort_year_cols(cols):
    def parse_year(x):
        s = str(x)
        try:
            return int(s[:4])
        except Exception:
            return 999999
    return sorted(cols, key=parse_year)


def drop_empty_timeseries_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows where all time columns are missing.
    """
    df = clean_headers(df)
    _, isin_col = get_id_cols(df)
    time_cols = get_time_cols(df)

    df = df.dropna(subset=[isin_col]).copy()
    df[time_cols] = df[time_cols].apply(pd.to_numeric, errors="coerce")
    df = df.loc[~df[time_cols].isna().all(axis=1)].copy()
    return df


def keep_isins(df: pd.DataFrame, valid_isins: set) -> pd.DataFrame:
    _, isin_col = get_id_cols(df)
    return df[df[isin_col].isin(valid_isins)].copy()


def annual_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing annual values using previous year only.
    This fills missing values in the middle/end, but not at the beginning.
    """
    out = df.copy()
    year_cols = sort_year_cols(get_time_cols(out))
    out[year_cols] = out[year_cols].apply(pd.to_numeric, errors="coerce")
    out[year_cols] = out[year_cols].ffill(axis=1)
    return out

""""
def detect_delisting(df: pd.DataFrame) -> pd.DataFrame:
    """
"""Detect delisted firms from NAME column and set prices to 0
    from delisting date onward."""
"""
    out = df.copy()
    name_col, _ = get_id_cols(out)
    if name_col is None:
        return out

    month_cols = sort_month_cols(get_time_cols(out))
    # parse month columns to datetime for comparison
    month_dts = pd.to_datetime(month_cols, errors="coerce")

    import re
    delist_pattern = re.compile(r"D(?:'|E)?LIST(?:ED)?\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE)

    for idx in out.index:
        name = str(out.at[idx, name_col])
        match = delist_pattern.search(name)
        if match:
            try:
                delist_dt = pd.to_datetime(match.group(1), dayfirst=True)
            except Exception:
                continue
            for c, cdt in zip(month_cols, month_dts):
                if pd.notna(cdt) and cdt >= delist_dt:
                    out.at[idx, c] = 0.0
    return out
"""
def detect_delisting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect delisted firms from NAME column and set prices to 0
    from delisting date onward.

    Two cases handled:
      1. Name contains a date: "COMPANY DELIST 15/09/2008" → price=0 from that date
      2. Name contains "DEAD" but no date → price=0 at first month after last valid price
    """
    import re

    out = df.copy()
    name_col, _ = get_id_cols(out)
    if name_col is None:
        return out

    month_cols = sort_month_cols(get_time_cols(out))
    month_dts = pd.to_datetime(month_cols, errors="coerce")

    delist_pattern = re.compile(r"D(?:'|E)?LIST(?:ED)?\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE)
    dead_pattern   = re.compile(r"\bDEAD\b", re.IGNORECASE)

    for idx in out.index:
        name = str(out.at[idx, name_col])

        # --- Case 1: explicit delisting date in name ---
        match = delist_pattern.search(name)
        if match:
            try:
                delist_dt = pd.to_datetime(match.group(1), dayfirst=True)
            except Exception:
                continue
            for c, cdt in zip(month_cols, month_dts):
                if pd.notna(cdt) and cdt >= delist_dt:
                    out.at[idx, c] = 0.0

        # --- Case 2: "DEAD" with no explicit date ---
        elif dead_pattern.search(name):
            vals = out.loc[idx, month_cols]
            last_valid = vals.last_valid_index()
            if last_valid is not None and last_valid != month_cols[-1]:
                pos = month_cols.index(last_valid) + 1
                out.at[idx, month_cols[pos]] = 0.0

    return out

def compute_monthly_returns(ri_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple monthly returns from cleaned RI prices.
    Handles:
      - standard return: R_t = P_t / P_{t-1} - 1
      - delisting (price == 0 with valid previous): R = -1.0
      - invalid gaps (NaN): R = NaN
    """
    df = ri_prices.copy()
    name_col, isin_col = get_id_cols(df)
    month_cols = sort_month_cols(get_time_cols(df))
    id_cols_list = [c for c in [name_col, isin_col] if c is not None]

    returns = df[id_cols_list].copy()

    for i, col in enumerate(month_cols):
        if i == 0:
            returns[col] = np.nan
            continue

        prev_col = month_cols[i - 1]
        prev = df[prev_col].values.astype(float)
        curr = df[col].values.astype(float)

        ret = np.full(len(df), np.nan)

        # standard return: both valid and > 0
        valid = (~np.isnan(prev)) & (prev > 0) & (~np.isnan(curr)) & (curr > 0)
        ret[valid] = curr[valid] / prev[valid] - 1.0

        # delisting: current == 0, previous was valid
        delist = (curr == 0.0) & (~np.isnan(prev)) & (prev > 0)
        ret[delist] = -1.0

        returns[col] = ret

    return returns


def flag_stale_prices(returns_df: pd.DataFrame, end_date: str, window_years: int = 10, threshold: float = 0.50):
    """
    Flag firms with a proportion of zero returns exceeding `threshold`
    over the `window_years`-year window ending at `end_date`.

    Parameters
    ----------
    returns_df : DataFrame with ISIN col + monthly return columns
    end_date   : end of estimation window, e.g. "2024-12-31"
    window_years : lookback in years (default 10)
    threshold  : max allowed fraction of zero-return months (default 0.50)

    Returns
    -------
    stale_isins : set of ISINs flagged as stale
    stale_info  : DataFrame with ISIN, zero_frac for diagnostics
    """
    _, isin_col = get_id_cols(returns_df)
    month_cols = sort_month_cols(get_time_cols(returns_df))

    end_dt = pd.Timestamp(end_date)
    start_dt = end_dt - pd.DateOffset(years=window_years)

    # select columns within [start_dt, end_dt]
    col_dates = pd.to_datetime(month_cols, errors="coerce")
    window_cols = [c for c, d in zip(month_cols, col_dates)
                   if pd.notna(d) and start_dt <= d <= end_dt]

    if not window_cols:
        return set(), pd.DataFrame()

    data = returns_df[window_cols].apply(pd.to_numeric, errors="coerce")

    # count valid (non-NaN) observations and zero returns per firm
    n_valid = data.notna().sum(axis=1)
    n_zero = (data == 0.0).sum(axis=1)

    zero_frac = n_zero / n_valid.replace(0, np.nan)

    stale_info = pd.DataFrame({
        "ISIN": returns_df[isin_col].values,
        "n_valid": n_valid.values,
        "n_zero": n_zero.values,
        "zero_frac": zero_frac.values,
    })

    stale_isins = set(stale_info.loc[stale_info["zero_frac"] > threshold, "ISIN"])
    return stale_isins, stale_info

####################################################################################
# 3. LOAD RAW FILES
####################################################################################

static = clean_headers(pd.read_csv(RAW_DIR / STATIC_FILE))
ri = clean_headers(pd.read_csv(RAW_DIR / RI_FILE))
mv = clean_headers(pd.read_csv(RAW_DIR / MV_FILE))
co2 = clean_headers(pd.read_csv(RAW_DIR / CO2_FILE))
rev = clean_headers(pd.read_csv(RAW_DIR / REV_FILE))

####################################################################################
# 4. CLEAN STATIC
####################################################################################

static_name_col, static_isin_col = get_id_cols(static)

if static_isin_col is None:
    raise ValueError("Static file must contain an ISIN column.")
if "Region" not in static.columns:
    raise ValueError("Static file must contain a 'Region' column.")

# remove missing ISIN
static = static.dropna(subset=[static_isin_col]).copy()

# keep only North America + Europe
static = static[static["Region"].isin(REGIONS)].copy()

# save intermediate static
static.to_csv(CLEAN_DIR / "cleaned_static.csv", index=False)

####################################################################################
# 5. CLEAN EACH DATASET
####################################################################################

ri = drop_empty_timeseries_rows(ri)
mv = drop_empty_timeseries_rows(mv)
co2 = drop_empty_timeseries_rows(co2)
rev = drop_empty_timeseries_rows(rev)

####################################################################################
# 6. FILTER EACH DATASET TO STATIC UNIVERSE
####################################################################################

region_isins = set(static[static_isin_col])

ri = keep_isins(ri, region_isins)
mv = keep_isins(mv, region_isins)
co2 = keep_isins(co2, region_isins)
rev = keep_isins(rev, region_isins)

####################################################################################
# 7. CLEAN RI MONTHLY
####################################################################################

ri_month_cols = sort_month_cols(get_time_cols(ri))
ri[ri_month_cols] = ri[ri_month_cols].apply(pd.to_numeric, errors="coerce")

# detect delisting BEFORE low-price filter (delisted → price = 0)
ri = detect_delisting(ri)

# treat low RI prices as missing (but keep 0.0 for delisted firms)
# only mask prices that are < 0.5 AND > 0 (preserve exact 0.0 for delisting)
for c in ri_month_cols:
    mask_low = (ri[c] < 0.5) & (ri[c] > 0)
    ri.loc[mask_low, c] = np.nan

# after low-price cleaning, remove rows with all missing price history
ri = ri.loc[~ri[ri_month_cols].isna().all(axis=1)].copy()

# compute monthly simple returns (handles delisting = -100%)
ri_ret = compute_monthly_returns(ri)

####################################################################################
# 7b. FLAG & EXCLUDE STALE-PRICE FIRMS
####################################################################################

STALE_THRESHOLD = 0.50
ESTIMATION_END = "2024-12-31"  # adjust per investment year Y

stale_isins, stale_info = flag_stale_prices(ri_ret, end_date=ESTIMATION_END,
                                             window_years=10,
                                             threshold=STALE_THRESHOLD)

print(f"Stale-price firms excluded: {len(stale_isins)} / {len(ri_ret)}")

# stale_isins will be removed in Section 11 when common_isins is built

####################################################################################
# 8. CLEAN MV MONTHLY
####################################################################################

mv_month_cols = sort_month_cols(get_time_cols(mv))
mv[mv_month_cols] = mv[mv_month_cols].apply(pd.to_numeric, errors="coerce")

####################################################################################
# 9. CLEAN CO2 SCOPE 1 YEARLY
####################################################################################

co2 = annual_forward_fill(co2)

####################################################################################
# 10. CLEAN REVENUE YEARLY
####################################################################################

rev = annual_forward_fill(rev)

####################################################################################
# 11. BUILD COMMON ISIN UNIVERSE ACROSS ALL CLEANED FILES
####################################################################################

_, ri_isin_col = get_id_cols(ri)
_, mv_isin_col = get_id_cols(mv)
_, co2_isin_col = get_id_cols(co2)
_, rev_isin_col = get_id_cols(rev)

common_isins = (
    set(static[static_isin_col])
    & set(ri[ri_isin_col])
    & set(mv[mv_isin_col])
    & set(co2[co2_isin_col])
    & set(rev[rev_isin_col])
)

# exclude stale-price firms
common_isins = common_isins - stale_isins

# keep only common universe everywhere
static = keep_isins(static, common_isins)
ri = keep_isins(ri, common_isins)
ri_ret = keep_isins(ri_ret, common_isins)
mv = keep_isins(mv, common_isins)
co2 = keep_isins(co2, common_isins)
rev = keep_isins(rev, common_isins)

####################################################################################
# 12. OPTIONAL: SORT ALL FILES BY ISIN
####################################################################################

static = static.sort_values(static_isin_col).reset_index(drop=True)
ri = ri.sort_values(ri_isin_col).reset_index(drop=True)
ri_ret = ri_ret.sort_values(ri_isin_col).reset_index(drop=True)
mv = mv.sort_values(mv_isin_col).reset_index(drop=True)
co2 = co2.sort_values(co2_isin_col).reset_index(drop=True)
rev = rev.sort_values(rev_isin_col).reset_index(drop=True)

####################################################################################
# 13. SAVE CLEANED OUTPUTS
####################################################################################

static.to_csv(CLEAN_DIR / "cleaned_static.csv", index=False)
ri.to_csv(CLEAN_DIR / "cleaned_RI_monthly_prices.csv", index=False)
ri_ret.to_csv(CLEAN_DIR / "cleaned_RI_monthly_returns.csv", index=False)
mv.to_csv(CLEAN_DIR / "cleaned_MV_monthly.csv", index=False)
co2.to_csv(CLEAN_DIR / "cleaned_CO2_scope1_yearly.csv", index=False)
rev.to_csv(CLEAN_DIR / "cleaned_REV_yearly.csv", index=False)

####################################################################################
# 14. PRINT SUMMARY
####################################################################################

print("Data cleaning complete.")
print(f"Common universe size: {len(common_isins)} firms")
print("Files saved in:", CLEAN_DIR)
print("- cleaned_static.csv")
print("- cleaned_RI_monthly_prices.csv")
print("- cleaned_RI_monthly_returns.csv")
print("- cleaned_MV_monthly.csv")
print("- cleaned_CO2_scope1_yearly.csv")
print("- cleaned_REV_yearly.csv")