####################################################################################
# SAAM Project 2026 - Part I
# DATA CLEANING / PRE-PROCESSING
# Group: North America + Europe, Scope 1
####################################################################################

from pathlib import Path
import re
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

STALE_THRESHOLD = 0.50
MIN_VALID_MONTHS = 36           # 3 years of monthly returns
WINDOW_YEARS = 10
START_SCREEN_YEAR = 2013
END_SCREEN_YEAR = 2024

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
    Drop rows where all time columns are missing.
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
    Fills missing values in the middle/end, but not at the beginning.
    """
    out = df.copy()
    year_cols = sort_year_cols(get_time_cols(out))
    out[year_cols] = out[year_cols].apply(pd.to_numeric, errors="coerce")
    out[year_cols] = out[year_cols].ffill(axis=1)
    return out


def detect_delisting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect delisted firms from NAME column and set RI prices to 0
    from delisting date onward where possible.

    Case 1:
      explicit date in name, e.g. "... DELIST 15/09/2008"
    Case 2:
      'DEAD' in name with no date -> set first month after last valid price to 0
    """
    out = df.copy()
    name_col, _ = get_id_cols(out)
    if name_col is None:
        return out

    month_cols = sort_month_cols(get_time_cols(out))
    month_dts = pd.to_datetime(month_cols, errors="coerce")

    delist_pattern = re.compile(r"D(?:'|E)?LIST(?:ED)?\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE)
    dead_pattern = re.compile(r"\bDEAD\b", re.IGNORECASE)

    for idx in out.index:
        name = str(out.at[idx, name_col])

        # explicit delisting date
        match = delist_pattern.search(name)
        if match:
            try:
                delist_dt = pd.to_datetime(match.group(1), dayfirst=True)
            except Exception:
                continue

            for c, cdt in zip(month_cols, month_dts):
                if pd.notna(cdt) and cdt >= delist_dt:
                    out.at[idx, c] = 0.0

        # DEAD without date
        elif dead_pattern.search(name):
            vals = out.loc[idx, month_cols]
            last_valid = vals.last_valid_index()
            if last_valid is not None:
                last_pos = month_cols.index(last_valid)
                if last_pos + 1 < len(month_cols):
                    out.at[idx, month_cols[last_pos + 1]] = 0.0

    return out


def compute_monthly_returns(ri_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns from cleaned RI prices.

    Rules:
    - standard return if P_t and P_{t-1} are both > 0
    - if P_t == 0 and P_{t-1} > 0, set return = -1.0
    - otherwise leave as NaN
    """
    df = ri_prices.copy()
    name_col, isin_col = get_id_cols(df)
    month_cols = sort_month_cols(get_time_cols(df))
    id_cols = [c for c in [name_col, isin_col] if c is not None]

    returns = df[id_cols].copy()

    for i, col in enumerate(month_cols):
        if i == 0:
            returns[col] = np.nan
            continue

        prev_col = month_cols[i - 1]
        prev = pd.to_numeric(df[prev_col], errors="coerce").to_numpy(dtype=float)
        curr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

        ret = np.full(len(df), np.nan)

        valid = (~np.isnan(prev)) & (prev > 0) & (~np.isnan(curr)) & (curr > 0)
        ret[valid] = curr[valid] / prev[valid] - 1.0

        delist = (~np.isnan(prev)) & (prev > 0) & (curr == 0.0)
        ret[delist] = -1.0

        returns[col] = ret

    return returns


def build_stale_flags(
    returns_df: pd.DataFrame,
    end_date: pd.Timestamp,
    window_years: int = 10,
    threshold: float = 0.50
) -> pd.DataFrame:
    """
    Compute stale-price diagnostics at a given year-end.
    """
    _, isin_col = get_id_cols(returns_df)
    month_cols = sort_month_cols(get_time_cols(returns_df))
    col_dates = pd.to_datetime(month_cols, errors="coerce")

    start_dt = end_date - pd.DateOffset(years=window_years)

    window_cols = [
        c for c, d in zip(month_cols, col_dates)
        if pd.notna(d) and start_dt < d <= end_date
    ]

    if not window_cols:
        return pd.DataFrame(columns=["ISIN", "n_valid", "n_zero", "zero_frac", "stale_flag"])

    data = returns_df[window_cols].apply(pd.to_numeric, errors="coerce")

    n_valid = data.notna().sum(axis=1)
    n_zero = (data == 0.0).sum(axis=1)
    zero_frac = n_zero / n_valid.replace(0, np.nan)

    stale_info = pd.DataFrame({
        "ISIN": returns_df[isin_col].values,
        "n_valid": n_valid.values,
        "n_zero": n_zero.values,
        "zero_frac": zero_frac.values,
    })
    stale_info["stale_flag"] = stale_info["zero_frac"] > threshold
    return stale_info


def get_december_col(month_cols, year):
    """
    Return the December column for a given year, if available.
    """
    dts = pd.to_datetime(month_cols, errors="coerce")
    candidates = [c for c, d in zip(month_cols, dts) if pd.notna(d) and d.year == year and d.month == 12]
    return candidates[-1] if candidates else None


####################################################################################
# 3. LOAD RAW FILES
####################################################################################

static = clean_headers(pd.read_csv(RAW_DIR / STATIC_FILE))
ri = clean_headers(pd.read_csv(RAW_DIR / RI_FILE))
mv = clean_headers(pd.read_csv(RAW_DIR / MV_FILE))
co2 = clean_headers(pd.read_csv(RAW_DIR / CO2_FILE))
rev = clean_headers(pd.read_csv(RAW_DIR / REV_FILE))

####################################################################################
# 4. A. RAW DATA CLEANING
####################################################################################

static_name_col, static_isin_col = get_id_cols(static)

if static_isin_col is None:
    raise ValueError("Static file must contain an ISIN column.")
if "Region" not in static.columns:
    raise ValueError("Static file must contain a 'Region' column.")

# static universe: region only
static = static.dropna(subset=[static_isin_col]).copy()
static = static[static["Region"].isin(REGIONS)].copy()

# remove firms with fully empty timeseries rows
ri = drop_empty_timeseries_rows(ri)
mv = drop_empty_timeseries_rows(mv)
co2 = drop_empty_timeseries_rows(co2)
rev = drop_empty_timeseries_rows(rev)

# restrict all tables to region universe
region_isins = set(static[static_isin_col])
ri = keep_isins(ri, region_isins)
mv = keep_isins(mv, region_isins)
co2 = keep_isins(co2, region_isins)
rev = keep_isins(rev, region_isins)

####################################################################################
# 5. B. PRICE AND RETURN CLEANING
####################################################################################

ri_month_cols = sort_month_cols(get_time_cols(ri))
ri[ri_month_cols] = ri[ri_month_cols].apply(pd.to_numeric, errors="coerce")

# detect delistings first, so explicit delisting can become 0.0
ri = detect_delisting(ri)

# treat RI < 0.5 as missing, but preserve exact 0.0 for delisting
for c in ri_month_cols:
    mask_low = (ri[c] < 0.5) & (ri[c] > 0)
    ri.loc[mask_low, c] = np.nan

# drop rows that become entirely missing after low-price cleaning
ri = ri.loc[~ri[ri_month_cols].isna().all(axis=1)].copy()

# compute monthly simple returns
ri_ret = compute_monthly_returns(ri)

####################################################################################
# 6. C. ANNUAL DATA CLEANING
####################################################################################

co2 = annual_forward_fill(co2)
rev = annual_forward_fill(rev)

# standardize annual column names as strings
co2.columns = [str(c) for c in co2.columns]
rev.columns = [str(c) for c in rev.columns]

# forward-fill MV monthly mid-sample gaps (used for portfolio weighting)
mv_month_cols = sort_month_cols(get_time_cols(mv))
mv[mv_month_cols] = mv[mv_month_cols].apply(pd.to_numeric, errors="coerce")
mv[mv_month_cols] = mv[mv_month_cols].ffill(axis=1)

####################################################################################
# 7. D. BASE COMMON UNIVERSE
# This is only the base cleaned universe across files.
# Do NOT apply year-specific stale/missing-price filters globally here.
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

static = keep_isins(static, common_isins)
ri = keep_isins(ri, common_isins)
ri_ret = keep_isins(ri_ret, common_isins)
mv = keep_isins(mv, common_isins)
co2 = keep_isins(co2, common_isins)
rev = keep_isins(rev, common_isins)

####################################################################################
# 8. E. YEAR-BY-YEAR INVESTABILITY / CLEANING DIAGNOSTICS
#
# These are NOT applied once globally.
# They are computed at the end of each year Y for investing in Y+1.
####################################################################################

# prep references
name_col, isin_col = get_id_cols(ri_ret)
ri_month_cols = sort_month_cols(get_time_cols(ri_ret))
co2_year_cols = sort_year_cols(get_time_cols(co2))

# useful maps
ri_prices_by_isin = ri.set_index(ri_isin_col)
ri_rets_by_isin = ri_ret.set_index(ri_isin_col)
co2_by_isin = co2.set_index(co2_isin_col)
rev_by_isin = rev.set_index(rev_isin_col)

screen_tables = []

for year in range(START_SCREEN_YEAR, END_SCREEN_YEAR + 1):
    dec_col = get_december_col(ri_month_cols, year)
    if dec_col is None:
        continue

    stale_info = build_stale_flags(
        returns_df=ri_ret,
        end_date=pd.Timestamp(f"{year}-12-31"),
        window_years=WINDOW_YEARS,
        threshold=STALE_THRESHOLD
    )

    if stale_info.empty:
        continue

    stale_info = stale_info.set_index("ISIN")

    screen_df = pd.DataFrame(index=sorted(common_isins))
    screen_df.index.name = "ISIN"

    # valid December price at end of year Y
    dec_price = pd.to_numeric(ri_prices_by_isin[dec_col], errors="coerce")
    screen_df["has_valid_dec_price"] = dec_price.notna()

    # enough valid monthly returns over trailing 10y window
    screen_df["n_valid_return_months"] = stale_info["n_valid"]
    screen_df["has_min_history"] = screen_df["n_valid_return_months"] >= MIN_VALID_MONTHS

    # stale price rule
    screen_df["zero_return_fraction"] = stale_info["zero_frac"]
    screen_df["stale_flag"] = stale_info["stale_flag"]

    # carbon data available at end of year Y
    year_str = str(year)
    if year_str in co2_by_isin.columns:
        screen_df["has_co2_at_year_end"] = pd.to_numeric(co2_by_isin[year_str], errors="coerce").notna()
    else:
        screen_df["has_co2_at_year_end"] = False

    # revenue data available at end of year Y
    if year_str in rev_by_isin.columns:
        screen_df["has_rev_at_year_end"] = pd.to_numeric(rev_by_isin[year_str], errors="coerce").notna()
    else:
        screen_df["has_rev_at_year_end"] = False

    # final investable flag for year Y+1
    screen_df["investable_for_next_year"] = (
        screen_df["has_valid_dec_price"]
        & screen_df["has_min_history"]
        & (~screen_df["stale_flag"])
        & screen_df["has_co2_at_year_end"]
        & screen_df["has_rev_at_year_end"]
    )

    screen_df = screen_df.reset_index()
    screen_df["screen_year"] = year
    screen_df["investment_year"] = year + 1

    screen_tables.append(screen_df)

screen_results = pd.concat(screen_tables, ignore_index=True) if screen_tables else pd.DataFrame()

####################################################################################
# 9. OPTIONAL: SORT CLEANED FILES
####################################################################################

static = static.sort_values(static_isin_col).reset_index(drop=True)
ri = ri.sort_values(ri_isin_col).reset_index(drop=True)
ri_ret = ri_ret.sort_values(ri_isin_col).reset_index(drop=True)
mv = mv.sort_values(mv_isin_col).reset_index(drop=True)
co2 = co2.sort_values(co2_isin_col).reset_index(drop=True)
rev = rev.sort_values(rev_isin_col).reset_index(drop=True)

####################################################################################
# 10. SAVE CLEANED OUTPUTS
####################################################################################

static.to_csv(CLEAN_DIR / "cleaned_static.csv", index=False)
ri.to_csv(CLEAN_DIR / "cleaned_RI_monthly_prices.csv", index=False)
ri_ret.to_csv(CLEAN_DIR / "cleaned_RI_monthly_returns.csv", index=False)
mv.to_csv(CLEAN_DIR / "cleaned_MV_monthly.csv", index=False)
co2.to_csv(CLEAN_DIR / "cleaned_CO2_scope1_yearly.csv", index=False)
rev.to_csv(CLEAN_DIR / "cleaned_REV_yearly.csv", index=False)

# diagnostics / year-specific screens
screen_results.to_csv(CLEAN_DIR / "investment_set_screens_by_year.csv", index=False)

####################################################################################
# 11. PRINT SUMMARY
####################################################################################

print("Data cleaning / pre-processing complete.")
print(f"Base common universe size: {len(common_isins)} firms")
print("Files saved in:", CLEAN_DIR)
print("- cleaned_static.csv")
print("- cleaned_RI_monthly_prices.csv")
print("- cleaned_RI_monthly_returns.csv")
print("- cleaned_MV_monthly.csv")
print("- cleaned_CO2_scope1_yearly.csv")
print("- cleaned_REV_yearly.csv")
print("- investment_set_screens_by_year.csv")

if not screen_results.empty:
    summary_by_year = (
        screen_results.groupby("screen_year")["investable_for_next_year"]
        .sum()
        .rename("n_investable")
    )
    print("\nInvestable firms by year-end screen:")
    print(summary_by_year)