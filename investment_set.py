####################################################################################
# SAAM Project 2026 - Part I
# SECTION 2.1 - INVESTMENT SET
####################################################################################

from pathlib import Path
import numpy as np
import pandas as pd

####################################################################################
# 0. PATHS
####################################################################################

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "cleaned_data"
OUT_DIR = BASE_DIR / "investment_set_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

####################################################################################
# 1. CONFIG
####################################################################################

START_YEAR = 2013
END_YEAR = 2024
WINDOW_YEARS = 10
MIN_VALID_MONTHS = 36
STALE_THRESHOLD = 0.50

RI_PRICES_FILE = "cleaned_RI_monthly_prices.csv"
RI_RETURNS_FILE = "cleaned_RI_monthly_returns.csv"
CO2_FILE = "cleaned_CO2_scope1_yearly.csv"

####################################################################################
# 2. HELPERS
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


def get_december_col(month_cols, year):
    dts = pd.to_datetime(month_cols, errors="coerce")
    candidates = [c for c, d in zip(month_cols, dts) if pd.notna(d) and d.year == year and d.month == 12]
    return candidates[-1] if candidates else None


def flag_stale_prices(
    returns_df: pd.DataFrame,
    end_date: str,
    window_years: int = 10,
    threshold: float = 0.50
):
    """
    Compute stale-price diagnostics over trailing window ending at end_date.
    """
    _, isin_col = get_id_cols(returns_df)
    month_cols = sort_month_cols(get_time_cols(returns_df))

    end_dt = pd.Timestamp(end_date)
    start_dt = end_dt - pd.DateOffset(years=window_years)

    col_dates = pd.to_datetime(month_cols, errors="coerce")
    window_cols = [
        c for c, d in zip(month_cols, col_dates)
        if pd.notna(d) and start_dt < d <= end_dt
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


####################################################################################
# 3. LOAD CLEANED FILES
####################################################################################

ri_prices = clean_headers(pd.read_csv(CLEAN_DIR / RI_PRICES_FILE))
ri_returns = clean_headers(pd.read_csv(CLEAN_DIR / RI_RETURNS_FILE))
co2 = clean_headers(pd.read_csv(CLEAN_DIR / CO2_FILE))

# make sure annual columns are strings
co2.columns = [str(c) for c in co2.columns]

####################################################################################
# 4. PREP
####################################################################################

_, ri_price_isin_col = get_id_cols(ri_prices)
_, ri_ret_isin_col = get_id_cols(ri_returns)
_, co2_isin_col = get_id_cols(co2)

ri_month_cols = sort_month_cols(get_time_cols(ri_prices))

ri_prices_by_isin = ri_prices.set_index(ri_price_isin_col)
co2_by_isin = co2.set_index(co2_isin_col)

all_isins = sorted(set(ri_prices_by_isin.index) & set(ri_returns[ri_ret_isin_col]) & set(co2_by_isin.index))

####################################################################################
# 5. BUILD YEAR-BY-YEAR INVESTMENT SET
####################################################################################

investment_tables = []

for year in range(START_YEAR, END_YEAR + 1):
    dec_col = get_december_col(ri_month_cols, year)
    if dec_col is None:
        print(f"Skipping {year}: no December column found.")
        continue

    stale_info = flag_stale_prices(
        returns_df=ri_returns,
        end_date=f"{year}-12-31",
        window_years=WINDOW_YEARS,
        threshold=STALE_THRESHOLD
    )

    if stale_info.empty:
        print(f"Skipping {year}: stale-price window empty.")
        continue

    stale_info = stale_info.set_index("ISIN")

    screen_df = pd.DataFrame(index=all_isins)
    screen_df.index.name = "ISIN"

    # 1. valid December price at end of year Y (exclude delisted firms with price == 0)
    dec_price = pd.to_numeric(ri_prices_by_isin[dec_col], errors="coerce")
    screen_df["has_valid_dec_price"] = dec_price.notna() & (dec_price > 0)

    # 2. enough monthly return history in trailing 10-year window
    screen_df["n_valid_return_months"] = stale_info["n_valid"]
    screen_df["has_min_history"] = screen_df["n_valid_return_months"] >= MIN_VALID_MONTHS

    # 3. stale-price screen
    screen_df["zero_return_fraction"] = stale_info["zero_frac"]
    screen_df["stale_flag"] = stale_info["stale_flag"]

    # 4. carbon data available at end of year Y
    year_str = str(year)
    if year_str in co2_by_isin.columns:
        screen_df["has_co2_at_year_end"] = pd.to_numeric(co2_by_isin[year_str], errors="coerce").notna()
    else:
        screen_df["has_co2_at_year_end"] = False

    # final investable flag for year Y+1
    screen_df["investable_for_next_year"] = (
        screen_df["has_valid_dec_price"]
        & screen_df["has_min_history"]
        & (~screen_df["stale_flag"])
        & screen_df["has_co2_at_year_end"]
    )

    screen_df = screen_df.reset_index()
    screen_df["screen_year"] = year
    screen_df["investment_year"] = year + 1

    investment_tables.append(screen_df)

investment_set_df = pd.concat(investment_tables, ignore_index=True) if investment_tables else pd.DataFrame()

####################################################################################
# 6. SAVE OUTPUTS
####################################################################################

investment_set_df.to_csv(OUT_DIR / "investment_set_by_year.csv", index=False)

summary_by_year = (
    investment_set_df.groupby("screen_year")["investable_for_next_year"]
    .sum()
    .rename("n_investable")
)

summary_by_year.to_csv(OUT_DIR / "investment_set_summary_by_year.csv")

####################################################################################
# 7. PRINT SUMMARY
####################################################################################

print("Section 2.1 complete.")
print("Files saved in:", OUT_DIR)
print("- investment_set_by_year.csv")
print("- investment_set_summary_by_year.csv")
print("\nNumber of investable firms by year:")
print(summary_by_year)