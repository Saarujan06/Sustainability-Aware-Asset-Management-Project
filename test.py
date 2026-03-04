####################################################################################
# Part I - Standard Portfolio Allocation
####################################################################################
# 1. Data Cleaning
####################################################################################

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV_DIR = BASE_DIR / "raw_data"
cleaned_dir = BASE_DIR / "cleaned_data"
cleaned_dir.mkdir(exist_ok=True)

####################################################################################
# STEP 1: CLEAN STATIC
####################################################################################

static = pd.read_csv(DATA_CSV_DIR / "Static_2025.csv")

static = static.dropna(subset=["ISIN"])
static = static[static["Region"].isin(["AMER", "EUR"])]

static.to_csv(cleaned_dir / "cleaned_static_2025.csv", index=False)

print("Static cleaned")

valid_isins = set(static["ISIN"])

####################################################################################
# STEP 2: CLEAN RI MONTHLY
####################################################################################

ri = pd.read_csv(DATA_CSV_DIR / "DS_RI_T_USD_M_2025.csv")

ri = ri.dropna(subset=["ISIN"])
ri = ri[ri["ISIN"].isin(valid_isins)]

date_cols = [c for c in ri.columns if c not in ["NAME", "ISIN"]]
date_cols = sorted(date_cols)

# remove dates after 2025
date_cols = [c for c in date_cols if c <= "2025-12-31 00:00:00"]

ri = ri[["NAME", "ISIN"] + date_cols]

ri[date_cols] = ri[date_cols].apply(pd.to_numeric, errors="coerce")

# treat prices < 0.5 as missing
ri[date_cols] = ri[date_cols].mask(ri[date_cols] < 0.5)

# compute monthly returns
ri_ret = ri.copy()
ri_ret[date_cols] = ri[date_cols].pct_change(axis=1)

ri.to_csv(cleaned_dir / "cleaned_RI_monthly_prices.csv", index=False)
ri_ret.to_csv(cleaned_dir / "cleaned_RI_monthly_returns.csv", index=False)

print("RI cleaned")

####################################################################################
# STEP 3: CLEAN MV MONTHLY
####################################################################################

mv = pd.read_csv(DATA_CSV_DIR / "DS_MV_T_USD_M_2025.csv")

mv = mv.dropna(subset=["ISIN"])
mv = mv[mv["ISIN"].isin(valid_isins)]

mv_date_cols = [c for c in mv.columns if c not in ["NAME", "ISIN"]]
mv_date_cols = sorted(mv_date_cols)

mv_date_cols = [c for c in mv_date_cols if c <= "2025-12-31 00:00:00"]

mv = mv[["NAME", "ISIN"] + mv_date_cols]

mv[mv_date_cols] = mv[mv_date_cols].apply(pd.to_numeric, errors="coerce")

mv.to_csv(cleaned_dir / "cleaned_MV_monthly.csv", index=False)

print("MV cleaned")

####################################################################################
# STEP 4: CLEAN CO2 SCOPE 1
####################################################################################

co2 = pd.read_csv(DATA_CSV_DIR / "DS_CO2_SCOPE_1_Y_2025.csv")

co2 = co2.dropna(subset=["ISIN"])
co2 = co2[co2["ISIN"].isin(valid_isins)]

year_cols = [c for c in co2.columns if c not in ["NAME", "ISIN"]]

co2[year_cols] = co2[year_cols].apply(pd.to_numeric, errors="coerce")

# forward fill missing values
co2[year_cols] = co2[year_cols].ffill(axis=1)

co2.to_csv(cleaned_dir / "cleaned_CO2_scope1_yearly.csv", index=False)

print("CO2 cleaned")

####################################################################################
# STEP 5: CLEAN REVENUE
####################################################################################

rev = pd.read_csv(DATA_CSV_DIR / "DS_REV_Y_2025.csv")

rev = rev.dropna(subset=["ISIN"])
rev = rev[rev["ISIN"].isin(valid_isins)]

rev_year_cols = [c for c in rev.columns if c not in ["NAME", "ISIN"]]

rev[rev_year_cols] = rev[rev_year_cols].apply(pd.to_numeric, errors="coerce")

rev[rev_year_cols] = rev[rev_year_cols].ffill(axis=1)

rev.to_csv(cleaned_dir / "cleaned_REV_yearly.csv", index=False)

print("Revenue cleaned")