####################################################################################
# Part I - Standard Portfolio Allocation
####################################################################################

####################################################################################
# 1. Data Cleaning
####################################################################################

from datetime import date
import os
import numpy as np
from pathlib import Path
import pandas as pd

####################################################################################
# ---------- STEP 1: CLEAN STATIC ----------
####################################################################################

# determine project base directory and data folder
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV_DIR = BASE_DIR / "raw_data"

# load the raw static file (use explicit path inside Data_2026/csv)
static = pd.read_csv(DATA_CSV_DIR / "Static_2025.csv")

# remove rows where ISIN is missing
static = static.dropna(subset=["ISIN"])

# keep only North America and Europe
static = static[static["Region"].isin(["AMER", "EUR"])]

# create folder cleaned_data (in project root) if it doesn't exist
cleaned_dir = BASE_DIR / "cleaned_data"
cleaned_dir.mkdir(parents=True, exist_ok=True)

# save cleaned file
static.to_csv(cleaned_dir / "cleaned_static_2025.csv", index=False)

print("Static file cleaned and saved.")

####################################################################################
# ---------- STEP 2: CLEAN RI MONTHLY ----------
####################################################################################

# load cleaned static to get the valid ISIN list
cleaned_static = pd.read_csv(cleaned_dir / "cleaned_static_2025.csv")
valid_isins = set(cleaned_static["ISIN"])

# load RI monthly
ri = pd.read_csv(DATA_CSV_DIR / "DS_RI_T_USD_M_2025.csv")

# remove empty rows
ri = ri.dropna(subset=["ISIN"])

# keep only AMER + EUR firms
ri = ri[ri["ISIN"].isin(valid_isins)].copy()

# identify date columns
date_cols = [c for c in ri.columns if c not in ["NAME", "ISIN"]]
date_cols = sorted(date_cols)

# convert to numeric
ri[date_cols] = ri[date_cols].apply(pd.to_numeric, errors="coerce")

# treat low prices as missing
ri[date_cols] = ri[date_cols].mask(ri[date_cols] < 0.5)

# compute returns
ri_ret = ri[["NAME", "ISIN"]].copy()
ri_ret[date_cols] = ri[date_cols].pct_change(axis=1)

# save
ri.to_csv(cleaned_dir / "cleaned_RI_monthly_prices.csv", index=False)
ri_ret.to_csv(cleaned_dir / "cleaned_RI_monthly_returns.csv", index=False)

print("RI cleaned (prices + returns) and saved.")

####################################################################################
# ---------- STEP 3: CLEAN MV MONTHLY ----------
####################################################################################

mv = pd.read_csv(DATA_CSV_DIR / "DS_MV_T_USD_M_2025.csv")

# remove Datastream empty row(s)
mv = mv.dropna(subset=["ISIN"])

# keep only AMER + EUR firms
mv = mv[mv["ISIN"].isin(valid_isins)].copy()

# date columns
mv_date_cols = [c for c in mv.columns if c not in ["NAME", "ISIN"]]

# numeric
mv[mv_date_cols] = mv[mv_date_cols].apply(pd.to_numeric, errors="coerce")

# save
mv.to_csv("cleaned_data/cleaned_MV_monthly.csv", index=False)

print("MV monthly cleaned and saved.")

mv_clean = pd.read_csv("cleaned_data/cleaned_MV_monthly.csv")

####################################################################################
# ---------- STEP 4: CLEAN CO2 SCOPE 1 (YEARLY) ----------
####################################################################################

co2 = pd.read_csv(DATA_CSV_DIR / "DS_CO2_SCOPE_1_Y_2025.csv")

# remove Datastream empty row(s)
co2 = co2.dropna(subset=["ISIN"])

# keep only AMER + EUR firms
co2 = co2[co2["ISIN"].isin(valid_isins)].copy()

# year columns (everything except NAME, ISIN)
year_cols = [c for c in co2.columns if c not in ["NAME", "ISIN"]]

# numeric
co2[year_cols] = co2[year_cols].apply(pd.to_numeric, errors="coerce")

# forward fill across years (use previous year if missing in middle/end)
co2[year_cols] = co2[year_cols].ffill(axis=1)

# save
co2.to_csv("cleaned_data/cleaned_CO2_scope1_yearly.csv", index=False)

print("CO2 Scope 1 yearly cleaned and saved.")

####################################################################################
# ---------- STEP 5: CLEAN REVENUE (YEARLY) ----------
####################################################################################

rev = pd.read_csv(DATA_CSV_DIR / "DS_REV_Y_2025.csv")

# remove Datastream empty row(s)
rev = rev.dropna(subset=["ISIN"])

# keep only AMER + EUR firms
rev = rev[rev["ISIN"].isin(valid_isins)].copy()

# year columns
rev_year_cols = [c for c in rev.columns if c not in ["NAME", "ISIN"]]

# numeric
rev[rev_year_cols] = rev[rev_year_cols].apply(pd.to_numeric, errors="coerce")

# forward fill across years (use previous year if missing in middle/end)
rev[rev_year_cols] = rev[rev_year_cols].ffill(axis=1)

# save
rev.to_csv("cleaned_data/cleaned_REV_yearly.csv", index=False)

print("Revenue yearly cleaned and saved.")
