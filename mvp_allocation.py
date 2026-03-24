####################################################################################
# SAAM Project 2026 - Part I
# SECTION 2.2 - MINIMUM VARIANCE PORTFOLIO (long-only, out-of-sample)
#
# For each screen year Y = 2013 ... 2024:
# 1. Load investable ISINs from Section 2.1 output
# 2. Re-estimate mu and Sigma from 10-year trailing monthly returns
#    (complete cases only, consistent with investment_set.py)
# 3. Solve long-only MVP:
#       min alpha' Sigma alpha
#       s.t. sum(alpha) = 1
#            alpha_i >= 0
# 4. Apply weights to months of year Y+1 with intra-year weight drift:
#       Rp,t+k = alpha_{t+k-1}' R_{t+k}
#       alpha_{i,t+k} = alpha_{i,t+k-1} * (1 + R_{i,t+k}) / (1 + Rp,t+k)
#
# Outputs (saved in mvp_output/):
# - mvp_monthly_returns.csv : monthly portfolio returns Jan 2014 - Dec 2025
# - mvp_weights_by_year.csv : optimal weights per screen year (long format)
# - mvp_performance_summary.csv : annualised mean, vol, Sharpe, min, max
####################################################################################

from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

####################################################################################
# 0. PATHS
####################################################################################

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "cleaned_data"
INV_DIR = BASE_DIR / "investment_set_output"
RAW_DIR = BASE_DIR / "raw_data"
OUT_DIR = BASE_DIR / "mvp_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

####################################################################################
# 1. CONFIG
####################################################################################

START_YEAR = 2013  # first screen year (Dec 2013 -> invest Jan-Dec 2014)
END_YEAR = 2024  # last screen year (Dec 2024 -> invest Jan-Dec 2025)
WINDOW_YEARS = 10  # trailing window for moment estimation

####################################################################################
# 2. HELPERS (mirrors investment_set.py)
####################################################################################


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_id_cols(df: pd.DataFrame):
    upper = {c.upper(): c for c in df.columns}
    return upper.get("NAME"), upper.get("ISIN")


def get_time_cols(df: pd.DataFrame):
    name_col, isin_col = get_id_cols(df)
    exclude = {c for c in [name_col, isin_col] if c is not None}
    return [c for c in df.columns if c not in exclude]


def sort_month_cols(cols):
    dt = pd.to_datetime(cols, errors="coerce")
    if dt.notna().all():
        return [c for _, c in sorted(zip(dt, cols))]
    return sorted(cols)


def get_trailing_window_cols(month_cols, end_year: int, window_years: int):
    """Return column names within the trailing window ending Dec of end_year."""
    end_dt = pd.Timestamp(f"{end_year}-12-31")
    start_dt = end_dt - pd.DateOffset(years=window_years)
    col_dates = pd.to_datetime(month_cols, errors="coerce")
    return [
        c
        for c, d in zip(month_cols, col_dates)
        if pd.notna(d) and start_dt < d <= end_dt
    ]


def get_investment_year_cols(month_cols, investment_year: int):
    """Return column names whose date falls in investment_year (Jan-Dec)."""
    col_dates = pd.to_datetime(month_cols, errors="coerce")
    return sorted(
        [
            c
            for c, d in zip(month_cols, col_dates)
            if pd.notna(d) and d.year == investment_year
        ],
        key=lambda c: pd.Timestamp(c),
    )


def solve_mvp(cov: np.ndarray) -> np.ndarray:
    """
    Solve the long-only minimum-variance portfolio via cvxpy (OSQP solver):
    min alpha' @ cov @ alpha
    s.t. sum(alpha) = 1, alpha_i >= 0
    Returns normalised weight vector of shape (N,).
    """
    n = cov.shape[0]
    alpha = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(alpha, cov))
    constraints = [cp.sum(alpha) == 1, alpha >= 0]
    prob = cp.Problem(objective, constraints)

    prob.solve(
        solver=cp.OSQP,
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=10000,
        warm_start=True,
        verbose=False,
    )

    if prob.status not in ("optimal", "optimal_inaccurate"):
        print(f" [WARNING] Solver status: {prob.status}. Falling back to equal weight.")
        return np.ones(n) / n

    w = np.array(alpha.value, dtype=float)
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        return np.ones(n) / n
    w /= total
    return w


####################################################################################
# 3. LOAD DATA
####################################################################################

print("Loading data ...")

# --- investment set flags from Section 2.1 ---
inv_set = clean_headers(pd.read_csv(INV_DIR / "investment_set_by_year.csv"))

# --- monthly returns (wide: rows=ISIN, cols=date strings) ---
ri_returns = clean_headers(pd.read_csv(CLEAN_DIR / "cleaned_RI_monthly_returns.csv"))
_, ret_isin_col = get_id_cols(ri_returns)
ret_month_cols = sort_month_cols(get_time_cols(ri_returns))
ri_ret_by_isin = ri_returns.set_index(ret_isin_col)  # index = ISIN

# --- risk-free rate (monthly, in percent -> divide by 100 for decimal) ---
rf_raw = pd.read_csv(RAW_DIR / "Risk_Free_Rate_2025.csv", header=0)
rf_raw.columns = ["date_key", "RF"]
rf_raw["date"] = pd.to_datetime(rf_raw["date_key"].astype(str), format="%Y%m")
rf_series = rf_raw.set_index("date")["RF"] / 100.0  # monthly decimal rate

print(f" Returns matrix : {ri_returns.shape[0]} ISINs x {len(ret_month_cols)} months")
print(f" RF rate        : {rf_series.index[0].date()} to {rf_series.index[-1].date()}")
print()

####################################################################################
# 4. MAIN LOOP
####################################################################################

print("Solving MVP and computing monthly returns ...")
print(f"{'Year':<6} {'N invest':>9} {'N in cov':>9} {'Top wt':>8} {'Months':>7}")
print("-" * 45)

all_weights_rows = []  # for mvp_weights_by_year.csv
monthly_ret_rows = []  # for mvp_monthly_returns.csv

for year in range(START_YEAR, END_YEAR + 1):
    inv_year = year + 1

    # 4.1 Investable ISINs for this screen year
    mask = (inv_set["screen_year"] == year) & (
        inv_set["investable_for_next_year"] == True
    )
    investable_isins = inv_set.loc[mask, "ISIN"].tolist()

    # 4.2 Trailing window columns for moment estimation
    window_cols = get_trailing_window_cols(ret_month_cols, year, WINDOW_YEARS)

    if not window_cols or not investable_isins:
        print(f"{year:<6} {'skipped':>9}")
        continue

    # 4.3 Extract returns for the window (exact same logic as investment_set.py)
    ret_window = (
        ri_ret_by_isin.reindex(investable_isins)[window_cols].apply(
            pd.to_numeric, errors="coerce"
        )
    )
    ret_window_t = ret_window.T  # (tau, N_investable)

    # Keep only firms with a completely filled window (no NaNs)
    complete_mask = ret_window_t.notna().all(axis=0)
    complete_assets = ret_window_t.columns[complete_mask].tolist()
    ret_complete = ret_window_t[complete_assets].values  # (tau, N)

    if ret_complete.shape[1] == 0:
        print(f"{year:<6} {'no complete assets':>9}")
        continue

    # 4.4 Compute mu and Sigma (same formula as investment_set.py)
    tau = float(ret_complete.shape[0])
    mu = ret_complete.mean(axis=0)  # (N,)
    centered = ret_complete - mu
    cov = (centered.T @ centered) / tau  # (N, N)

    # 4.5 Solve long-only MVP
    weights = solve_mvp(cov)
    n_assets = len(complete_assets)

    # Store weights (long format, skip negligible weights)
    for isin, w in zip(complete_assets, weights):
        if w > 1e-8:
            all_weights_rows.append(
                {
                    "screen_year": year,
                    "investment_year": inv_year,
                    "ISIN": isin,
                    "weight": w,
                }
            )

    # 4.6 Compute monthly portfolio returns for investment_year with weight drift
    inv_cols = get_investment_year_cols(ret_month_cols, inv_year)

    if not inv_cols:
        print(
            f"{year:<6} {len(investable_isins):>9} {n_assets:>9} "
            f"{weights.max():>8.4f} {'no ret cols':>7}"
        )
        continue

    # Returns matrix for investment year: shape (N, T_inv)
    inv_ret_mat = (
        ri_ret_by_isin.reindex(complete_assets)[inv_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)  # firm delisted mid-year -> 0 return after delisting
        .values
    )  # (N, T_inv)

    alpha = weights.copy()  # weights at start of the investment year

    for k, col in enumerate(inv_cols):
        rt = inv_ret_mat[:, k]  # (N,) stock returns this month
        rp = float(alpha @ rt)  # portfolio return this month

        monthly_ret_rows.append(
            {
                "date": pd.Timestamp(col),
                "Rp_mvp": rp,
            }
        )

        # Drift: alpha_{t+1,i} = alpha_{t,i} * (1 + R_{t,i}) / (1 + Rp_t)
        denom = 1.0 + rp
        if abs(denom) < 1e-10:
            # Total wipeout edge case: reset to equal weight
            alpha = np.ones(n_assets) / n_assets
        else:
            alpha = alpha * (1.0 + rt) / denom
            s = alpha.sum()
            if s > 1e-10:
                alpha /= s  # renormalise to handle floating-point drift
            else:
                alpha = np.ones(n_assets) / n_assets

    print(
        f"{year:<6} {len(investable_isins):>9} {n_assets:>9} "
        f"{weights.max():>8.4f} {len(inv_cols):>7}"
    )

####################################################################################
# 5. BUILD RETURN SERIES
####################################################################################

ret_df = pd.DataFrame(monthly_ret_rows)

if ret_df.empty:
    raise RuntimeError("No monthly returns were generated. Check input data and year range.")

ret_df = ret_df.sort_values("date").set_index("date")

print(
    f"\nTotal months in return series: {len(ret_df)} "
    f"({ret_df.index[0].strftime('%b %Y')} - {ret_df.index[-1].strftime('%b %Y')})"
)

####################################################################################
# 6. ALIGN RISK-FREE RATE AND COMPUTE PERFORMANCE
####################################################################################

# Match RF by year-month period to avoid timestamp misalignment
ret_periods = ret_df.index.to_period("M")
rf_by_period = rf_series.copy()
rf_by_period.index = rf_series.index.to_period("M")
rf_aligned = rf_by_period.reindex(ret_periods).values  # (T,)

if np.isnan(rf_aligned).any():
    n_missing = int(np.isnan(rf_aligned).sum())
    print(f"[WARNING] {n_missing} RF month(s) missing; filling with 0.")
    rf_aligned = np.nan_to_num(rf_aligned, nan=0.0)

rp = ret_df["Rp_mvp"].values
excess = rp - rf_aligned

ann = 12
mu_ann = rp.mean() * ann
vol_ann = rp.std(ddof=1) * np.sqrt(ann)
excess_std = excess.std(ddof=1)
sr = np.nan if excess_std == 0 else (excess.mean() / excess_std) * np.sqrt(ann)
ret_min = rp.min()
ret_max = rp.max()

print(f"\n{'=' * 52}")
print(" MVP OUT-OF-SAMPLE PERFORMANCE")
print(
    f" ({ret_df.index[0].strftime('%b %Y')} - {ret_df.index[-1].strftime('%b %Y')})"
)
print(f"{'=' * 52}")
print(f" Annualised mean return : {mu_ann * 100:>8.2f}%")
print(f" Annualised volatility  : {vol_ann * 100:>8.2f}%")
print(f" Sharpe ratio           : {sr:>8.3f}")
print(f" Min monthly return     : {ret_min * 100:>8.2f}%")
print(f" Max monthly return     : {ret_max * 100:>8.2f}%")
print(f" Number of months       : {len(rp):>8d}")
print(f"{'=' * 52}")

####################################################################################
# 7. SAVE OUTPUTS
####################################################################################

ret_df.to_csv(OUT_DIR / "mvp_monthly_returns.csv")

pd.DataFrame(
    [
        {
            "portfolio": "MVP_long_only_OOS",
            "start_date": ret_df.index[0].strftime("%Y-%m"),
            "end_date": ret_df.index[-1].strftime("%Y-%m"),
            "n_months": len(rp),
            "ann_mean_return": mu_ann,
            "ann_volatility": vol_ann,
            "sharpe_ratio": sr,
            "min_monthly_ret": ret_min,
            "max_monthly_ret": ret_max,
        }
    ]
).to_csv(OUT_DIR / "mvp_performance_summary.csv", index=False)

pd.DataFrame(all_weights_rows).to_csv(OUT_DIR / "mvp_weights_by_year.csv", index=False)

print(f"\nFiles saved in: {OUT_DIR}")
print(" - mvp_monthly_returns.csv")
print(" - mvp_performance_summary.csv")
print(" - mvp_weights_by_year.csv")
print("\nSection 2.2 complete.")