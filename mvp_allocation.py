####################################################################################
# SAAM Project 2026 - Part I
# SECTION 2.2 - MINIMUM VARIANCE PORTFOLIO
# BLOCK 1: SETUP + LOAD INPUTS
####################################################################################

from pathlib import Path
import numpy as np
import pandas as pd
import cvxpy as cp

####################################################################################
# 0. PATHS
####################################################################################

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "cleaned_data"
INV_DIR = BASE_DIR / "investment_set_output"
OUT_DIR = BASE_DIR / "mvp_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

####################################################################################
# 1. CONFIG
####################################################################################

START_YEAR = 2013
END_YEAR = 2024
WINDOW_YEARS = 10

RI_RETURNS_FILE = "cleaned_RI_monthly_returns.csv"
INVESTMENT_SET_FILE = "investment_set_by_year.csv"

####################################################################################
# 2. HELPERS
####################################################################################

def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_id_cols(df: pd.DataFrame):
    upper = {c.upper(): c for c in df.columns}
    name_col = upper.get("NAME", None)
    isin_col = upper.get("ISIN", None)
    return name_col, isin_col


def get_time_cols(df: pd.DataFrame):
    name_col, isin_col = get_id_cols(df)
    exclude = {c for c in [name_col, isin_col] if c is not None}
    return [c for c in df.columns if c not in exclude]


def sort_month_cols(cols):
    dt = pd.to_datetime(cols, errors="coerce")
    if dt.notna().all():
        return [c for _, c in sorted(zip(dt, cols))]
    return sorted(cols)


def get_trailing_window_cols(month_cols, end_year: int, window_years: int = 10):
    """
    Return monthly columns in the trailing 10-year window ending in Dec of end_year.
    Example: end_year=2013 -> Jan 2004 to Dec 2013
    """
    end_dt = pd.Timestamp(f"{end_year}-12-31")
    start_dt = end_dt - pd.DateOffset(years=window_years)

    col_dates = pd.to_datetime(month_cols, errors="coerce")

    return [
        c for c, d in zip(month_cols, col_dates)
        if pd.notna(d) and start_dt < d <= end_dt
    ]


def get_investment_year_cols(month_cols, investment_year: int):
    """
    Return monthly columns in Jan-Dec of investment_year.
    """
    col_dates = pd.to_datetime(month_cols, errors="coerce")

    return [
        c for c, d in zip(month_cols, col_dates)
        if pd.notna(d) and d.year == investment_year
    ]

####################################################################################
# 3. LOAD INPUTS
####################################################################################

# 3.1 Monthly stock returns from cleaned data
ri_returns = clean_headers(pd.read_csv(CLEAN_DIR / RI_RETURNS_FILE))
_, ret_isin_col = get_id_cols(ri_returns)
ret_month_cols = sort_month_cols(get_time_cols(ri_returns))

ri_ret_by_isin = ri_returns.set_index(ret_isin_col)

# 3.2 Investment-set output from Section 2.1
investment_set = clean_headers(pd.read_csv(INV_DIR / INVESTMENT_SET_FILE))

####################################################################################
# 4. BASIC CHECKS
####################################################################################

required_cols = {
    "screen_year",
    "investment_year",
    "ISIN",
    "investable_for_next_year"
}
missing_cols = required_cols - set(investment_set.columns)
if missing_cols:
    raise ValueError(f"Missing columns in investment set file: {missing_cols}")

print(f"Returns matrix: {ri_ret_by_isin.shape[0]} firms x {len(ret_month_cols)} months")
print(f"Investment set rows: {len(investment_set)}")
print(f"Years covered: {START_YEAR} to {END_YEAR}")

####################################################################################
# 5. MVP HELPERS
####################################################################################

def make_psd(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Force a symmetric matrix to be positive semidefinite by clipping
    negative eigenvalues.
    """
    mat = np.asarray(mat, dtype=float)
    mat = 0.5 * (mat + mat.T)

    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals_clipped = np.clip(eigvals, eps, None)

    mat_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    mat_psd = 0.5 * (mat_psd + mat_psd.T)
    return mat_psd


def solve_mvp(cov: np.ndarray) -> np.ndarray:
    """
    Solve the long-only minimum-variance portfolio:
        min w' Σ w
        s.t. sum(w) = 1
             w >= 0
    """
    n = cov.shape[0]
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    constraints = [cp.sum(w) == 1, w >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"MVP optimisation failed with status: {problem.status}")

    weights = np.array(w.value).flatten()
    weights = np.clip(weights, 0, None)

    if weights.sum() <= 0:
        raise RuntimeError("Optimiser returned non-positive total weight.")

    return weights / weights.sum()

####################################################################################
# 6. YEAR-BY-YEAR MOMENT ESTIMATION AND MVP WEIGHTS
####################################################################################

weights_rows = []
year_summary_rows = []

for year in range(START_YEAR, END_YEAR + 1):
    investment_year = year + 1

    # 6.1 Get investable ISINs from Section 2.1
    investable_isins = investment_set.loc[
        (investment_set["screen_year"] == year)
        & (investment_set["investable_for_next_year"] == True),
        "ISIN"
    ].tolist()

    if len(investable_isins) == 0:
        print(f"{year}: no investable firms")
        continue

    # 6.2 Get trailing 10-year return window
    window_cols = get_trailing_window_cols(
        ret_month_cols,
        end_year=year,
        window_years=WINDOW_YEARS
    )

    if len(window_cols) == 0:
        print(f"{year}: no trailing window columns")
        continue

    # 6.3 Extract returns for investable firms only
    ret_window = (
        ri_ret_by_isin
        .reindex(investable_isins)[window_cols]
        .apply(pd.to_numeric, errors="coerce")
    )

    # transpose: rows = months, cols = assets
    ret_window_t = ret_window.T

    # 6.4 Compute means with available observations
    mu = ret_window_t.mean(axis=0, skipna=True)

    # keep only assets with a defined mean
    valid_assets = mu.index[mu.notna()].tolist()
    if len(valid_assets) == 0:
        print(f"{year}: no valid assets after mean estimation")
        continue

    ret_window_t = ret_window_t[valid_assets]
    mu = mu[valid_assets]

    # 6.5 Pairwise covariance matrix using overlapping available data
    cov_df = ret_window_t.cov(ddof=0)

    # drop assets whose covariance rows/cols are entirely missing
    valid_cov_assets = cov_df.index[~cov_df.isna().all(axis=1)].tolist()
    if len(valid_cov_assets) == 0:
        print(f"{year}: no valid assets after covariance estimation")
        continue

    cov_df = cov_df.loc[valid_cov_assets, valid_cov_assets]
    mu = mu.loc[valid_cov_assets]

    # 6.6 Fill remaining NaNs conservatively
    # off-diagonal NaNs -> 0
    # diagonal NaNs -> available sample variance
    diag_vars = ret_window_t[valid_cov_assets].var(axis=0, ddof=0).to_numpy()
    cov = cov_df.to_numpy(dtype=float)

    for i in range(cov.shape[0]):
        if np.isnan(cov[i, i]):
            cov[i, i] = diag_vars[i]

    cov = np.nan_to_num(cov, nan=0.0)

    # 6.7 Force PSD for optimisation stability
    cov = make_psd(cov, eps=1e-8)

    # 6.8 Solve MVP
    weights = solve_mvp(cov)

    # 6.9 Save weights
    for isin, w in zip(valid_cov_assets, weights):
        weights_rows.append({
            "screen_year": year,
            "investment_year": investment_year,
            "ISIN": isin,
            "weight": w
        })

    year_summary_rows.append({
        "screen_year": year,
        "investment_year": investment_year,
        "n_investable_2_1": len(investable_isins),
        "n_assets_used_in_cov": len(valid_cov_assets),
        "max_weight": float(weights.max()),
        "min_weight": float(weights.min())
    })

    print(
        f"{year}: investable={len(investable_isins)}, "
        f"used_in_cov={len(valid_cov_assets)}, "
        f"max_weight={weights.max():.4f}"
    )

####################################################################################
# 7. SAVE BLOCK 2 OUTPUTS
####################################################################################

weights_by_year = pd.DataFrame(weights_rows)
weights_summary = pd.DataFrame(year_summary_rows)

weights_by_year.to_csv(OUT_DIR / "mvp_weights_by_year.csv", index=False)
weights_summary.to_csv(OUT_DIR / "mvp_weights_summary_by_year.csv", index=False)

print("- mvp_weights_by_year.csv")
print("- mvp_weights_summary_by_year.csv")

####################################################################################
# 8. BUILD MONTHLY OUT-OF-SAMPLE MVP RETURNS
####################################################################################

monthly_return_rows = []

for year in range(START_YEAR, END_YEAR + 1):
    investment_year = year + 1

    # 8.1 Get the optimal weights chosen at end of year Y
    year_weights = weights_by_year.loc[weights_by_year["screen_year"] == year].copy()

    if year_weights.empty:
        print(f"{year}: no weights available, skipping return computation")
        continue

    asset_list = year_weights["ISIN"].tolist()
    alpha = year_weights["weight"].to_numpy(dtype=float)

    # 8.2 Get monthly return columns for Jan-Dec of investment_year
    inv_cols = get_investment_year_cols(ret_month_cols, investment_year)

    if len(inv_cols) == 0:
        print(f"{year}: no monthly return columns for investment year {investment_year}")
        continue

    # 8.3 Extract monthly returns for the assets in the portfolio
    # Remaining missing returns are proxied as 0, while the delisting month
    # itself should already have been captured as -100% in cleaned returns.
    inv_ret_df = (
        ri_ret_by_isin
        .reindex(asset_list)[inv_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    inv_ret_mat = inv_ret_df.to_numpy(dtype=float)   # shape = (N assets, 12 months)

    # 8.4 Compute monthly portfolio returns with intra-year weight drift
    for k, col in enumerate(inv_cols):
        rt = inv_ret_mat[:, k]              # stock returns in month k
        rp = float(alpha @ rt)              # portfolio return in month k

        monthly_return_rows.append({
            "screen_year": year,
            "investment_year": investment_year,
            "date": pd.Timestamp(col),
            "Rp_mvp": rp
        })

        # Weight drift:
        # alpha_{i,t+1} = alpha_{i,t} * (1 + R_{i,t+1}) / (1 + R_{p,t+1})
        denom = 1.0 + rp

        if abs(denom) < 1e-12:
            # Extreme edge case: reset to equal weights if portfolio return = -100%
            alpha = np.ones_like(alpha) / len(alpha)
        else:
            alpha = alpha * (1.0 + rt) / denom

            # numerical clean-up
            alpha = np.clip(alpha, 0.0, None)
            s = alpha.sum()

            if s <= 1e-12:
                alpha = np.ones_like(alpha) / len(alpha)
            else:
                alpha = alpha / s

    print(
        f"{year}: computed monthly MVP returns for {investment_year} "
        f"using {len(asset_list)} assets"
    )

####################################################################################
# 9. BUILD FINAL RETURN SERIES
####################################################################################

mvp_monthly_returns = pd.DataFrame(monthly_return_rows)

if mvp_monthly_returns.empty:
    raise RuntimeError("No monthly MVP returns were generated.")

mvp_monthly_returns = mvp_monthly_returns.sort_values("date").reset_index(drop=True)

# save the raw monthly return series
mvp_monthly_returns.to_csv(OUT_DIR / "mvp_monthly_returns.csv", index=False)

print("- mvp_monthly_returns.csv")

print(
    f"\nReturn series covers: "
    f"{mvp_monthly_returns['date'].min().strftime('%b %Y')} "
    f"to {mvp_monthly_returns['date'].max().strftime('%b %Y')}"
)
print(f"Total months: {len(mvp_monthly_returns)}")

####################################################################################
# 10. PERFORMANCE STATISTICS
####################################################################################

# Load the return series (already in memory, but keeping it explicit)
ret_df = mvp_monthly_returns.copy()

# Set date as index
ret_df["date"] = pd.to_datetime(ret_df["date"])
ret_df = ret_df.set_index("date")

rp = ret_df["Rp_mvp"].to_numpy()

# Number of periods per year
ann = 12

# 10.1 Annualised mean return
mu_ann = rp.mean() * ann

# 10.2 Annualised volatility
vol_ann = rp.std(ddof=1) * np.sqrt(ann)

# 10.3 Sharpe ratio (no RF for now unless required)
# If you want RF later, we can add it
if rp.std(ddof=1) == 0:
    sharpe = np.nan
else:
    sharpe = (rp.mean() / rp.std(ddof=1)) * np.sqrt(ann)

# 10.4 Min / max monthly returns
ret_min = rp.min()
ret_max = rp.max()

####################################################################################
# 11. SAVE PERFORMANCE SUMMARY
####################################################################################

performance_df = pd.DataFrame([
    {
        "portfolio": "MVP_long_only_OOS",
        "start_date": ret_df.index.min().strftime("%Y-%m"),
        "end_date": ret_df.index.max().strftime("%Y-%m"),
        "n_months": len(rp),
        "ann_mean_return": mu_ann,
        "ann_volatility": vol_ann,
        "sharpe_ratio": sharpe,
        "min_monthly_return": ret_min,
        "max_monthly_return": ret_max
    }
])

performance_df.to_csv(OUT_DIR / "mvp_performance_summary.csv", index=False)

####################################################################################
# 12. PRINT RESULTS
####################################################################################

print("\n" + "="*50)
print("MVP OUT-OF-SAMPLE PERFORMANCE")
print("="*50)

print(f"Period: {performance_df['start_date'][0]} to {performance_df['end_date'][0]}")
print(f"Months: {len(rp)}")

print("\nPerformance:")
print(f"Annualised return     : {mu_ann * 100:.2f}%")
print(f"Annualised volatility : {vol_ann * 100:.2f}%")
print(f"Sharpe ratio          : {sharpe:.3f}")
print(f"Min monthly return    : {ret_min * 100:.2f}%")
print(f"Max monthly return    : {ret_max * 100:.2f}%")

print("\nSaved:")
print("- mvp_performance_summary.csv")

####################################################################################
# SECTION 2.3 - VALUE-WEIGHTED BENCHMARK
####################################################################################

MV_FILE = "cleaned_MV_monthly.csv"

# Load cleaned monthly market caps
mv = clean_headers(pd.read_csv(CLEAN_DIR / MV_FILE))
_, mv_isin_col = get_id_cols(mv)
mv_month_cols = sort_month_cols(get_time_cols(mv))
mv_by_isin = mv.set_index(mv_isin_col)

vw_rows = []

for year in range(START_YEAR, END_YEAR + 1):
    investment_year = year + 1

    investable_isins = investment_set.loc[
        (investment_set["screen_year"] == year)
        & (investment_set["investable_for_next_year"] == True),
        "ISIN"
    ].tolist()

    if len(investable_isins) == 0:
        print(f"{year}: no investable firms for VW")
        continue

    inv_cols = get_investment_year_cols(ret_month_cols, investment_year)
    if len(inv_cols) == 0:
        print(f"{year}: no investment-year columns for VW")
        continue

    for col in inv_cols:
        current_dt = pd.Timestamp(col)
        prev_month_dt = current_dt - pd.DateOffset(months=1)

        prev_candidates = [c for c in mv_month_cols if pd.Timestamp(c) == prev_month_dt]
        if len(prev_candidates) == 0:
            continue

        prev_col = prev_candidates[0]

        caps = (
            mv_by_isin.reindex(investable_isins)[prev_col]
            .apply(pd.to_numeric, errors="coerce")
        )

        rets = (
            ri_ret_by_isin.reindex(investable_isins)[col]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        valid_mask = caps.notna() & (caps > 0)
        caps = caps[valid_mask]
        rets = rets[valid_mask]

        if len(caps) == 0:
            continue

        weights_vw = caps / caps.sum()
        rp_vw = float((weights_vw * rets).sum())

        vw_rows.append({
            "screen_year": year,
            "investment_year": investment_year,
            "date": current_dt,
            "Rp_vw": rp_vw
        })

    print(f"{year}: computed VW returns for {investment_year}")

vw_monthly_returns = pd.DataFrame(vw_rows)

if vw_monthly_returns.empty:
    raise RuntimeError("No value-weighted returns were generated.")

vw_monthly_returns = vw_monthly_returns.sort_values("date").reset_index(drop=True)
vw_monthly_returns.to_csv(OUT_DIR / "vw_monthly_returns.csv", index=False)

print("\nSection 2.3 Block 1 complete.")
print("- vw_monthly_returns.csv")

####################################################################################
# COMPARE MVP VS VALUE-WEIGHTED
####################################################################################

mvp_compare = mvp_monthly_returns[["date", "Rp_mvp"]].copy()
vw_compare = vw_monthly_returns[["date", "Rp_vw"]].copy()

mvp_compare["date"] = pd.to_datetime(mvp_compare["date"])
vw_compare["date"] = pd.to_datetime(vw_compare["date"])

compare_df = pd.merge(mvp_compare, vw_compare, on="date", how="inner").sort_values("date")
compare_df = compare_df.set_index("date")

if compare_df.empty:
    raise RuntimeError("Merged comparison dataframe is empty.")

def summary_stats(series: pd.Series, name: str) -> dict:
    ann = 12
    vals = series.to_numpy(dtype=float)

    mu_ann = vals.mean() * ann
    vol_ann = vals.std(ddof=1) * np.sqrt(ann)
    sr = np.nan if vals.std(ddof=1) == 0 else (vals.mean() / vals.std(ddof=1)) * np.sqrt(ann)

    return {
        "portfolio": name,
        "start_date": series.index.min().strftime("%Y-%m"),
        "end_date": series.index.max().strftime("%Y-%m"),
        "n_months": len(vals),
        "ann_mean_return": mu_ann,
        "ann_volatility": vol_ann,
        "sharpe_ratio": sr,
        "min_monthly_return": vals.min(),
        "max_monthly_return": vals.max()
    }

mvp_stats = summary_stats(compare_df["Rp_mvp"], "MVP_long_only_OOS")
vw_stats = summary_stats(compare_df["Rp_vw"], "VW_benchmark")

comparison_summary = pd.DataFrame([mvp_stats, vw_stats])
comparison_summary.to_csv(OUT_DIR / "mvp_vs_vw_performance_summary.csv", index=False)

# Cumulative returns (saved instead of plotted)
compare_df["cum_mvp"] = (1.0 + compare_df["Rp_mvp"]).cumprod() - 1.0
compare_df["cum_vw"] = (1.0 + compare_df["Rp_vw"]).cumprod() - 1.0

compare_df.to_csv(OUT_DIR / "mvp_vs_vw_monthly_comparison.csv", index=True)
compare_df[["cum_mvp", "cum_vw"]].to_csv(
    OUT_DIR / "mvp_vs_vw_cumulative_returns.csv", index=True
)

####################################################################################
# PRINT SUMMARY
####################################################################################

print("\n" + "=" * 60)
print("MVP VS VALUE-WEIGHTED BENCHMARK")
print("=" * 60)
print(comparison_summary.to_string(index=False))

print("\nSaved:")
print("- mvp_vs_vw_performance_summary.csv")
print("- mvp_vs_vw_monthly_comparison.csv")
print("- mvp_vs_vw_cumulative_returns.csv")