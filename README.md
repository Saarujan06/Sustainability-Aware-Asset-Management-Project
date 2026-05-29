# Sustainability Aware Asset Management
# Portfolio Allocation with a Carbon Objective

Carbon-aware portfolio construction on an AMER + EUR equity universe, January 2014 to
December 2025, using Scope 1 emissions. Builds and evaluates six portfolios:

| Symbol            | Description                                                         |
|-------------------|---------------------------------------------------------------------|
| `P(vw)`           | Value-weighted benchmark (monthly rebalanced, per spec §2.3)        |
| `P(mv)_oos`       | Long-only minimum-variance portfolio (annual rebalance + drift)     |
| `P(mv)_oos(0.5)`  | MVP with carbon footprint cut by 50% vs unconstrained MVP           |
| `P(vw)_oos(0.5)`  | Tracking-error-minimising portfolio with CF ≤ 0.5·CF^(vw)           |
| `P(vw)_oos(NZ)`   | Net-zero portfolio: CF contracts by 10%/year from a 2013 baseline   |

The full methodology, results and discussion are in the accompanying report
(`SAAM_report.pdf`); this README documents the notebook only.

---

## 1. Requirements

- Python ≥ 3.10
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  The pinned set is `pandas`, `numpy`, `matplotlib`, `openpyxl`. The notebook also
  uses `cvxpy` (with the bundled OSQP solver), so add it if not already present:
  ```bash
  pip install cvxpy
  ```
- A Jupyter front-end (JupyterLab, classic Notebook, or VS Code).

## 2. Folder layout

```
.
├── Group_AS_SAAM.ipynb          # the analysis notebook
├── requirements.txt
├── Data_2026/                   # raw Datastream Excel files (input)
│   ├── Static_2025.xlsx
│   ├── DS_RI_T_USD_M_2025.xlsx
│   ├── DS_MV_T_USD_M_2025.xlsx
│   ├── DS_CO2_SCOPE_1_Y_2025.xlsx
│   ├── DS_REV_Y_2025.xlsx
│   └── Risk_Free_Rate_2025.xlsx
├── cleaned_data/                # produced by §2 of the notebook
│   ├── cleaned_static.csv
│   ├── cleaned_RI_monthly_returns.csv
│   ├── cleaned_MV_monthly.csv
│   ├── cleaned_CO2_scope1_yearly.csv
│   ├── cleaned_REV_yearly.csv
│   └── investment_set_by_year.csv
└── results/                     # produced by §3–§5 of the notebook (CSVs + PNGs)
```

`cleaned_data/` and `results/` are created automatically if missing.

## 3. How to run

```bash
jupyter lab Group_AS_SAAM.ipynb
```

Then **Run All**. End-to-end runtime is a few minutes on a laptop; the QP solves
(one per rebalance year × portfolio) are the slow part. Cells are intended to be
executed in order, top to bottom.

## 4. What the notebook does, section by section

| Notebook section | Spec ref | What it builds |
|---|---|---|
| §1 Imports & config         | §0        | Constants: `REGIONS=["AMER","EUR"]`, `START_YEAR=2013`, `END_YEAR=2024`, `WINDOW_YEARS=10`, `STALE_THRESHOLD=0.50`, `MIN_VALID_MONTHS=120`, `COV_METHOD="ledoit_wolf"`. |
| §2 Data cleaning            | §1     | Loads raw `Data_2026/*.xlsx`, drops empty rows, detects delistings, screens by region, computes monthly simple returns, writes `cleaned_data/*.csv`. |
| §3 Investment set           | §2.1     | For each screen year Y, marks firms `investable_for_next_year` (≥36 valid months in trailing 10y, monthly stale share < 50%, non-missing Y-end Cap and Scope 1 + Revenue). |
| §4 Minimum-variance portfolio | §2.2   | Annual rebalance + monthly drift. Covariance estimated via Ledoit–Wolf shrinkage on the 120-month complete-case subset. Outputs `mvp_*.csv`. |
| §5 VW benchmark             | §2.3     | Monthly-rebalanced VW over the investable set. Outputs `vw_monthly_returns.csv`. |
| §6 MVP vs VW                | §2.3     | Out-of-sample performance comparison. Outputs `mvp_vs_vw_*.csv`, `cumulative_returns_plot.png`. |
| §7 Covariance robustness    | §2.4     | Reruns the MVP under sample, single-factor, and Ledoit–Wolf covariance. Outputs `robustness_*.csv/png`. |
| §8 Carbon emissions (§3.1)  | §3.1     | WACI and CF of P(mv) and P(vw). Top-10 contributor tables. |
| §9 P(mv)_oos(0.5)            | §3.2     | CF-constrained MVP: `cf_limit = 0.5 × CF_MVP`. Outputs `mvp05_*.csv`. |
| §10 P(vw)_oos(0.5)           | §3.3     | Tracking-error QP with `cf_limit = 0.5 × CF^(vw)`. Outputs `te05_*.csv`. |
| §11 Four-portfolio summary  | §3.4     | Side-by-side performance and carbon profile. Outputs `section34_*.csv/png`. |
| §12 Net-Zero (§4.1)         | §4.1     | TE-QP with annual CF target = 0.9^(Y − 2013) × CF_2013^(vw). Outputs `nz_*.csv`. |
| §13 Three-VW comparison     | §4.2     | P(vw) vs P(vw)_oos(0.5) vs P(vw)_oos(NZ). Outputs `section42_*.csv/png`. |

## 5. Methodology — key choices

- **Two universes per QP year, by design.** The investable set (≥36 months of
  trailing data, per spec §2.1) defines `CF^(vw)`. The optimisation runs on the
  *complete-case* 120-month subset of that universe — the Ledoit–Wolf estimator
  requires no missing values. So:
  - `cf_limit` is computed over the full investable set (spec §3.3).
  - The QP decision variables and `α^(vw)` are over the complete-case subset.
- **Covariance estimator:** Ledoit–Wolf (2003, 2004) single-index shrinkage,
  shrinking the sample covariance toward a CAPM-target. Toggle via `COV_METHOD` in
  §1: `"sample"`, `"single_factor"`, or `"ledoit_wolf"`.
- **Carbon constraint:** `CF^(p) = Σ_i α_i · E_i / Cap_i`, linear in α and applied
  directly to the QP. A multiplicative `(1 − 10⁻⁶)` safety factor is applied before
  the solve so the constraint still binds after the post-solve clip-and-renormalise
  step.
- **QP solver:** OSQP via `cvxpy`, with `eps_abs = eps_rel = 1e-8`, `max_iter = 50_000`.
- **Buy-and-hold drift:** weights are rebalanced once at the start of each
  investment year, then drift with realised returns for the next 12 months
  (`drift()` helper).

## 6. Outputs

All CSVs and PNGs land in `results/`. Highlights:

- **Per-portfolio returns:** `mvp_monthly_returns.csv`, `vw_monthly_returns.csv`,
  `mvp05_monthly_returns.csv`, `te05_monthly_returns.csv`, `nz_monthly_returns.csv`.
- **Per-portfolio weights:** `*_weights_by_year.csv` (long format: screen_year, ISIN, weight).
- **Per-portfolio diagnostics:** `*_summary_by_year.csv` (CF target, CF realised,
  binding flag, max weight, effective N, realised TE where applicable).
- **Comparison tables and plots:** `mvp_vs_vw_*`, `mvp_vs_mvp05_*`, `vw_vs_te05_*`,
  `vw_vs_nz_*`, `section34_*` (four-portfolio summary), `section42_*` (three-VW summary).
- **Robustness:** `robustness_covariance_comparison.csv`,
  `robustness_covariance_cumulative.png`, `robustness_rolling_variance.png`.

## 7. Reproducibility notes

- The notebook is deterministic: no random seeds are used; OSQP converges to the
  same optimum given the same inputs.
- All file paths are relative to the project root (`BASE_DIR = Path.cwd()`), so
  the notebook must be launched from this directory.
- Excel files use `openpyxl` as the engine; if you regenerate them from
  Datastream, keep the same column conventions (firms × months with ISIN/NAME
  as identifier columns).

## 8. Known caveats

- Scope 1 only — Scope 2 + Scope 3 would produce tighter, costlier constraints.
- 2014–2025 was a strong period for low-carbon mega-caps; the near-zero cost of
  decarbonisation is partly sample-specific. See §6 and §8 of the report.
- The optimisation universe (120-month complete-case) is narrower than the
  investable set; survivorship in this subset slightly tilts the matched-universe
  VW benchmark vs. the spec's monthly VW. The notebook reports the spec VW in §5.

## 9. Files

- `Group_AS_SAAM.ipynb` — the notebook (all code lives here).
- `requirements.txt` — Python dependencies (add `cvxpy` if not present).
- `Data_2026/` — raw input data.
- `cleaned_data/`, `results/` — generated; safe to delete and regenerate.
- `SAAM_Project_2026.pdf` — project brief (the spec).
- `Groups_Strategy_2026.pdf` — group/region/scope assignment.