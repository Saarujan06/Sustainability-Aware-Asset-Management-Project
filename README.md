# Sustainability Aware Asset Management
## Project: Portfolio Allocation with a Carbon Objective

---
 
The goal of this project is to implement some climate aware asset management concepts seen in class.
 
- The first part of the project consists of building a portfolio based on the minimum-variance criterion (deadline: April 12 at midnight, for submitting preliminary results).
- The second part focuses on the climate impact of the portfolio (deadline: May 29 at midnight, for submitting the complete package).
 
You are given a certain freedom within the framework, provided you explain the methodology that seems the most relevant to you. The project includes some empirical work that must be done with Python or Matlab.
 
There will be a preliminary presentation of the first part of the project (April 14). Details on the deliverables of the project are provided at the end of this paper.
 
---
 
## Data Information
 
You must use the strategy assigned to your group, i.e., the region and the climate strategy. Projects provided after the deadline will not be graded.
 
In the **"Static.xlsx"** file, you are given the following information for 2545 firms with carbon data:
 
- ISIN code
- Name of the company
- Country and region
 
You are then given additional files containing time series of carbon data and market values:
 
- Scope 1 and Scope 2 CO2 emissions (annual) (in tonnes) $(E_{i,Y})$ (`DS_CO2_SCOPE_1.xlsx` and `DS_CO2_SCOPE_2.xlsx`)
- Revenues (annual) (in thousands USD) $(Rev_{i,Y})$ (`DS_REV_USD_Y.xlsx`)
- Market capitalization (end of year) (in million USD) $(Cap_{i,Y})$ (`DS_MV_T_USD_Y.xlsx`)
- Market capitalization (end of month) (in million USD) $(Cap_{i,t})$ (`DS_MV_T_USD_M.xlsx`)
- Total return index (price index, including dividend payments) (end of year) (in USD) $(P_{i,Y})$ (`DS_RI_T_USD_Y.xlsx`)
- Total return index (price index, including dividend payments) (end of month) (in USD) $(P_{i,t})$ (`DS_RI_T_USD_M.xlsx`)
 
**Additional important information:**
 
- The data is collected from end 1999 to end 2025 for all firms and variables. When the data is missing, Datastream reports a blank. For market values (RI and MV), the data covers in principle 1999 to 2025. For carbon data, at best, the data covers 2002 to 2024 (last year available); however, coverage is limited before 2010. For this reason, we start the allocation exercise at the end of 2013.
 
- Price data are collected at the monthly frequency. Prices should be converted into returns using the simple return definition. Returns at date $t$ are collected in the vector $R_t = \{R_{1,t}, \ldots, R_{N,t}\}$, where $N$ denotes the number of firms. The 10 years of data from 2004 to 2013 are used to initialize the calculation of expected returns and the covariance matrix. See Section 2.1 below.
 
- Revenues and CO2 emissions are updated annually. Some data might be missing. When the missing value is between two available years or at the end of the sample, just use the number from the previous year. When the missing value is at the beginning of the sample, you cannot invest in this firm until numbers are made available. The allocation is based on information available at the end of the year (carbon and return data) for the next year. The portfolio is rebalanced once per year; performance is computed month by month.
 
- It is important to distinguish the frequency of returns (monthly) from the frequency of rebalancing (annual). The investment decision is made at the end of a given year for the next year. However, expected returns and the covariance matrix must be estimated from monthly returns. To avoid confusion, we use the index $t$ for monthly variables and the index $Y$ for annual variables. Using monthly returns also allows computation of key portfolio characteristics (volatility, Sharpe ratio, etc.).
 
---
 
# Part I - Standard Portfolio Allocation
 
## 1 Data Cleaning
 
The dataset provided is the direct output from Datastream. Some cleaning is necessary to design an investment strategy that makes sense.
 
- **Missing prices:** For some ISINs, Datastream could not find any matching firm. For instance, in Datastream, the ISIN code is associated with share class B, while the request is based on share class A. For some ISINs, Datastream could not find any associated data (e.g., the firm does not report carbon emissions). In these cases, the full row is missing. You should delete it from all tables.
 
- **Missing values:** There are three types of missing data, at the beginning, in the middle, and at the end of the sample.
 
  - Missing values at the beginning of the sample can be explained by the firm not being listed yet, or not reporting the data. There is nothing we can do.
 
  - Missing values around the end of the sample usually correspond to a firm default or delisting. In general, when a firm is delisted, the delisting date is appended to the firm name in Datastream. In such cases, you should acknowledge that the price goes to 0 at the time of delisting, implying a realized return of $-100\%$.
 
  - Missing values between two available values typically correspond to misreporting or a failure to report. I suggest using the number from the previous year.
 
- **Low prices:** For some firms, the total return index (RI) can be very low (e.g., below 0.5) or even equal to 0 (due to rounding for values below 0.05). In such cases, computing returns can result in extreme (or infinite) numbers. I suggest treating all prices below 0.5 as missing values. This has implications for investing:
 
  - If the price is missing at the end of year $Y$, you do not invest in this firm in year $Y+1$ (exclude the firm from your investment set).
  - If the price is not missing at the end of year $Y$ but is missing at the end of year $Y+1$, you may have invested in the firm and will suffer a large loss during year $Y+1$.
 
- **Stale prices:** For some firms, the price does not vary for several months or years (no trades). In theory, this is not an issue (returns are 0). In practice, it is an issue because volatility is artificially low, and you may end up investing an excessive share of wealth in illiquid firms, potentially producing a near-zero estimated portfolio volatility. I suggest proceeding as follows:
 
  - For all firms in your investment set, compute the proportion of months (over the 10-year estimation window up to end of year $Y$) with a return equal to 0 (no price change).
  - If this proportion exceeds a fixed threshold (e.g., 50%), treat the firm as subject to stale prices and exclude it from investment. (There is no look-ahead bias, as the decision is based on past information only.)
 
---
 
## 2 Minimum-Variance Portfolio Allocation
 
### 2.1 Investment Set
 
To construct a portfolio, we start by defining our investment set. It is determined by the region assigned to the group. As we construct portfolios out of sample, we use only past data to compute return moments. For instance, we use 10 years of monthly returns (for instance, from Jan. 2004 to Dec. 2013) to compute the vector of expected returns and the covariance matrix. Therefore, we have $\tau = 10 \times 12 = 120$ months of data available to compute the expected returns and the covariance matrix used for the allocation decided at the end of Dec. 2013 (for implementation over 2014). Dec. 2013 corresponds to year $Y_0 = 2013$ for annual data and to $t_0 = \tau = 120$ for monthly data. Dec. 2014 corresponds to $Y_0 + 1 = 2014$ and $t_0 + 12 = 132$. We associate year $Y$ to month $t$ (December of year $Y$).
 
Therefore, at the end of year $Y$, we define the investment set as the list of firms we can possibly invest in. This should include firms satisfying the following criteria:
 
- Firms belong to the assigned region.
- Firms have a sufficient number of return observations to estimate expected returns and the covariance matrix. If a firm's returns contain too many missing values over the last 10 years (e.g., less than 3 years of available monthly returns), exclude the firm from the investment set.
- **Importantly, because we want to use the same investment set for both parts of the project, exclude firms without carbon data available at the end of year $Y$.**
 
We compute the expected returns as the average over the 10 years up to the end of year $Y$:
 
$$\hat{\mu}_Y = \frac{1}{\tau} \sum_{k=0}^{\tau-1} R_{t-k}$$
 
The covariance matrix is
 
$$\Sigma_Y = \frac{1}{\tau} \sum_{k=0}^{\tau-1} (R_{t-k} - \hat{\mu}_Y)'(R_{t-k} - \hat{\mu}_Y)$$
 
---
 
### 2.2 Minimum Variance Portfolio
 
We construct the minimum variance portfolio out of sample. The allocation is determined at the end of year $Y$ for year $Y+1$. In general, the covariance matrix $\Sigma_Y$ is not invertible, so the closed-form optimal weight formula will not work. Instead, we solve the following optimization problem, restricting weights to be non-negative:
 
$$\min_{\{\alpha_Y\}} \quad \sigma^2_{p,Y} = \alpha'_Y \Sigma_Y \alpha_Y$$
 
$$\text{s.t.} \quad \alpha'_Y e = 1 \qquad e = (1, \ldots, 1)'$$
 
$$\alpha_{i,Y} \geq 0 \quad \text{for all } i$$
 
Then, we roll the window by one year and iterate until the end of the sample, so that the portfolio is rebalanced every year from Dec. 2013 to Dec. 2024.
 
We compute the ex-post performance of the portfolio. The portfolio return can be computed each month of year $Y+1$ using monthly stock returns of year $Y+1$ and the optimal weights calculated at the end of year $Y$. This gives: $R_{p,t+k} = \alpha'_{t+k-1} R_{t+k}$, for $k = 1, \ldots, 12$, where $\alpha_{i,t+k-1} = \alpha_{i,t+k-2} \times (1 + R_{i,t+k-1})/(1 + R_{p,t+k-1})$, with $\alpha_t = \alpha_Y$. We thus obtain a time series of ex-post portfolio returns: $\{R_{p,\tau+1}, \ldots, R_{p,T}\}$, where $T = 144$ denotes the total number of months in the sample (from 2014 to 2025).
 
Compute the characteristics of this portfolio (denoted $P^{(mv)}_{oos}$) over the sample: annualized average return ($\bar{\mu}_p$), annualized volatility ($\sigma_p$), Sharpe ratio ($SR_p$), minimum, and maximum.
 
> **Remark:** The reason for considering a long-only portfolio is to facilitate interpretation of the portfolio's carbon footprint.
 
---
 
### 2.3 Comparison with the Value-Weighted Portfolio
 
We compare these properties to those of the value-weighted portfolio (denoted $P^{(vw)}$, the *benchmark*). The performance of the value-weighted portfolio is:
 
$$R^{(vw)}_{t+1} = \sum_{i=1}^{N} w_{i,t} R_{i,t+1}$$
 
where $w_{i,t} = Cap_{i,t} / \sum_{j=1}^{N} Cap_{j,t}$ denotes the relative market capitalization of firm $i$ at the end of month $t$.
 
In particular, plot the cumulative return series of both strategies and compare their summary statistics.
 
---
 
# Part II - Portfolio Allocation with Carbon Emission Reduction
 
## 3 Allocation with a 50% Reduction in Carbon Emissions
 
We now add a layer in the portfolio construction by taking into account the CO$_2$ emissions associated to the portfolio. The scope of the CO$_2$ emissions is given by the strategy assigned to the group.
 
### 3.1 Carbon Emissions
 
We consider the carbon intensity ($CI_{i,Y}$) of all firms in our investment set. It is computed in tonnes of $CO_2$ equivalent per million U.S. dollars of revenue. Be careful: in the dataset, $CO_2$ emissions are in tonnes, but revenue is in thousands, not in millions. So, you should divide the $Rev_{i,Y}$ series by 1000 before computing the carbon intensity.
 
We compute the weighted-average carbon intensity and the carbon footprint of the portfolio as the amount of annual carbon emissions that is attributed to the investor per million U.S. dollars invested:
 
$$WACI^{(p)}_Y = \sum_{i=1}^{N} \alpha_{i,Y} CI_{i,Y} \qquad CF^{(p)}_Y = \frac{1}{V_Y} \sum_{i=1}^{N} o_{i,Y} E_{i,Y}$$
 
where $o_{i,Y} = V_{i,Y} / Cap_{i,Y}$ measures the fraction of the equity of the firm owned by the portfolio, with $V_{i,Y} = \alpha_{i,Y} V_Y$ the dollar value invested in firm $i$ and $V_Y = \sum_{i=1}^{N} V_{i,Y}$ the dollar value of the portfolio. Compute the carbon footprint of $P^{(+)}_{oos}$ for every year in the sample, assuming a starting wealth equal to $V_{2013} = 1$ million U.S. dollars.
 
Compute the WACI and CF of the value-weighted portfolio. Plot and comment on these indicators. Which firms drive the carbon intensity up (e.g., top 10; report firm names along with ISIN codes)?
 
---
 
### 3.2 Long-only Portfolio with a Carbon Footprint Objective
 
We now construct an optimal long-only portfolio with a carbon footprint 50% below the carbon footprint of the optimal long-only portfolio $P^{(mv)}_{oos}$ determined in Section 2.2 (*minimum variance active investor*). For every year in the sample, compute the optimal weights of the long-only portfolio with a carbon footprint lower or equal to $0.5 \times$ the carbon footprint of $P^{(mv)}_{oos}$. This can be done by solving:
 
$$\min_{\{\alpha_Y\}} \quad \sigma^2_{p,Y} = \alpha'_Y \Sigma_Y \alpha_Y$$
 
$$\text{s.t.} \quad CF^{(p)}_Y \leq 0.5 \times CF^{(P^{(mv)}_{oos})}_Y$$
 
$$\alpha'_Y e = 1$$
 
$$\alpha_{i,Y} \geq 0 \quad \text{for all } i$$
 
We call this portfolio $P^{(mv)}_{oos}(0.5)$. Compute the characteristics of this out-of-sample portfolio over the sample. As before, plot the cumulative return series of both strategies and compare summary statistics. Plot and comment the evolution of the WACI and CF of the portfolio. Provide details on the main changes regarding the composition of the portfolio (for instance, excluded or over-weighted firms or sectors).
 
---
 
### 3.3 Tracking Error Minimization
 
Another interesting decarbonization strategy consists in designing the portfolio that stays as close as possible to the benchmark, while reducing the carbon footprint by 50% (*otherwise passive investor*). This is done by solving the tracking error each year:
 
$$\min_{\{\alpha_Y\}} \quad TE_{p,Y} = \sqrt{(\alpha_Y - \alpha_Y^{(vw)})' \Sigma_Y (\alpha_Y - \alpha_Y^{(vw)})}$$
 
$$\text{s.t.} \quad CF_Y^{(p)} \leq 0.5 \times CF_Y^{(P^{(vw)})}$$
 
$$\alpha'_Y e = 1$$
 
$$\alpha_{i,Y} \geq 0 \quad \text{for all } i$$
 
where $CF^{(P^{(vw)})}_Y = \frac{1}{Cap_Y} \sum_{i=1}^{N} E_{i,Y}$ denotes the carbon footprint of the value-weighted portfolio, with $Cap_Y = \sum_{i=1}^{N} Cap_{i,Y}$ the total market value of the investment set.
 
We call this portfolio $P^{(vw)}_{oos}(0.5)$. Compute the characteristics of the portfolio over the sample. Again, plot the cumulative return series of both strategies and compare summary statistics.
 
---
 
### 3.4 Comparison of Portfolios
 
Comment on the trade-off between the financial performance of the portfolio and the reduction in its carbon footprint. Elaborate on the difference between portfolios $P^{(mv)}_{oos}$ and $P^{(mv)}_{oos}(0.5)$ and between portfolios $P^{(vw)}_{oos}$ and $P^{(vw)}_{oos}(0.5)$.
 
---
 
## 4 Portfolio Allocation with a Net Zero Objective
 
Finally, we want to construct a minimum variance portfolio, while cumulatively reducing its carbon emissions.
 
### 4.1 Net Zero Portfolio
 
We implement a decarbonization strategy in which the carbon footprint of the portfolio is reduced by $\theta = 10\%$ per year every year from Dec. 2013 to Dec. 2024.
 
We adopt the point of view of the otherwise passive investor. The optimization problem is the same as in Section 3.3 except that the carbon emissions reduction constraint is now defined as
 
$$CF^{(p)}_Y \leq (1 - \theta)^{Y - Y_0 + 1} \times CF^{(P^{(vw)})}_{Y_0} \qquad \text{for } Y = 2013, \cdots, 2024$$
 
with $Y_0 = 2013$.
 
We call this portfolio $P^{(vw)}_{oos}(NZ)$. Compute the characteristics of this portfolio over the sample. Again, plot the cumulative return series of both strategies and compare summary statistics.
 
### 4.2 Comparison of Portfolios
 
Compare the cumulative performance of the three portfolios $P^{(vw)}_{oos}$, $P^{(vw)}_{oos}(0.5)$, and $P^{(vw)}_{oos}(NZ)$. Comment on the possible cost of constructing a net zero portfolio.
 
---
 
## 5 Deliverables
 
### 5.1 Preliminary Results (April 12 at midnight)
 
Using the **"Template for Part I-SAAM.xlsx"**, available on Moodle, you report your results associated with Part I - Standard Portfolio Allocation. This includes summary statistics associated with the value-weighted portfolio and the global minimum variance portfolio. You also report the monthly returns of both portfolios from 2014 to 2025.
 
---
 
### 5.2 Final Results (May 29 at midnight)
 
You must submit a single folder containing:
 
1. **Report (PDF).** The report must include: (i) a clear description of the implementation (data cleaning, investment set construction, estimators, optimization problems, and rebalancing logic); (ii) results presented in well-labeled tables and figures; (iii) interpretation and discussion of the financial and carbon outcomes; and (iv) suggestions for improvements or robustness checks; (v) five bullet points discussing limitations issues (e.g., data quality, parameter sensitivity and estimation issues, constraint feasibility, etc.).
 
   The report must be no longer than 30 pages (including tables and figures; references and appendices may be included within this limit). It must be written in a professional report style (not a notebook listing), with clear structure, coherent writing, and well-motivated methodological choices. Tables and figures must be clean and readable, with labels and captions. Each table/figure must be self-contained: include a brief note explaining what is reported, how it is constructed, and how it should be interpreted.
 
2. **Sales pitch (PDF, 1 page max).** Write a short, professional sales pitch that could be delivered by an asset manager to a client (e.g., a pension fund or a wealth management client) to promote decarbonization and net-zero strategies. Your pitch should: (i) clearly state the client objective and why it matters; (ii) explain the strategy in simple terms; (iii) summarize the key evidence from your results (financial and carbon performance), using 2 or 3 numbers; (iv) acknowledge trade-offs and risks (sector tilts, constraint feasibility, data limitations, etc.) and how they are managed.
 
3. **Final notebook (one notebook).** A single notebook (`.ipynb`) that reproduces all results shown in the report (tables and figures) when run from top to bottom. It must run without manual intervention: no hard-coded local paths, no missing dependencies, no hidden steps. The grader should be able to (i) open the notebook, (ii) run all cells, and (iii) obtain the report outputs without any additional effort.
 
4. **Video presentation (10 minutes max).** A short video presenting the project. It should cover: the main idea and objectives, a brief methodology overview (high level), a focus on results (financial performance and carbon metrics), and a discussion of key issues/limitations and takeaways. The video should be clear, well-structured, and aimed at a professional audience.
 
---
 
### 5.3 Evaluation of the Project
 
The project grade will be based on the different deliverables.
 
#### Peer Evaluation (within-group contribution)
 
Because the project is completed in groups, each student will complete an individual peer evaluation at the end of the project. The goal is to (i) measure individual contribution and ensure fair grading within the group, and (ii) document each member's work effort and overall contribution to the project deliverables.
 
- **Confidential individual form.** Each student will complete a short, confidential evaluation form for their group members (and a self-evaluation). Only the instructor and teaching staff will have access to these evaluations.
 
- **Single required metric: % contribution.** For each group member, you must allocate a percentage reflecting that person's overall contribution to the project deliverables. The percentages must sum to 100%.
 
- **Supporting comments (short).** The evaluation form will include space for brief comments to support your assessment. Providing evidence is not required, but comments should be specific and professional.
 
- **Impact on individual grades.** The default is that all members receive the same project grade. Peer evaluation may lead to individual grade adjustments:
  - Students with substantially lower contribution/effort than the group average may receive a reduced grade.
  - Students with substantially higher contribution/effort may receive a modest increase.
 
- **Consistency checks.** The teaching staff may follow up in case of inconsistent evaluations (e.g., strong disagreement within a group) or if the pattern suggests free-riding. This may include a brief interview with one or more group members.
 
- **Good faith requirement.** Peer evaluations must be completed honestly and professionally. Misreporting, retaliatory ratings, or coordination to manipulate outcomes may be treated as misconduct and can lead to sanctions.
