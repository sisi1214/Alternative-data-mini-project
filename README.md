Purpose

Test whether a free public dataset (HK Census & Statistics Dept, C&SD, jewellery retail sales) predicts reported same-store sales growth (SSSG) of listed HK jewellers. Built as a portfolio piece for a hedge fund data analyst application, so rigour and honest limitations matter more than a strong headline.

Pre-registered hypothesis and primary test
Hypothesis: C&SD jewellery retail sales growth explains HK quarterly SSSG beyond what gold price explains.
Primary specification (fixed before results): lag 0, all complete quarters, nested F-test of SSSG ~ gold + C&SD vs SSSG ~ gold, with a Freedman-Lane permutation p-value.

Decisions and assumptions (log any change here)
COVID exclusion rule (mine, not pre-specified by the user): a quarter is excluded if it or its prior-year base window touches Feb 2023 or earlier (border fully reopened 6 Feb 2023, mask mandate ended 1 Mar 2023), i.e. quarters starting on or before Feb 2024. At lag 1 the predictor quarter is also checked.
Gold: quarterly average LBMA Gold Price PM (US$/oz) from World Gold Council Gold Demand Trends tables, Q2 2022 to Q2 2026. Q3 2026 average not found (LBMA Q3 report gives low/high only), so 2Q2026 is excluded from gold models. No 2021 averages, so gold models start at 1Q2023.
FX: constant USD/HKD 7.80 placeholder, so HKD yoy equals USD yoy (peg band effect at most about 1.3%). Replace with HKMA quarterly averages.
Permutation tests: seed 12345; 20,000 draws (stats.py), 10,000 draws (pooled.py). A printed p of 0.000 means below about 1/B, report as "< 0.001".
Pooled model: common slopes, company fixed effects, SE clustered by quarter, block permutation by quarter. Effective n is the number of quarters, not company-quarters.
Press-added CTF values (not user-supplied, kept as a separate sensitivity): 3Q2025 +14.3 (confirmed), 4Q2025 +40.1 (read from a chart label, unconfirmed), 1Q2026 +41.7 (confirmed).

Results 
Luk Fook, lag 0, n = 10 (2Q2023 to 1Q2026): r = 0.98 (95% CI 0.93 to 1.00), SSSG about 1.4x C&SD yoy. Adding C&SD to gold-only: R2 0.13 to 0.97, p < 0.001. Gold adds nothing beyond C&SD (p = 0.27). Leave-one-out beta 1.34 to 1.50, no sign flips.
Ex-COVID (n = 8): r = 0.99, beta 1.58.
Lag 1: no reliable lead. Ex-COVID, C&SD adds delta R2 0.05 (p = 0.17); lagged gold carries the signal.
Nowcast (first two months): leave-one-out RMSE 9.6pp vs 32.8pp gold-only; expanding window (5 test quarters) 10.2pp vs 29.1pp training mean. Live 2Q2026: Jul-Aug C&SD +15.7% implied SSSG about +20%; reported +13%.
CTF, lag 0, n = 15: r = 0.98, beta 1.2. Pooled beta 1.25, p < 0.001; ex-COVID pooled still p = 0.002; pooled lag 1 ex-COVID C&SD adds nothing (p = 0.24).
Luk Fook and CTF SSSG correlate 0.97 over 10 shared quarters: they are almost one signal, so pooling adds little independent evidence.
