# Nightly log

- 2026-10-23: set up nightly runs. Temporal aggregation of per-engine predictions: no gain (CV RMSE within +-0.05 on all subsets).
- 2026-10-23: LightGBM 5-seed bagging: FD001 CV RMSE 9.49 -> 9.33, FD004 20.89 -> 20.87 (5x training cost), judged not worth it; kept LGB_SEEDS = 1.
