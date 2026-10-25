# Nightly log

- 2026-10-23: set up nightly runs. Temporal aggregation of per-engine predictions: no gain (CV RMSE within +-0.05 on all subsets).
- 2026-10-23: LightGBM 5-seed bagging: FD001 CV RMSE 9.49 -> 9.33, FD004 20.89 -> 20.87 (5x training cost), judged not worth it; kept LGB_SEEDS = 1.
- 2026-10-24: baseline window for delta features 15 -> 40 cycles (windows 10/25/40 tried, CV only). LGB CV RMSE mean -0.186 (FD002 20.33 -> 19.70, FD004 20.89 -> 20.67, FD001/FD003 +0.03/+0.08); ensemble CV mean -0.192 (9.18/18.80/8.35/20.37 -> 9.12/18.36/8.28/20.17). ACCEPTED. Full run: test RMSE ens 9.63 / 16.42 / 10.02 / 20.44.
- 2026-10-25: LightGBM random search (12 seeded configs on FD002/FD004, best then run on all four). Best N3_rs05 CV RMSE mean -0.090 (9.52/19.70/8.91/20.67 -> 9.47/19.50/8.94/20.53). REJECTED (gate needs -0.15). Defaults kept.
