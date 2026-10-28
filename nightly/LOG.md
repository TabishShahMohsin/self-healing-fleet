# Nightly log

- 2026-10-23: set up nightly runs. Temporal aggregation of per-engine predictions: no gain (CV RMSE within +-0.05 on all subsets).
- 2026-10-23: LightGBM 5-seed bagging: FD001 CV RMSE 9.49 -> 9.33, FD004 20.89 -> 20.87 (5x training cost), judged not worth it; kept LGB_SEEDS = 1.
- 2026-10-24: baseline window for delta features 15 -> 40 cycles (windows 10/25/40 tried, CV only). LGB CV RMSE mean -0.186 (FD002 20.33 -> 19.70, FD004 20.89 -> 20.67, FD001/FD003 +0.03/+0.08); ensemble CV mean -0.192 (9.18/18.80/8.35/20.37 -> 9.12/18.36/8.28/20.17). ACCEPTED. Full run: test RMSE ens 9.63 / 16.42 / 10.02 / 20.44.
- 2026-10-25: LightGBM random search (12 seeded configs on FD002/FD004, best then run on all four). Best N3_rs05 CV RMSE mean -0.090 (9.52/19.70/8.91/20.67 -> 9.47/19.50/8.94/20.53). REJECTED (gate needs -0.15). Defaults kept.
- 2026-10-26: ExtraTrees blended 50/50 with LightGBM as a diverse tree member. CV RMSE mean +0.257 (9.52/19.70/8.91/20.67 -> 10.00/19.80/9.36/20.66), critical window +1.17. REJECTED. Defaults kept.
- 2026-10-27: early-life noise features (std and slope of each sensor over first 30 cycles). CV RMSE mean +1.820 (9.52/19.70/8.91/20.67 -> 11.40/21.74/10.35/22.59), critical window +2.14. REJECTED. Defaults kept.
- 2026-10-28: predict log(RUL+1) instead of RUL. CV RMSE mean +0.071 (9.52/19.70/8.91/20.67 -> 9.27/19.65/8.88/21.29, FD004 +0.62), critical window -1.889 (mean). REJECTED (overall RMSE gate). Defaults kept.
