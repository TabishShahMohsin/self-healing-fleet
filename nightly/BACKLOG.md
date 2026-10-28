# Improvement backlog (status: todo / running / accepted / rejected)

Gate: mean CV RMSE over FD001-FD004 must improve by >= 0.15, no subset worse by > 0.3, critical-window RMSE not worse by > 0.3.
Current champion: see `nightly/champion.json`.

| # | Idea | How | Status |
|---|---|---|---|
| 1 | LightGBM seed bagging (5 seeds) | `LGB_SEEDS` in run_final.py, `seeds=5` in cv_lgb | rejected: FD001 9.49 -> 9.33, FD004 20.89 -> 20.87 for 5x the cost (FD002/FD003 not run) |
| 2 | Baseline window length for the per-engine delta (10 / 25 / 40 cycles instead of 15) | parameterise `engine_feats` | accepted: window 40 (bw=10: +0.15 worse; bw=25: -0.14; bw=40: -0.19 CV RMSE mean, ens -0.19); test RMSE ens 9.6 / 16.4 / 10.0 / 20.4 |
| 3 | LightGBM random search on FD002/FD004 (num_leaves, min_child_samples, colsample, reg_lambda, learning rate) | 12 random configs, then gate the best | rejected: best of 12 (N3_rs05: leaves 31, min_child 150, 500 trees) CV RMSE mean -0.09 (FD001 9.52->9.47, FD002 19.70->19.50, FD003 8.91->8.94, FD004 20.67->20.53), below the 0.15 gate; other 11 configs worse on FD002/FD004 mean |
| 4 | Extra diverse ensemble member: ExtraTreesRegressor or HistGradientBoosting on the same features | OOF-blend weight tuned like the CNN | rejected: 50/50 LightGBM+ExtraTrees (150 trees, leaf 5) CV RMSE mean +0.26 worse (FD001 9.52->10.00, FD002 19.70->19.80, FD003 8.91->9.36, FD004 20.67->20.66), critical window +1.17 worse; HistGradientBoosting not tried |
| 5 | Per-engine noise features: early-life std and slope of each sensor (first 30 cycles) | new feature group "early" | rejected: CV RMSE mean +1.82 worse (FD001 9.52->11.40, FD002 19.70->21.74, FD003 8.91->10.35, FD004 20.67->22.59), critical window +2.14; per-engine constants let trees memorise engines |
| 6 | Predict log(RUL+1) instead of RUL (relative error near end of life) | transform target in cv_lgb, invert | rejected: CV RMSE mean +0.07 (FD001 9.52->9.27, FD002 19.70->19.65, FD003 8.91->8.88, FD004 20.67->21.29, worst subset +0.62), though critical-window RMSE improved a lot (mean -1.89; FD002 14.39->10.93, FD004 17.56->13.95); fails the overall gate. Possible follow-up: use only as a critical-window specialist | |
| 7 | Drop the noisiest sensors per subset (by early-life SNR) | feature selection inside CV | todo |
| 8 | Quantile LightGBM (5% / 95%) as an alternative to conformal bins; compare interval width at equal coverage | new interval method | todo |
| 9 | Retune triage thresholds to cut false P1 alarms on FD002/FD004 while keeping every urgent engine flagged | edit `tune_agent` objective | todo |
| 10 | CNN: 2-3 seeds averaged, window 50, critical-window weight | exp_cnn.py (slow) | todo |

## Already tried (do not repeat)
- Running min/max of deltas; EWMA 0.15 / 0.08; slow-trend features; regime id as a feature; per-regime EWMA state; healthy-reference normalisation.
- RUL cap 115 / 135 (125 is best); slower or bigger LightGBM; CNN without age channel; blending the CNN that has no age channel.
- Temporal aggregation (averaging p(t-j) - j over the last 5-50 cycles of each engine): no gain, errors are persistent per engine rather than per-cycle noise.
