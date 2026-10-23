# Improvement backlog (status: todo / running / accepted / rejected)

Gate: mean CV RMSE over FD001-FD004 must improve by >= 0.15, no subset worse by > 0.3, critical-window RMSE not worse by > 0.3.
Current champion: see `nightly/champion.json`.

| # | Idea | How | Status |
|---|---|---|---|
| 1 | LightGBM seed bagging (5 seeds) | `LGB_SEEDS` in run_final.py, `seeds=5` in cv_lgb | rejected: FD001 9.49 -> 9.33, FD004 20.89 -> 20.87 for 5x the cost (FD002/FD003 not run) |
| 2 | Baseline window length for the per-engine delta (10 / 25 / 40 cycles instead of 15) | parameterise `engine_feats` | todo |
| 3 | LightGBM random search on FD002/FD004 (num_leaves, min_child_samples, colsample, reg_lambda, learning rate) | 12 random configs, then gate the best | todo |
| 4 | Extra diverse ensemble member: ExtraTreesRegressor or HistGradientBoosting on the same features | OOF-blend weight tuned like the CNN | todo |
| 5 | Per-engine noise features: early-life std and slope of each sensor (first 30 cycles) | new feature group "early" | todo |
| 6 | Predict log(RUL+1) instead of RUL (relative error near end of life) | transform target in cv_lgb, invert | todo |
| 7 | Drop the noisiest sensors per subset (by early-life SNR) | feature selection inside CV | todo |
| 8 | Quantile LightGBM (5% / 95%) as an alternative to conformal bins; compare interval width at equal coverage | new interval method | todo |
| 9 | Retune triage thresholds to cut false P1 alarms on FD002/FD004 while keeping every urgent engine flagged | edit `tune_agent` objective | todo |
| 10 | CNN: 2-3 seeds averaged, window 50, critical-window weight | exp_cnn.py (slow) | todo |

## Already tried (do not repeat)
- Running min/max of deltas; EWMA 0.15 / 0.08; slow-trend features; regime id as a feature; per-regime EWMA state; healthy-reference normalisation.
- RUL cap 115 / 135 (125 is best); slower or bigger LightGBM; CNN without age channel; blending the CNN that has no age channel.
- Temporal aggregation (averaging p(t-j) - j over the last 5-50 cycles of each engine): no gain, errors are persistent per engine rather than per-cycle noise.
