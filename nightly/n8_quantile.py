"""N8: quantile LightGBM 5/95 vs Mondrian conformal 90% intervals (CV only, no test sets). Compares width at ~equal coverage."""
import os, json, sys; os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import numpy as np, lightgbm as lgb
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import exp_lib as E, pipeline as P
CFG = dict(groups=__import__("experiments").M, crit_w=3.0, bw=40)
out = {}
for fd in ["FD001", "FD002", "FD003", "FD004"]:
    d = E.prep(fd); tr = d["tr"]; F = E.build_features(tr, CFG["groups"], d["k"], 40)
    y = np.minimum(tr["RUL"].values, 125); units = tr["unit"].unique()
    w = 1.0 + 3.0 * (tr["RUL"].values <= P.CRIT)
    _, oof, _ = E.cv_lgb(fd, **CFG)
    lo = np.zeros(len(tr)); hi = np.zeros(len(tr))
    for tu, vu in E.folds_for(units, 5):
        mt = tr["unit"].isin(tu).values
        for a, arr in ((0.05, lo), (0.95, hi)):
            arr[~mt] = lgb.LGBMRegressor(**{**E.LGB_DEFAULT, "objective": "quantile", "alpha": a}).fit(F[mt], y[mt], sample_weight=w[mt]).predict(F[~mt])
    rows = E.cut_rows(tr); pos = tr.index.get_indexer(rows); yy = y[pos]; p = np.clip(oof[pos], 0, 125)
    ue = tr.loc[rows, "unit"].values; half = np.isin(ue, units[::2])    # calibrate on half the engines, evaluate on the other half (and swap)
    res = {"conformal": [], "quantile": [], "quantile_cqr": []}
    for cal in (half, ~half):
        ev = ~cal
        c = P.MondrianConformal((0, 30, 60, 100)).fit(p[cal], np.abs(yy[cal] - p[cal])); q = c.q(p[ev])
        res["conformal"].append((np.mean(np.abs(yy[ev] - p[ev]) <= q), np.mean(2 * q)))
        l, h = lo[pos][ev], hi[pos][ev]
        res["quantile"].append((np.mean((yy[ev] >= l) & (yy[ev] <= h)), np.mean(h - l)))
        # CQR: conformalise the quantile band (single global correction on the calibration half)
        s = np.maximum(lo[pos][cal] - yy[cal], yy[cal] - hi[pos][cal]); k = np.quantile(s, min(1, np.ceil((cal.sum() + 1) * .9) / cal.sum()))
        res["quantile_cqr"].append((np.mean((yy[ev] >= l - k) & (yy[ev] <= h + k)), np.mean(h - l + 2 * k)))
    out[fd] = {k: dict(coverage=round(float(np.mean([a for a, _ in v])), 3), width=round(float(np.mean([b for _, b in v])), 2)) for k, v in res.items()}
    print(fd, json.dumps(out[fd]), flush=True)
json.dump(out, open("out/n8_quantile.json", "w"), indent=1)
