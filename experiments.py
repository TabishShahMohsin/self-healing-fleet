"""Run named experiments (LightGBM, 5-fold engine-grouped CV, random-cut scoring). Appends to out/experiments.csv.
   python experiments.py E0 E1 ... [--fds FD001,FD004]"""
import sys, time, json
import numpy as np, pandas as pd
import exp_lib as E
from pipeline import OUT

B = {"base"}
CFG = {
 "E0_current_features":       dict(groups=B),
 "E1_+baseline_delta":        dict(groups=B | {"delta"}),
 "E2_+multiscale":            dict(groups=B | {"delta", "ms"}),
 "E3_+running_extremes":      dict(groups=B | {"delta", "ms", "ext"}),
}
M = B | {"delta", "ms"}
CFG.update({
 "E4a_ewma0.15":   dict(groups=M, ewma=0.15),
 "E4b_ewma0.08":   dict(groups=M, ewma=0.08),
 "E5_+slow_trend": dict(groups=M | {"slow"}),
 "E6_+regime_id":  dict(groups=M | {"reg"}),
 "E7_healthy_ref": dict(groups=M, ref=0.2),
})
CFG.update({
 "E8_+per_regime_state": dict(groups=M | {"rwise"}),
 "E8b_rwise_only_deltas": dict(groups={"delta", "rwise"}),
})
CFG.update({
 "E9_critw3":      dict(groups=M, crit_w=3.0, params=dict(n_jobs=1)),
 "E10_cap115":     dict(groups=M, cap=115, params=dict(n_jobs=1)),
 "E10b_cap135":    dict(groups=M, cap=135, params=dict(n_jobs=1)),
 "E11_slow_lr":    dict(groups=M, params=dict(n_jobs=1, n_estimators=800, learning_rate=0.025, num_leaves=15, min_child_samples=80, colsample_bytree=0.4)),
 "E11b_big":       dict(groups=M, params=dict(n_jobs=1, n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=30)),
})
CFG.update({
 "E12_bag5_critw3": dict(groups=M, crit_w=3.0, seeds=5),
})
def add(name, **kw): CFG[name] = kw

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fds = ["FD001", "FD004"]
    for a in sys.argv[1:]:
        if a.startswith("--fds"): fds = a.split("=")[1].split(",")
    res_f = OUT / "experiments.csv"
    for name in args:
        cfg = CFG[name]
        for fd in fds:
            t = time.time()
            tr, oof, nf = E.cv_lgb(fd, **cfg)
            rows = E.cut_rows(tr)
            m = E.score(tr, oof, rows)
            rec = dict(exp=name, fd=fd, n_feat=nf, secs=round(time.time() - t), **{k: round(v, 3) for k, v in m.items()})
            print(json.dumps(rec), flush=True)
            pd.DataFrame([rec]).to_csv(res_f, mode="a", header=not res_f.exists(), index=False)
            np.save(OUT / "cache" / f"oof_{name}_{fd}.npy", oof)
