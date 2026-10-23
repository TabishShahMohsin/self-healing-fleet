"""Final pipeline (v2).  python run_final.py [FD001 FD002 ...] [--refresh]

Every modelling decision is made on out-of-fold (OOF) predictions of the TRAINING engines
(5-fold, engine-grouped, scored at random truncation points, see exp_lib.py).  The official test
sets are touched once, at the very end, with frozen models.

  1. clean -> per-regime z-score -> EWMA -> causal features incl. per-engine baseline delta
  2. OOF predictions: LightGBM (critical-window weighted) and regime-gated 1D-CNN
  3. ensemble weight, Mondrian cross-conformal 90% intervals and triage-agent thresholds all chosen on OOF
  4. final models refit on 100% of training engines, evaluated once on the test set
  5. Isolation-Forest health index, triage agent, work orders, MLflow logging
"""
import sys, json, time, hashlib
import numpy as np, pandas as pd, lightgbm as lgb, mlflow
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
import pipeline as P, exp_lib as E, exp_cnn as C
from pipeline import SENS, CAP, CRIT, OUT, ROOT, SEED, load, validate, nasa_score, slope, triage, MondrianConformal

GROUPS = {"base", "delta", "ms"}                       # raw+rolling stats, per-engine baseline delta, multi-scale deltas
CRIT_W = 3.0                                           # extra sample weight for the last 30 cycles of life
CNN_EPOCHS = {"FD001": 30, "FD002": 25, "FD003": 30, "FD004": 25}
EDGES = (0, 30, 60, 100)                               # Mondrian bins on predicted RUL
LGB_SEEDS = 1                                          # LightGBM seeds averaged (bagging)
FEATURES_VERSION = "v1"                                # bump when exp_lib.engine_feats / preprocessing changes
CNN_VERSION = "gated-v1"                               # bump when exp_cnn architecture / inputs change
CACHE = E.CACHE


def _key(kind, fd):
    """Cache file name that changes whenever the configuration behind an OOF prediction changes."""
    cfg = (dict(groups=sorted(GROUPS), crit_w=CRIT_W, lgb=E.LGB_DEFAULT, seeds=LGB_SEEDS, feat=FEATURES_VERSION) if kind == "lgb"
           else dict(epochs=CNN_EPOCHS[fd], cnn=CNN_VERSION, feat=FEATURES_VERSION))
    h = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"final_oof_{kind}_{fd}_{h}.npy"


def _cached(name, fn, refresh):
    f = CACHE / name
    if f.exists() and not refresh:
        return np.load(f)
    a = fn(); np.save(f, a); return a


def oof_predictions(fd, refresh=False):
    """OOF RUL predictions of both models on the training engines (same folds for both).
    Cached by configuration hash (committed to git), so unchanged models are never re-cross-validated."""
    d = E.prep(fd)
    def _lgb():
        _, o, _ = E.cv_lgb(fd, GROUPS, crit_w=CRIT_W, seeds=LGB_SEEDS); return o
    def _cnn():
        _, o = C.cv_cnn(fd, "gated", epochs=CNN_EPOCHS[fd]); return o
    return d, _cached(_key("lgb", fd), _lgb, refresh), _cached(_key("cnn", fd), _cnn, refresh)


def health_index(trn, life_frac=0.25, fit_mask=None):
    life = trn.groupby("unit")["cycle"].transform("max")
    healthy = (trn["cycle"] <= life_frac * life).values
    if fit_mask is not None:
        healthy = healthy & fit_mask
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(trn.loc[healthy, SENS])
    raw = -iso.score_samples(trn[SENS])
    lo, hi = np.percentile(raw[healthy], 50), np.percentile(raw, 99.5)
    return lambda d: np.clip((-iso.score_samples(d[SENS]) - lo) / (hi - lo), 0, 1)


def health_slope_rows(df, h):
    """slope of the health index over the last 10 cycles, for every row (causal)."""
    out = np.zeros(len(df))
    s = pd.Series(h, index=df.index)
    for u, idx in df.groupby("unit").groups.items():
        out[df.index.get_indexer(idx)] = s.loc[idx].rolling(10, min_periods=2).apply(slope, raw=True).fillna(0).values
    return out


def decide(lo, hi, health, hs, p_g, p_c, data_ok, cfg):
    """Vectorised-safe wrapper of the rule-based triage agent; cfg holds OOF-tuned thresholds."""
    disagree = abs(p_g - p_c) > cfg["disagree"] or (lo > 50 and health >= 0.8)
    return triage(lo, hi - lo, health, hs, disagree, data_ok, cfg["width_hr"])


def tune_agent(y_true_raw, lo, hi, health, hs, p_g, p_c):
    """Pick (width_hr, disagree) on OOF cut rows: keep >=98% of truly-urgent (RUL<=20) flagged, minimise
    human-review load + false P1 alarms."""
    best = None
    urgent = y_true_raw <= 20
    for w in (50, 60, 80, 100, 130):
        for dg in (30, 40, 60):
            cfg = dict(width_hr=w, disagree=dg)
            sev = np.array([decide(lo[i], hi[i], health[i], hs[i], p_g[i], p_c[i], True, cfg)[0] for i in range(len(lo))])
            caught = np.isin(sev[urgent], ["P1_CRITICAL", "P2_HIGH", "HUMAN_REVIEW"]).mean()
            hr = (sev == "HUMAN_REVIEW").mean(); fp1 = ((sev == "P1_CRITICAL") & (y_true_raw > 50)).mean()
            obj = hr + fp1 + (0 if caught >= 0.98 else 10 * (0.98 - caught))
            if best is None or obj < best[0]:
                best = (obj, cfg, dict(urgent_caught=float(caught), hr_rate=float(hr), false_p1_rate=float(fp1)))
    return best[1], best[2]


def run(fd, refresh=False, oof_only=False):
    t0 = time.time()
    k = P.N_REG[fd]
    d, oof_g, oof_c = oof_predictions(fd, refresh)
    tr, te, rul_true = d["tr"], d["te"], d["rul"]
    y_all = np.minimum(tr["RUL"].values, CAP)

    # ---------------- 1. ensemble weight, conformal and agent tuning on OOF
    rows_eval = E.cut_rows(tr)                                         # reporting rows (seed 11)
    rows_cal = E.cut_rows(tr, per_engine=30, seed=23)                  # calibration rows (different seed)
    pos = lambda r: tr.index.get_indexer(r)
    best_w, best_r = 0.0, 1e9
    for w in np.arange(0, 0.61, 0.05):
        p = np.clip((1 - w) * oof_g[pos(rows_cal)] + w * oof_c[pos(rows_cal)], 0, None)
        r = np.sqrt(np.mean((y_all[pos(rows_cal)] - p) ** 2))
        if r < best_r - 1e-9:
            best_w, best_r = float(w), r
    blend = lambda g, c: np.clip((1 - best_w) * g + best_w * c, 0, None)
    oof_p = blend(oof_g, oof_c)
    cv = {f"cv_{n}_{key}": v for n, o in [("lgb", np.clip(oof_g, 0, None)), ("cnn", np.clip(oof_c, 0, None)), ("ens", oof_p)]
          for key, v in E.score(tr, o, rows_eval).items()}
    conf = MondrianConformal(EDGES).fit(oof_p[pos(rows_cal)], np.abs(y_all[pos(rows_cal)] - oof_p[pos(rows_cal)]))
    q_e = conf.q(oof_p[pos(rows_eval)]); y_e = y_all[pos(rows_eval)]
    cv["cv_coverage"] = float(np.mean(np.abs(y_e - oof_p[pos(rows_eval)]) <= q_e))      # in-sample coverage check on OOF

    # health index for agent tuning (iso trained on first 25% of life of all training engines)
    hidx_all = health_index(tr)
    h_tr = hidx_all(tr); hs_tr = health_slope_rows(tr, h_tr)
    pe = pos(rows_eval)
    lo_e = np.clip(oof_p[pe] - q_e, 0, None); hi_e = oof_p[pe] + q_e
    agent_cfg, agent_cv = tune_agent(tr.loc[rows_eval, "RUL"].values, lo_e, hi_e, h_tr[pe], hs_tr[pe],
                                     np.clip(oof_g[pe], 0, None), np.clip(oof_c[pe], 0, None))

    if oof_only:
        info = dict(w_cnn=best_w, agent=agent_cfg, agent_cv=agent_cv, conf_q=[round(q, 1) for q in conf.qs],
                    **{a: round(b, 3) for a, b in cv.items() if a.startswith(("cv_ens", "cv_lgb_rmse", "cv_cnn_rmse", "cv_cov"))})
        print(fd, json.dumps(info), flush=True)
        return info

    # ---------------- 2. final models on 100% of training engines
    Ftr = E.build_features(tr, GROUPS, k); Fte = E.build_features(te, GROUPS, k)
    w_s = 1.0 + CRIT_W * (tr["RUL"].values <= CRIT)
    gbms = [lgb.LGBMRegressor(**{**E.LGB_DEFAULT, "random_state": E.LGB_DEFAULT["random_state"] + i}).fit(Ftr, y_all, sample_weight=w_s)
            for i in range(LGB_SEEDS)]
    gbm = gbms[0]                                       # used for SHAP-style contributions and MLflow
    A_tr, b_tr = C.build_arrays(tr, k, "gated"); A_te, b_te = C.build_arrays(te, k, "gated")
    net = C.fit(A_tr, b_tr, y_all, np.ones(len(tr)), np.arange(len(tr)), CNN_EPOCHS[fd], seed=0)

    # ---------------- 3. one frozen evaluation on the official test set (last cycle of every engine)
    last = te.groupby("unit").tail(1).index; lp = te.index.get_indexer(last)
    p_g = np.clip(np.mean([g.predict(Fte.loc[last]) for g in gbms], axis=0), 0, None)
    p_c = np.clip(C.predict(net, A_te, b_te, lp), 0, None)
    p_te = blend(p_g, p_c)
    y_te = np.minimum(rul_true, CAP)
    q_te = conf.q(p_te); lo = np.clip(p_te - q_te, 0, None); hi = p_te + q_te
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    m = dict(rmse_ens=rmse(y_te, p_te), rmse_lgb=rmse(y_te, p_g), rmse_cnn=rmse(y_te, p_c), nasa=nasa_score(y_te, p_te),
             coverage=float(np.mean((y_te >= lo) & (y_te <= hi))), w_lgb=1 - best_w, w_cnn=best_w,
             width=float(np.mean(2 * q_te)), width_near_eol=float(np.mean(2 * q_te[p_te < 50])) if (p_te < 50).any() else float("nan"),
             n_test=len(last), **cv, agent_width_hr=agent_cfg["width_hr"], agent_disagree=agent_cfg["disagree"],
             **{f"agent_cv_{a}": b for a, b in agent_cv.items()})
    crit = (rul_true <= CRIT).astype(int)
    m["pr_auc_crit"] = float(average_precision_score(crit, -p_te)) if crit.sum() else float("nan")
    m["crit_rate"] = float(crit.mean())
    m["rmse_crit"] = rmse(y_te[crit == 1], p_te[crit == 1]) if crit.sum() else float("nan")

    # ---------------- 4. anomaly early-warning (iso fit WITHOUT the held-out engines) + health for test
    units = tr["unit"].unique(); rng = np.random.default_rng(SEED); rng.shuffle(units)
    cal_u = set(units[: max(10, len(units) // 5)])
    hidx_ev = health_index(tr, fit_mask=~tr["unit"].isin(cal_u).values)
    h_ev = hidx_ev(tr)
    leads = []
    for u in cal_u:
        idx = np.where(tr["unit"].values == u)[0]; h = pd.Series(h_ev[idx])
        flag = (h >= 0.6).rolling(3).sum() >= 3
        if flag.any():
            leads.append(int(len(idx) - 1 - flag.idxmax()))
    m["anomaly_lead_median"] = float(np.median(leads)) if leads else float("nan")
    m["anomaly_detect_rate"] = len(leads) / len(cal_u)
    h_te = hidx_all(te); hs_te = health_slope_rows(te, h_te)
    ok_te = validate(load(fd)[1], ref=load(fd)[0])

    # ---------------- 5. triage agent -> work orders
    wos, sev_arr = [], []
    contrib = gbm.predict(Fte.loc[last], pred_contrib=True)[:, :-1]
    for i, r in enumerate(last):
        u = int(te.at[r, "unit"])
        sev, action, due = decide(lo[i], hi[i], h_te[lp[i]], hs_te[lp[i]], p_g[i], p_c[i], bool(ok_te[te["unit"].values == u].all()), agent_cfg)
        sev_arr.append(sev if sev in ("P1_CRITICAL", "P2_HIGH", "HUMAN_REVIEW") else "NO_WO")
        if sev in ("P1_CRITICAL", "P2_HIGH", "HUMAN_REVIEW"):
            top = list(dict.fromkeys(Ftr.columns[j].split("_")[0] for j in np.argsort(contrib[i])[:8]))[:3]
            wos.append(dict(wo_id=f"WO-{fd}-{u:03d}", engine=f"{fd}/unit{u}", severity=sev, rul_pred=round(float(p_te[i]), 1),
                            rul_interval=[round(float(lo[i]), 1), round(float(hi[i]), 1)], health_index=round(float(h_te[lp[i]]), 2),
                            top_sensors=top, action=action, due_within_cycles=due, model_version=f"{fd}-ens-v2", true_rul=int(rul_true[i])))
    sev_all = np.array([decide(lo[i], hi[i], h_te[lp[i]], hs_te[lp[i]], p_g[i], p_c[i], True, agent_cfg)[0] for i in range(len(last))])
    sev_counts = {s: int((sev_all == s).sum()) for s in np.unique(sev_all)}
    sev_arr = np.array(sev_arr)
    urgent = rul_true <= 20
    m["agent_urgent_caught"] = float(np.isin(sev_arr[urgent], ["P1_CRITICAL", "P2_HIGH", "HUMAN_REVIEW"]).mean()) if urgent.any() else float("nan")
    m["agent_false_p1"] = int(((sev_all == "P1_CRITICAL") & (rul_true > 50)).sum())
    m["n_urgent"] = int(urgent.sum())
    m["human_review_rate"] = sev_counts.get("HUMAN_REVIEW", 0) / len(last)
    m["runtime_s"] = round(time.time() - t0, 1)

    with mlflow.start_run(run_name=f"{fd}-ensemble-v2"):
        mlflow.log_params(dict(subset=fd, cap=CAP, window=P.WIN, regimes=k, conformal_alpha=P.ALPHA, crit_w=CRIT_W,
                               cnn_weight=best_w, **{f"agent_{a}": b for a, b in agent_cfg.items()}))
        mlflow.log_metrics({kk: v for kk, v in m.items() if isinstance(v, (int, float)) and np.isfinite(v)})
        mlflow.lightgbm.log_model(gbm, name="lgbm")

    (OUT / f"workorders_{fd}.json").write_text(json.dumps(wos, indent=2))
    pd.DataFrame(dict(unit=te.loc[last, "unit"].values, true=y_te, true_raw=rul_true, pred=p_te, lo=lo, hi=hi,
                      lgb=p_g, cnn=p_c, health=h_te[lp], severity=sev_all)).to_csv(OUT / f"pred_{fd}.csv", index=False)
    pd.DataFrame(dict(unit=tr.unit, cycle=tr.cycle, rul=tr.RUL, health=h_ev))[tr.unit.isin(cal_u)].to_csv(OUT / f"health_{fd}.csv", index=False)
    m["sev_counts"] = sev_counts
    print(fd, json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in m.items()}), flush=True)
    return m


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
    mlflow.set_experiment("self-healing-fleet")
    fds = [a for a in sys.argv[1:] if a.startswith("FD")] or ["FD001", "FD002", "FD003", "FD004"]
    oof_only = "--oof-only" in sys.argv                 # cross-validation view only: never touches the test sets
    f = OUT / ("oof_only.json" if oof_only else "metrics.json")
    res = json.loads(f.read_text()) if f.exists() and len(fds) < 4 else {}
    for fd in fds:
        res[fd] = run(fd, "--refresh" in sys.argv, oof_only)
        f.write_text(json.dumps(res, indent=2))
