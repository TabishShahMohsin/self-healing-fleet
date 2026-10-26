"""Experiment harness: vectorised causal features + engine-grouped CV with test-like random truncation."""
import pickle, time, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
import pipeline as P
from pipeline import SENS, N_REG, OUT

CACHE = OUT / "cache"; CACHE.mkdir(exist_ok=True)
EVAL_CAP = 125


# ------------------------------------------------------------------ preprocessing (cached)
def prep(fd, ewma=0.3, ref=None):
    f = CACHE / f"prep_{fd}_{ewma}_{ref}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    tr, te, rul = P.load(fd)
    tr, te = P.clean(tr), P.clean(te)
    norm = P.RegimeNorm().fit(tr, N_REG[fd], ref)            # unsupervised stats, train only
    d = dict(tr=norm.transform(tr, ewma), te=norm.transform(te, ewma), rul=rul, k=N_REG[fd])
    pickle.dump(d, open(f, "wb"))
    return d


# ------------------------------------------------------------------ causal features (numpy, per engine)
def _roll_sum(cs, w):
    T = cs.shape[0] - 1
    t = np.arange(1, T + 1)
    lo = np.maximum(0, t - w)
    return cs[t] - cs[lo], np.minimum(t, w)


def engine_feats(X, cyc, groups, reg=None, k=1, bw=15):
    T, S = X.shape
    cs = np.vstack([np.zeros((1, S)), np.cumsum(X, 0)])
    cs2 = np.vstack([np.zeros((1, S)), np.cumsum(X ** 2, 0)])
    tt = np.arange(T, dtype=float)[:, None]
    cst = np.vstack([np.zeros((1, 1)), np.cumsum(tt, 0)])
    cstt = np.vstack([np.zeros((1, 1)), np.cumsum(tt ** 2, 0)])
    cstx = np.vstack([np.zeros((1, S)), np.cumsum(tt * X, 0)])

    def mean(w):
        s, n = _roll_sum(cs, w); return s / n[:, None]

    def std(w):
        s, n = _roll_sum(cs, w); s2, _ = _roll_sum(cs2, w)
        return np.sqrt(np.maximum(s2 / n[:, None] - (s / n[:, None]) ** 2, 0))

    def slope(w):
        sx, n = _roll_sum(cs, w); st, _ = _roll_sum(cst, w); stt, _ = _roll_sum(cstt, w); stx, _ = _roll_sum(cstx, w)
        n = n[:, None].astype(float)
        den = stt - st ** 2 / n
        out = np.where(n > 1, (stx - st * sx / n) / np.where(den == 0, 1, den), 0.0)
        return out

    def lag(k):
        idx = np.maximum(np.arange(T) - k, 0); return X - X[idx]

    out, names = [X], [f"{s}" for s in SENS]
    def add(M, tag):
        out.append(M); names.extend(f"{s}_{tag}" for s in SENS)
    if "base" in groups:
        add(mean(10), "m10"); add(mean(30), "m30"); add(std(30), "sd30"); add(slope(30), "sl30")
    if "delta" in groups or "ext" in groups:
        n0 = np.minimum(np.arange(1, T + 1), bw)[:, None]
        b = np.where(np.arange(T)[:, None] < bw, cs[1:] / n0, cs[min(bw, T)] / min(bw, T))
        d = X - b
        if "delta" in groups: add(d, "d")
        if "ext" in groups:
            add(np.maximum.accumulate(d, 0), "dmax"); add(np.minimum.accumulate(d, 0), "dmin")
    if "slow" in groups:
        n0 = np.minimum(np.arange(1, T + 1), bw)[:, None]
        b = np.where(np.arange(T)[:, None] < bw, cs[1:] / n0, cs[min(bw, T)] / min(bw, T))
        dd = pd.DataFrame(X - b)
        add(dd.ewm(alpha=0.1).mean().values, "e10"); add(dd.ewm(alpha=0.03).mean().values, "e03"); add(slope(50), "sl50")
    if "rwise" in groups:   # per-regime EWMA state of baseline-deltas (smoothing only within the same operating condition)
        n0 = np.minimum(np.arange(1, T + 1), bw)[:, None]
        b = np.where(np.arange(T)[:, None] < bw, cs[1:] / n0, cs[min(bw, T)] / min(bw, T))
        dd = X - b
        for r in range(k):
            Dr = pd.DataFrame(np.where((reg == r)[:, None], dd, np.nan))
            out.append(Dr.ewm(alpha=0.15, ignore_na=True).mean().values); names.extend(f"{s}_r{r}" for s in SENS)
    if "ms" in groups:
        add(mean(5), "m5"); add(std(10), "sd10"); add(slope(10), "sl10"); add(lag(5), "dx5"); add(lag(10), "dx10")
    return np.hstack(out).astype(np.float32), names


def build_features(df, groups, k, bw=15):
    cols, names, parts = None, None, []
    for u, d in df.groupby("unit", sort=False):
        M, names = engine_feats(d[SENS].values.astype(np.float64), d["cycle"].values, groups, d["regime"].values, k, bw)
        parts.append(M)
    F = pd.DataFrame(np.vstack(parts), columns=names, index=df.index)
    F["cycle"] = df["cycle"].values
    if "reg" in groups:
        F["regime"] = df["regime"].values
    for j in range(k):
        F[f"reg{j}_cnt"] = df[f"reg{j}_cnt"].values
    return F


# ------------------------------------------------------------------ CV
def folds_for(units, n=5, seed=7):
    kf = KFold(n, shuffle=True, random_state=seed)
    return [(units[a], units[b]) for a, b in kf.split(units)]


def cut_rows(tr, per_engine=12, seed=11, min_cycle=20):
    rng = np.random.default_rng(seed)
    rows = []
    for u, d in tr.groupby("unit"):
        c = d.index[d["cycle"] >= min_cycle][:-1]
        rows += list(rng.choice(c, size=min(per_engine, len(c)), replace=False))
    return np.array(rows)


LGB_DEFAULT = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5, random_state=42, verbose=-1, n_jobs=2)


def cv_lgb(fd, groups, cap=125, crit_w=0.0, params=None, ewma=0.3, ref=None, n_folds=5, seeds=1, bw=15, et=0.0):
    d = prep(fd, ewma, ref); tr = d["tr"]
    F = build_features(tr, groups, d["k"], bw)
    y = np.minimum(tr["RUL"].values, cap)
    w = 1.0 + crit_w * (tr["RUL"].values <= P.CRIT)
    units = tr["unit"].unique(); oof = np.zeros(len(tr))
    pr = {**LGB_DEFAULT, **(params or {})}
    for tr_u, va_u in folds_for(units, n_folds):
        m_tr = tr["unit"].isin(tr_u).values; m_va = ~m_tr
        oof[m_va] = np.mean([lgb.LGBMRegressor(**{**pr, "random_state": pr["random_state"] + s_}).fit(
            F[m_tr], y[m_tr], sample_weight=w[m_tr]).predict(F[m_va]) for s_ in range(seeds)], axis=0)
        if et:   # optional diverse member: ExtraTrees on the same features, blended with weight `et`
            from sklearn.ensemble import ExtraTreesRegressor
            xt = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.3, n_jobs=2, random_state=42)
            oof[m_va] = (1 - et) * oof[m_va] + et * xt.fit(F[m_tr], y[m_tr], sample_weight=w[m_tr]).predict(F[m_va])
    return tr, oof, F.shape[1]


def nasa(y, p):
    dlt = p - y
    return float(np.mean(np.where(dlt < 0, np.exp(-dlt / 13) - 1, np.exp(dlt / 10) - 1)))


def score(tr, oof, rows):
    y = np.minimum(tr.loc[rows, "RUL"].values, EVAL_CAP)
    p = np.clip(oof[tr.index.get_indexer(rows)], 0, None)
    crit = tr.loc[rows, "RUL"].values <= P.CRIT
    return dict(rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                rmse_crit=float(np.sqrt(np.mean((y[crit] - p[crit]) ** 2))),
                nasa=nasa(y, p),
                pr_auc=float(average_precision_score(crit, -p)))
