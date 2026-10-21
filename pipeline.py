"""Self-Healing Fleet - early PoC on NASA C-MAPSS (FD001-FD004).

ingest -> validate/clean -> regime normalisation -> features -> anomaly (IsolationForest)
-> RUL (LightGBM + 1D-CNN ensemble) -> split-conformal interval -> rule-based triage agent
-> JSON work orders. Every run is logged to MLflow.

Usage:  python pipeline.py            (all four subsets)
        python pipeline.py FD001      (one subset)
"""
import json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
DATA, OUT = ROOT / "data", ROOT / "out"
OUT.mkdir(exist_ok=True)

SEED = 42
CAP = 125            # piecewise-linear RUL cap
WIN = 30             # window length (cycles)
CRIT = 30            # last N cycles = critical class
ALPHA = 0.10         # conformal miscoverage -> 90% interval
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
FLAT = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SENS = [f"s{i}" for i in range(1, 22) if f"s{i}" not in FLAT]       # 14 informative sensors
N_REG = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------- ingest
def load(fd):
    rd = lambda n: pd.read_csv(DATA / f"{n}_{fd}.txt", sep=r"\s+", header=None)
    tr, te = rd("train"), rd("test")
    tr.columns = te.columns = COLS
    rul = rd("RUL")[0].values
    tr["RUL"] = tr.groupby("unit")["cycle"].transform("max") - tr["cycle"]
    return tr, te, rul


# ----------------------------------------------------------------------------- validate + clean
def validate(df, ref=None):
    """Schema + range checks. Returns mask of rows that pass."""
    assert list(df.columns[:26]) == COLS, "schema mismatch"
    ok = df[SENS].notna().all(axis=1) | True          # gaps are repaired below, not rejected
    if ref is not None:                               # physically-plausible band from training data
        lo, hi = ref[SENS].quantile(0.0001), ref[SENS].quantile(0.9999)
        span = hi - lo
        ok &= ((df[SENS] >= lo - span) & (df[SENS] <= hi + span)).all(axis=1)
    return ok


def hampel(x, k=7, t=3.0):
    med = x.rolling(k, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(k, center=True, min_periods=1).median() * 1.4826
    return x.where((x - med).abs() <= t * mad + 1e-9, med)


def clean(df):
    g = df.groupby("unit", group_keys=False)
    df[SENS] = g[SENS].apply(lambda d: d.interpolate(limit_direction="both").ffill().bfill())
    df[SENS] = g[SENS].apply(lambda d: d.apply(hampel))
    return df


# ----------------------------------------------------------------------------- regime normalisation
class RegimeNorm:
    def fit(self, df, k):
        self.km = KMeans(k, n_init=10, random_state=SEED).fit(df[["op1", "op2", "op3"]].round(1)) if k > 1 else None
        r = self.regime(df)
        self.mu = df[SENS].groupby(r).mean()
        self.sd = df[SENS].groupby(r).std().replace(0, 1)
        self.k = k
        return self

    def regime(self, df):
        return pd.Series(self.km.predict(df[["op1", "op2", "op3"]].round(1)) if self.km else 0, index=df.index)

    def transform(self, df):
        df = df.copy()
        r = self.regime(df)
        df[SENS] = (df[SENS].values - self.mu.loc[r].values) / self.sd.loc[r].values
        df["regime"] = r.values
        g = df.groupby("unit")
        for j in range(self.k):                                   # regime history (time-in-regime)
            df[f"reg{j}_cnt"] = g["regime"].transform(lambda s, j=j: (s == j).cumsum())
        df[SENS] = g[SENS].transform(lambda s: s.ewm(alpha=0.3).mean())   # EWMA denoise
        return df


# ----------------------------------------------------------------------------- features
def slope(a):
    n = len(a); x = np.arange(n) - (n - 1) / 2
    return (x * (a - a.mean())).sum() / (x ** 2).sum() if n > 1 else 0.0


def features(df, k):
    g = df.groupby("unit")
    F = pd.DataFrame(index=df.index)
    F["cycle"] = df["cycle"]
    for s in SENS:
        F[f"{s}"] = df[s]
        F[f"{s}_m10"] = g[s].transform(lambda x: x.rolling(10, min_periods=1).mean())
        F[f"{s}_m30"] = g[s].transform(lambda x: x.rolling(WIN, min_periods=1).mean())
        F[f"{s}_sd30"] = g[s].transform(lambda x: x.rolling(WIN, min_periods=2).std()).fillna(0)
        F[f"{s}_sl30"] = g[s].transform(lambda x: x.rolling(WIN, min_periods=2).apply(slope, raw=True)).fillna(0)
    for j in range(k):
        F[f"reg{j}_cnt"] = df[f"reg{j}_cnt"]
    return F


def windows(df, units=None, last_only=False):
    """30 x 14 windows for the CNN; short series are front-padded with their first row."""
    X, idx = [], []
    for u, d in df.groupby("unit"):
        if units is not None and u not in units:
            continue
        a = d[SENS].values.astype(np.float32)
        pad = max(0, WIN - len(a))
        if pad:
            a = np.vstack([np.repeat(a[:1], pad, 0), a])
        ends = [len(a)] if last_only else range(WIN, len(a) + 1)
        for e in ends:
            X.append(a[e - WIN:e])
            idx.append(d.index[e - 1 - pad])
    return np.stack(X), idx


# ----------------------------------------------------------------------------- models
def train_lgb(X, y):
    m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, verbose=-1)
    return m.fit(X, y)


def train_cnn(X, y, epochs=60):
    import torch, torch.nn as nn
    torch.manual_seed(SEED); torch.set_num_threads(2)
    # Li, Ding & Sun (2018) style: stacked same-padded convs along time, keep temporal order (no pooling)
    net = nn.Sequential(
        nn.Conv1d(len(SENS), 16, 9, padding=4), nn.Tanh(),
        nn.Conv1d(16, 16, 9, padding=4), nn.Tanh(),
        nn.Conv1d(16, 16, 9, padding=4), nn.Tanh(),
        nn.Conv1d(16, 4, 3, padding=1), nn.Tanh(),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * WIN, 100), nn.Tanh(), nn.Linear(100, 1))
    Xt = torch.tensor(X).transpose(1, 2); yt = torch.tensor(y, dtype=torch.float32)[:, None] / CAP
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = perm[i:i + 256]
            loss = nn.functional.mse_loss(net(Xt[b]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval()
    return net


def pred_cnn(net, X):
    import torch
    with torch.no_grad():
        return net(torch.tensor(X).transpose(1, 2)).numpy().ravel() * CAP


class MondrianConformal:
    """Split conformal with bins on predicted RUL: tight intervals near end-of-life, wider when healthy."""
    EDGES = [0, 50, 90, np.inf]

    def fit(self, p, res):
        qq = lambda r: float(np.quantile(r, min(1, np.ceil((len(r) + 1) * (1 - ALPHA)) / len(r))))
        self.glob = qq(res)
        b = np.digitize(p, self.EDGES[1:-1])
        self.qs = [qq(res[b == i]) if (b == i).sum() >= 30 else self.glob for i in range(len(self.EDGES) - 1)]
        return self

    def q(self, p):
        return np.array(self.qs)[np.digitize(p, self.EDGES[1:-1])]


def nasa_score(y, p):
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


# ----------------------------------------------------------------------------- triage agent (no LLM)
def triage(rul_lo, width, health, health_slope, disagree, data_ok):
    if not data_ok or disagree or (width > 60 and rul_lo <= 50):   # uncertain AND decision-relevant
        return "HUMAN_REVIEW", "Exception queue: engineer review", None
    if rul_lo <= 20 or (health >= 0.8 and health_slope > 0 and rul_lo <= 50):
        return "P1_CRITICAL", "Ground unit and inspect", 5
    if rul_lo <= 50:
        return "P2_HIGH", "Schedule at next service window", 25
    if rul_lo <= 100 or health >= 0.6:
        return "P3_WATCH", "Watch-list, increase monitoring", None
    return "NORMAL", "No action", None


# ----------------------------------------------------------------------------- run one subset
def run(fd, use_cnn=True):
    t0 = time.time()
    tr, te, rul_true = load(fd)
    k = N_REG[fd]

    ok_tr, ok_te = validate(tr), validate(te, ref=tr)
    tr, te = clean(tr), clean(te)
    norm = RegimeNorm().fit(tr, k)                       # fit on train only
    trn, ten = norm.transform(tr), norm.transform(te)
    Ftr, Fte = features(trn, k), features(ten, k)
    ytr = np.minimum(tr["RUL"].values, CAP)

    # engine-level split: 80% fit, 20% conformal calibration (no leakage across engines)
    units = tr["unit"].unique(); rng.shuffle(units)
    cal_u = set(units[: max(10, len(units) // 5)])
    fit_m = ~tr["unit"].isin(cal_u).values

    # --- anomaly model on healthy early life (first 25% of each training engine)
    life = tr.groupby("unit")["cycle"].transform("max")
    healthy = fit_m & (tr["cycle"] <= 0.25 * life).values
    iso = IsolationForest(n_estimators=300, random_state=SEED).fit(trn.loc[healthy, SENS])
    raw_tr = -iso.score_samples(trn[SENS])
    lo_s, hi_s = np.percentile(raw_tr[healthy], 50), np.percentile(raw_tr, 99.5)
    hidx = lambda d: np.clip((-iso.score_samples(d[SENS]) - lo_s) / (hi_s - lo_s), 0, 1)
    h_tr, h_te = hidx(trn), hidx(ten)

    # --- RUL models
    gbm = train_lgb(Ftr[fit_m], ytr[fit_m])
    if use_cnn:
        Xw, iw = windows(trn, units=set(units) - cal_u)
        net = train_cnn(Xw, ytr[tr.index.get_indexer(iw)])

    def predict(dfn, F, rows):
        p_g = gbm.predict(F.loc[rows])
        if not use_cnn:
            return p_g, p_g, p_g
        p_c = []
        for r in rows:                                  # window ending at row r
            u = dfn.at[r, "unit"]; d = dfn[(dfn.unit == u) & (dfn.index <= r)]
            p_c.append(d)
        Xc = np.stack([windows(d, last_only=True)[0][0] for d in p_c])
        p_c = pred_cnn(net, Xc)
        return W[0] * p_g + W[1] * p_c, p_g, p_c

    # calibration points: 5 random truncation points per calibration engine (mimics test truncation)
    cal_rows = []
    for u in cal_u:
        idx = tr.index[tr.unit == u]
        cal_rows += list(rng.choice(idx, size=min(25, len(idx)), replace=False))
    W = [0.5, 0.5]
    _, pg_cal, pc_cal = predict(trn, Ftr, cal_rows)
    if use_cnn:                                          # inverse-MSE ensemble weights from calibration engines
        inv = np.array([1 / np.mean((ytr[cal_rows] - pg_cal) ** 2), 1 / np.mean((ytr[cal_rows] - pc_cal) ** 2)])
        W = list(inv / inv.sum())
    p_cal = W[0] * pg_cal + W[1] * pc_cal
    res = np.abs(ytr[cal_rows] - p_cal)
    conf = MondrianConformal().fit(p_cal, res)

    # test: last cycle of every test engine
    last = te.groupby("unit").tail(1).index
    p_te, p_g, p_c = predict(ten, Fte, list(last))
    p_te = np.clip(p_te, 0, None)
    y_te = np.minimum(rul_true, CAP)
    q_te = conf.q(p_te)
    lo, hi = np.clip(p_te - q_te, 0, None), p_te + q_te

    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    m = dict(rmse_ens=rmse(y_te, p_te), rmse_lgb=rmse(y_te, p_g), rmse_cnn=rmse(y_te, p_c),
             nasa=nasa_score(y_te, p_te), coverage=float(np.mean((y_te >= lo) & (y_te <= hi))),
             w_lgb=float(W[0]), width=float(np.mean(2 * q_te)), width_near_eol=float(np.mean(2 * q_te[p_te < 50])) if (p_te < 50).any() else float("nan"),
             n_test=len(last))

    # critical-window detection (true RUL <= 30) -- PR-AUC using -predicted RUL as score
    crit = (rul_true <= CRIT).astype(int)
    m["pr_auc_crit"] = float(average_precision_score(crit, -p_te)) if crit.sum() else float("nan")
    m["crit_rate"] = float(crit.mean())

    # early warning on held-out run-to-failure engines: first cycle health >= 0.6 for 3 cycles
    leads = []
    for u in cal_u:
        idx = tr.index[tr.unit == u]; h = pd.Series(h_tr[tr.index.get_indexer(idx)])
        flag = (h >= 0.6).rolling(3).sum() >= 3
        if flag.any():
            leads.append(int(len(idx) - 1 - flag.idxmax()))
    m["anomaly_lead_median"] = float(np.median(leads)) if leads else float("nan")
    m["anomaly_detect_rate"] = len(leads) / len(cal_u)

    # --- triage agent -> work orders
    wos, sev_counts = [], {}
    imp = pd.Series(gbm.booster_.feature_importance("gain"), index=Ftr.columns)
    contrib = gbm.predict(Fte.loc[last], pred_contrib=True)[:, :-1]
    for i, r in enumerate(last):
        u = int(te.at[r, "unit"])
        hu = pd.Series(h_te[te.index.get_indexer(te.index[te.unit == u])])
        hs = slope(hu.tail(10).values) if len(hu) > 1 else 0.0
        disagree = abs(p_g[i] - p_c[i]) > 40 or (lo[i] > 50 and hu.iloc[-1] >= 0.8)   # RUL says fine, anomaly says not
        sev, action, due = triage(lo[i], hi[i] - lo[i], hu.iloc[-1], hs, disagree, bool(ok_te.loc[te.unit == u].all()))
        sev_counts[sev] = sev_counts.get(sev, 0) + 1
        top = [Ftr.columns[j].split("_")[0] for j in np.argsort(contrib[i])[:6]]
        top = list(dict.fromkeys(top))[:3]               # sensors pushing RUL down the most
        if sev in ("P1_CRITICAL", "P2_HIGH", "HUMAN_REVIEW"):
            wos.append(dict(wo_id=f"WO-{fd}-{u:03d}", engine=f"{fd}/unit{u}", severity=sev,
                            rul_pred=round(float(p_te[i]), 1), rul_interval=[round(float(lo[i]), 1), round(float(hi[i]), 1)],
                            health_index=round(float(hu.iloc[-1]), 2), top_sensors=top, action=action,
                            due_within_cycles=due, model_version=f"{fd}-ens-v1", true_rul=int(rul_true[i])))
    # agent quality vs ground truth
    sev_arr = []
    for i, r in enumerate(last):
        w = next((w for w in wos if w["wo_id"] == f"WO-{fd}-{int(te.at[r,'unit']):03d}"), None)
        sev_arr.append(w["severity"] if w else "NO_WO")
    sev_arr = np.array(sev_arr)
    urgent = rul_true <= 20
    m["agent_urgent_caught"] = float(np.isin(sev_arr[urgent], ["P1_CRITICAL", "P2_HIGH", "HUMAN_REVIEW"]).mean()) if urgent.any() else float("nan")
    m["agent_false_p1"] = float(((sev_arr == "P1_CRITICAL") & (rul_true > 50)).sum())
    m["n_urgent"] = int(urgent.sum())
    m["human_review_rate"] = sev_counts.get("HUMAN_REVIEW", 0) / len(last)
    m["runtime_s"] = round(time.time() - t0, 1)

    # --- MLflow
    with mlflow.start_run(run_name=f"{fd}-ensemble"):
        mlflow.log_params(dict(subset=fd, cap=CAP, window=WIN, regimes=k, conformal_alpha=ALPHA, cnn=use_cnn))
        mlflow.log_metrics({kk: v for kk, v in m.items() if isinstance(v, (int, float)) and np.isfinite(v)})
        mlflow.lightgbm.log_model(gbm, name="lgbm")

    (OUT / f"workorders_{fd}.json").write_text(json.dumps(wos, indent=2))
    pd.DataFrame(dict(unit=te.loc[last, "unit"].values, true=y_te, true_raw=rul_true, pred=p_te, lo=lo, hi=hi,
                      lgb=p_g, cnn=p_c, health=h_te[te.index.get_indexer(last)], severity=sev_arr)
                 ).to_csv(OUT / f"pred_{fd}.csv", index=False)
    # health trajectories of calibration engines (for plotting)
    traj = pd.DataFrame(dict(unit=tr.unit, cycle=tr.cycle, rul=tr.RUL, health=h_tr))[tr.unit.isin(cal_u)]
    traj.to_csv(OUT / f"health_{fd}.csv", index=False)
    m["sev_counts"] = sev_counts
    print(fd, json.dumps({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in m.items()}), flush=True)
    return m


if __name__ == "__main__":
    mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
    mlflow.set_experiment("self-healing-fleet")
    fds = [a for a in sys.argv[1:] if a.startswith("FD")] or ["FD001", "FD002", "FD003", "FD004"]
    res = {fd: run(fd, use_cnn="--no-cnn" not in sys.argv) for fd in fds if fd.startswith("FD")}
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2))
