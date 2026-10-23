"""Self-Healing Fleet - shared building blocks for the NASA C-MAPSS (FD001-FD004) PoC.

Data loading and validation, cleaning (interpolation + Hampel), per-regime normalisation, Mondrian conformal
intervals, NASA score and the rule-based triage agent. The end-to-end run is `python run_final.py`.
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
    def fit(self, df, k, ref_frac=None):
        self.km = KMeans(k, n_init=10, random_state=SEED).fit(df[["op1", "op2", "op3"]].round(1)) if k > 1 else None
        r = self.regime(df)
        ref = df if ref_frac is None else df[df["cycle"] <= ref_frac * df.groupby("unit")["cycle"].transform("max")]
        self.mu = ref[SENS].groupby(r[ref.index]).mean()          # optional: healthy-reference statistics
        self.sd = ref[SENS].groupby(r[ref.index]).std().replace(0, 1)
        self.k = k
        return self

    def regime(self, df):
        return pd.Series(self.km.predict(df[["op1", "op2", "op3"]].round(1)) if self.km else 0, index=df.index)

    def transform(self, df, ewma=0.3):
        df = df.copy()
        r = self.regime(df)
        df[SENS] = (df[SENS].values - self.mu.loc[r].values) / self.sd.loc[r].values
        df["regime"] = r.values
        g = df.groupby("unit")
        for j in range(self.k):                                   # regime history (time-in-regime)
            df[f"reg{j}_cnt"] = g["regime"].transform(lambda s, j=j: (s == j).cumsum())
        df[SENS] = g[SENS].transform(lambda s: s.ewm(alpha=ewma).mean())   # EWMA denoise
        return df


# ----------------------------------------------------------------------------- features
def slope(a):
    n = len(a); x = np.arange(n) - (n - 1) / 2
    return (x * (a - a.mean())).sum() / (x ** 2).sum() if n > 1 else 0.0


# ----------------------------------------------------------------------------- conformal intervals
class MondrianConformal:
    """Split/cross conformal with bins on predicted RUL: tight intervals near end-of-life, wider when healthy."""

    def __init__(self, edges=(0, 50, 90)):
        self.edges = list(edges)

    def fit(self, p, res):
        qq = lambda r: float(np.quantile(r, min(1, np.ceil((len(r) + 1) * (1 - ALPHA)) / len(r))))
        self.glob = qq(res)
        b = np.digitize(p, self.edges[1:])
        self.qs = [qq(res[b == i]) if (b == i).sum() >= 30 else self.glob for i in range(len(self.edges))]
        return self

    def q(self, p):
        return np.array(self.qs)[np.digitize(p, self.edges[1:])]


def nasa_score(y, p):
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


# ----------------------------------------------------------------------------- triage agent (no LLM)
def triage(rul_lo, width, health, health_slope, disagree, data_ok, width_hr=60):
    if not data_ok or disagree or (width > width_hr and rul_lo <= 50):   # uncertain AND decision-relevant
        return "HUMAN_REVIEW", "Exception queue: engineer review", None
    if rul_lo <= 20 or (health >= 0.8 and health_slope > 0 and rul_lo <= 50):
        return "P1_CRITICAL", "Ground unit and inspect", 5
    if rul_lo <= 50:
        return "P2_HIGH", "Schedule at next service window", 25
    if rul_lo <= 100 or health >= 0.6:
        return "P3_WATCH", "Watch-list, increase monitoring", None
    return "NORMAL", "No action", None
