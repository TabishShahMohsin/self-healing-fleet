"""Nightly idea 9: wider triage-threshold search (CV/OOF only, no test sets). Compares the current tune_agent grid with a
wider grid (disagree up to inf, P1 rul threshold, P1 health rule) using a 2-fold engine split: tune on one half, score on the other."""
import os, sys, json, inspect
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import run_final as R
import pipeline as P

CAP = {}
def spy(y, lo, hi, health, hs, p_g, p_c):
    fr = inspect.currentframe().f_back.f_locals
    CAP["d"] = dict(y=y, lo=lo, hi=hi, h=health, hs=hs, pg=p_g, pc=p_c, unit=fr["tr"].loc[fr["rows_eval"], "unit"].values)
    return R.tune_agent_orig(y, lo, hi, health, hs, p_g, p_c)
R.tune_agent_orig = R.tune_agent; R.tune_agent = spy

def tri(lo, w, h, hs, dis, wh, p1r, p1h):
    if dis or (w > wh and lo <= 50): return "HR"
    if lo <= p1r or (h >= p1h and hs > 0 and lo <= 50): return "P1"
    if lo <= 50: return "P2"
    return "O"

def sev(d, idx, wh, dg, p1r, p1h):
    out = []
    for i in idx:
        dis = abs(d["pg"][i] - d["pc"][i]) > dg or (d["lo"][i] > 50 and d["h"][i] >= 0.8)
        out.append(tri(d["lo"][i], d["hi"][i] - d["lo"][i], d["h"][i], d["hs"][i], dis, wh, p1r, p1h))
    return np.array(out)

def metrics(d, idx, cfg):
    s = sev(d, idx, *cfg); y = d["y"][idx]; u = y <= 20
    return dict(caught=float(np.isin(s[u], ["P1", "P2", "HR"]).mean()), hr=float((s == "HR").mean()),
                fp1=float(((s == "P1") & (y > 50)).mean()), p1_urgent=float((s[u] == "P1").mean()))

def obj(m): return m["hr"] + m["fp1"] + (0 if m["caught"] >= 0.98 else 10 * (0.98 - m["caught"]))

old = [(w, g, 20, 0.8) for w in (50, 60, 80, 100, 130) for g in (30, 40, 60)]
new = [(w, g, r, hh) for w in (50, 60, 80, 100, 130) for g in (30, 40, 60, 80, 120, 1e9) for r in (10, 15, 20) for hh in (0.8, 0.9, 2.0)]
res = {}
for fd in ["FD001", "FD002", "FD003", "FD004"]:
    CAP.clear(); R.run(fd, oof_only=True); d = CAP["d"]
    units = np.unique(d["unit"]); rng = np.random.RandomState(0); rng.shuffle(units)
    halves = [np.isin(d["unit"], units[:len(units)//2]), np.isin(d["unit"], units[len(units)//2:])]
    r = {"old": [], "new": []}
    for a in (0, 1):
        tr_i, te_i = np.where(halves[a])[0], np.where(halves[1 - a])[0]
        for name, grid in (("old", old), ("new", new)):
            best = min(grid, key=lambda c: obj(metrics(d, tr_i, c)))
            r[name].append(dict(cfg=[float(x) for x in best], **metrics(d, te_i, best)))
    res[fd] = {k: dict(held_out_mean={m: float(np.mean([x[m] for x in v])) for m in ("caught", "hr", "fp1", "p1_urgent")}, folds=v) for k, v in r.items()}
    print(fd, json.dumps({k: res[fd][k]["held_out_mean"] for k in res[fd]}), flush=True)
json.dump(res, open(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "out" / "n9_triage.json"), "w"), indent=1)
