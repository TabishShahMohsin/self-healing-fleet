"""MLOps demo: a new fault mode appears in the fleet -> drift detected -> retrain -> champion/challenger gate.

Champion: trained on FD001 fleet (HPC fault only).
New batch: FD003 engines arrive (HPC + Fan faults) -> PSI drift check -> challenger retrained on FD001+FD003
-> both evaluated on the same held-out engines -> promote only if better. Logged to MLflow registry.
"""
import json
import numpy as np, pandas as pd, mlflow, lightgbm as lgb
from mlflow import MlflowClient
from pipeline import load, clean, RegimeNorm, nasa_score, SENS, CAP, CRIT, ROOT, OUT
import exp_lib as E
from run_final import GROUPS, CRIT_W, BASELINE_WIN      # same feature set / loss weighting as the production pipeline


def features(df, k):
    return E.build_features(df, GROUPS, k, BASELINE_WIN)


def train_lgb(F, y_raw):
    w = 1.0 + CRIT_W * (y_raw <= CRIT)
    return lgb.LGBMRegressor(**E.LGB_DEFAULT).fit(F, np.minimum(y_raw, CAP), sample_weight=w)


def psi(ref, new, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    a = np.histogram(new, edges)[0] / len(new) + 1e-4
    return float(np.sum((a - e) * np.log(a / e)))


def prep(fd, offset):
    tr, te, rul = load(fd)
    tr, te = clean(tr), clean(te)
    tr["unit"] += offset; te["unit"] += offset
    return tr, te, rul


mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
mlflow.set_experiment("retraining-demo")
client = MlflowClient()
NAME = "rul-fleet-model"

a_tr, a_te, a_rul = prep("FD001", 0)
b_tr, b_te, b_rul = prep("FD003", 1000)

# ---- champion on fleet A
norm_a = RegimeNorm().fit(a_tr, 1)
Fa = features(norm_a.transform(a_tr), 1)
champ = train_lgb(Fa, a_tr.RUL.values)

# ---- new batch arrives: drift check on regime-normalised sensors, last 30 cycles of each new engine
na, nb = norm_a.transform(a_tr), norm_a.transform(b_tr)
late = lambda d: d[d.groupby("unit")["cycle"].transform("max") - d["cycle"] < 30]
psis = {s: psi(late(na)[s].values, late(nb)[s].values) for s in SENS}
drifted = {s: round(v, 2) for s, v in sorted(psis.items(), key=lambda kv: -kv[1]) if v > 0.2}
trigger = len(drifted) > 0

# ---- evaluation set = official test engines of both fleets (never trained on)
def evaluate(model, norm):
    out = {}
    for name, te, rul in [("FD001", a_te, a_rul), ("FD003", b_te, b_rul)]:
        F = features(norm.transform(te), 1)
        last = te.groupby("unit").tail(1).index
        p = np.clip(model.predict(F.loc[last]), 0, None); y = np.minimum(rul, CAP)
        out[name] = dict(rmse=float(np.sqrt(np.mean((p - y) ** 2))), nasa=nasa_score(y, p))
    out["mean_rmse"] = float(np.mean([out[k]["rmse"] for k in ("FD001", "FD003")]))
    out["total_nasa"] = float(sum(out[k]["nasa"] for k in ("FD001", "FD003")))
    return out

champ_eval = evaluate(champ, norm_a)

with mlflow.start_run(run_name="champion-FD001") as r:
    mlflow.log_metrics({"mean_rmse": champ_eval["mean_rmse"], "total_nasa": champ_eval["total_nasa"]})
    mlflow.lightgbm.log_model(champ, name="model", registered_model_name=NAME)
v1 = client.get_latest_versions(NAME)[0].version if hasattr(client, "get_latest_versions") else "1"
client.set_registered_model_alias(NAME, "champion", v1)

result = dict(drift_trigger=trigger, drifted_sensors=drifted, champion=champ_eval)
if trigger:
    ab = pd.concat([a_tr, b_tr], ignore_index=True)
    norm_ab = RegimeNorm().fit(ab, 1)
    Fab = features(norm_ab.transform(ab), 1)
    chall = train_lgb(Fab, ab.RUL.values)
    chall_eval = evaluate(chall, norm_ab)
    promote = chall_eval["mean_rmse"] < champ_eval["mean_rmse"] and chall_eval["total_nasa"] < champ_eval["total_nasa"]
    with mlflow.start_run(run_name="challenger-FD001+FD003"):
        mlflow.log_metrics({"mean_rmse": chall_eval["mean_rmse"], "total_nasa": chall_eval["total_nasa"]})
        mlflow.log_dict(drifted, "drift_report.json")
        info = mlflow.lightgbm.log_model(chall, name="model", registered_model_name=NAME)
    v2 = info.registered_model_version
    client.set_registered_model_alias(NAME, "challenger", v2)
    if promote:
        client.set_registered_model_alias(NAME, "champion", v2)
    result.update(challenger=chall_eval, promoted=promote)

(OUT / "retrain_demo.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
