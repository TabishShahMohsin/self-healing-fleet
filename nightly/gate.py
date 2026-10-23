"""Mechanical accept/reject gate for nightly experiments (cross-validation only, never the test sets).

  python nightly/gate.py EXP_NAME            LightGBM-side experiment: reads out/experiments.csv (all four subsets required)
  python nightly/gate.py --ens               full-blend check: reads out/oof_only.json written by `python run_final.py --oof-only`
  python nightly/gate.py --refresh-champion  rewrite nightly/champion.json from out/metrics.json (after an accepted full run)

ACCEPT needs: mean RMSE over the four subsets improves by >= MIN_GAIN, no subset gets worse by more than MAX_WORSE,
and the mean RMSE on the critical window (true RUL <= 30) does not get worse by more than MAX_WORSE.
Exit code 0 = accept, 1 = reject, 2 = incomplete input.
"""
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FDS = ["FD001", "FD002", "FD003", "FD004"]
MIN_GAIN, MAX_WORSE = 0.15, 0.30
champ = json.loads((ROOT / "nightly" / "champion.json").read_text())

if "--refresh-champion" in sys.argv:
    M = json.loads((ROOT / "out" / "metrics.json").read_text())
    champ["lgb"] = {f: {"rmse": round(M[f]["cv_lgb_rmse"], 3), "rmse_crit": round(M[f]["cv_lgb_rmse_crit"], 3)} for f in FDS}
    champ["ens"] = {f: {"rmse": round(M[f]["cv_ens_rmse"], 3), "rmse_crit": round(M[f]["cv_ens_rmse_crit"], 3)} for f in FDS}
    (ROOT / "nightly" / "champion.json").write_text(json.dumps(champ, indent=2)); print("champion refreshed"); sys.exit(0)

if "--ens" in sys.argv:
    J = json.loads((ROOT / "out" / "oof_only.json").read_text())
    if any(f not in J for f in FDS): print("need all four subsets in out/oof_only.json"); sys.exit(2)
    new, ref = {f: {"rmse": J[f]["cv_ens_rmse"], "rmse_crit": J[f]["cv_ens_rmse_crit"]} for f in FDS}, champ["ens"]
else:
    name = sys.argv[1]
    d = pd.read_csv(ROOT / "out" / "experiments.csv"); d = d[d.exp == name].drop_duplicates("fd", keep="last").set_index("fd")
    if any(f not in d.index for f in FDS): print(f"{name}: need all four subsets in out/experiments.csv, have {list(d.index)}"); sys.exit(2)
    new, ref = {f: {"rmse": float(d.loc[f, "rmse"]), "rmse_crit": float(d.loc[f, "rmse_crit"])} for f in FDS}, champ["lgb"]

dm = {k: sum(new[f][k] - ref[f][k] for f in FDS) / 4 for k in ("rmse", "rmse_crit")}
worst = max(new[f]["rmse"] - ref[f]["rmse"] for f in FDS)
for f in FDS:
    print(f"{f}: rmse {ref[f]['rmse']:.2f} -> {new[f]['rmse']:.2f}   crit {ref[f]['rmse_crit']:.2f} -> {new[f]['rmse_crit']:.2f}")
print(f"mean change: rmse {dm['rmse']:+.3f}  crit {dm['rmse_crit']:+.3f}  worst subset rmse change {worst:+.3f}")
ok = dm["rmse"] <= -MIN_GAIN and worst <= MAX_WORSE and dm["rmse_crit"] <= MAX_WORSE
print("ACCEPT" if ok else "REJECT"); sys.exit(0 if ok else 1)
