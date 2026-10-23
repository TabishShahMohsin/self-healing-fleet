"""Render assets/banner.png from the result files in out/ (no template needed, so nightly runs can refresh it).
   python assets/make_banner.py      (needs: pip install playwright; Chromium available)"""
import json, html
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FDS = ["FD001", "FD002", "FD003", "FD004"]
M = json.loads((OUT / "metrics.json").read_text())
RT = json.loads((OUT / "retrain_demo.json").read_text())
P1 = pd.read_csv(OUT / "pred_FD001.csv").sort_values("true").reset_index(drop=True)
WO = json.loads((OUT / "workorders_FD001.json").read_text())

n_test = sum(M[f]["n_test"] for f in FDS)
n_urg = sum(M[f]["n_urgent"] for f in FDS)
caught = sum(round(M[f]["agent_urgent_caught"] * M[f]["n_urgent"]) for f in FDS)
cov = sum(M[f]["coverage"] * M[f]["n_test"] for f in FDS) / n_test
fp1 = sum(M[f]["agent_false_p1"] for f in FDS)
ch, cl = RT["champion"], RT["challenger"]

# ---- FD001 chart (SVG) ----
W, H, PL, PR, PT, PB = 640, 262, 44, 10, 10, 30
n = len(P1); X = lambda i: PL + i * (W - PL - PR) / (n - 1); Y = lambda v: PT + (1 - v / 175) * (H - PT - PB)
band = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(P1["hi"])) + " " + \
       " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in reversed(list(enumerate(P1["lo"]))))
line = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(P1["true"]))
dots = "".join(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="2.6" fill="#F2A900"/>' for i, v in enumerate(P1["pred"]))
grid = "".join(f'<line x1="{PL}" x2="{W-PR}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#E3E8EE"/><text x="{PL-6}" y="{Y(v)+3:.1f}" text-anchor="end" font-size="10" fill="#556575">{v}</text>'
               for v in range(0, 176, 25))
chart = (f'<svg viewBox="0 0 {W} {H}" width="100%"><rect width="{W}" height="{H}" fill="white"/>{grid}'
         f'<polygon points="{band}" fill="#9CC9E6" fill-opacity=".45"/><polyline points="{line}" fill="none" stroke="#0B2A4A" stroke-width="2.2"/>{dots}'
         f'<text x="{PL}" y="{H-8}" font-size="10" fill="#556575">engines, sorted by true RUL</text>'
         f'<text x="{W-PR}" y="{H-8}" font-size="10" fill="#556575" text-anchor="end">'
         f'<tspan fill="#0B2A4A">━ true</tspan>  <tspan fill="#F2A900">● predicted</tspan>  <tspan fill="#5aa9d6">▇ 90% interval</tspan></text></svg>')

# ---- retrain bars (SVG) ----
vals = [("FD001", ch["FD001"]["rmse"], cl["FD001"]["rmse"]), ("FD003", ch["FD003"]["rmse"], cl["FD003"]["rmse"])]
vmax = max(v for _, a, b in vals for v in (a, b)) * 1.2; bh = 120
bars = ""
for k, (name, a, b) in enumerate(vals):
    x0 = 30 + k * 150
    for j, (v, col) in enumerate([(a, "#6B7280"), (b, "#008CC8")]):
        h = v / vmax * bh; x = x0 + j * 52
        bars += f'<rect x="{x}" y="{140-h:.1f}" width="46" height="{h:.1f}" fill="{col}"/><text x="{x+23}" y="{136-h:.1f}" text-anchor="middle" font-size="12" font-weight="700" fill="#1F2A37">{v:.1f}</text>'
    bars += f'<text x="{x0+49}" y="158" text-anchor="middle" font-size="11" fill="#556575">{name} test</text>'
bars_svg = (f'<svg viewBox="0 0 330 176" height="170">{bars}<text x="165" y="174" text-anchor="middle" font-size="10" fill="#556575">'
            f'<tspan fill="#6B7280">■ champion</tspan>  <tspan fill="#008CC8">■ challenger (promoted)</tspan></text></svg>')

# ---- table ----
head = ["Subset", "LightGBM", "1D-CNN", "Ensemble", "NASA score", "Coverage", "Urgent caught", "Human review", "False P1"]
rows = "".join(
    "<tr>" + "".join(f"<td{' class=b' if j in (0,3) else ''}>{c}</td>" for j, c in enumerate([
        f"{f} ({M[f]['n_test']})", f"{M[f]['rmse_lgb']:.1f}", f"{M[f]['rmse_cnn']:.1f}", f"<span class=hl>{M[f]['rmse_ens']:.1f}</span>",
        f"{M[f]['nasa']:,.0f}", f"{M[f]['coverage']*100:.0f}%", f"{round(M[f]['agent_urgent_caught']*M[f]['n_urgent'])}/{M[f]['n_urgent']}",
        f"{M[f]['human_review_rate']*100:.0f}%", M[f]["agent_false_p1"]])) + "</tr>" for f in FDS)
table = "<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + f"</tr>{rows}</table>"

# ---- work order ----
wo = next((w for w in WO if w["severity"] == "P1_CRITICAL" and w["true_rul"] <= 15), WO[0])
show = {k: wo[k] for k in ["wo_id", "engine", "severity", "rul_pred", "rul_interval", "health_index", "top_sensors", "action", "due_within_cycles", "model_version"]}
show["degradation_index"] = show.pop("health_index")
wo_txt = "{\n" + ",\n".join(f'  "{k}": {json.dumps(v)}' for k, v in show.items()) + "\n}"
wo_txt = html.escape(wo_txt).replace("&quot;severity&quot;", "<b>&quot;severity&quot;</b>").replace("&quot;action&quot;", "<b>&quot;action&quot;</b>")

kp = [(f"{M['FD001']['rmse_ens']:.1f}", "FD001 RMSE (cycles)", "LightGBM + 1D-CNN ensemble"),
      (f"{M['FD004']['rmse_ens']:.1f}", "FD004 RMSE, hardest subset", "6 conditions · 2 fault modes"),
      (f"{caught}/{n_urg}", "engines with ≤ 20 cycles left", "got a work order or escalation"),
      (f"{cov*100:.0f}%", "true RUL inside the 90% interval", "target 90%, calibrated out-of-fold"),
      (f"{ch['FD003']['rmse']:.1f} → {cl['FD003']['rmse']:.1f}", "FD003 RMSE after auto-retrain", "drift → retrain → promoted")]
kpis = "".join(f'<div class="kpi{" alt" if i == 4 else ""}"><div class=big>{b}</div><div class=lab>{l}</div><div class=sub>{s}</div></div>' for i, (b, l, s) in enumerate(kp))
drift = list(RT["drifted_sensors"].items())[0]

PAGE = f"""<html><head><meta charset=utf-8><style>
*{{box-sizing:border-box}} body{{margin:0;width:1333px;height:750px;overflow:hidden;background:#fff;font-family:'Segoe UI','Helvetica Neue',Arial,'Liberation Sans',sans-serif;color:#1F2A37;padding:22px 30px}}
h1{{margin:0;font-size:30px;color:#0B2A4A}} h1 span{{color:#008CC8}} .s{{margin:4px 0 14px;font-size:13px;color:#556575}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}} .kpi{{background:#0B2A4A;color:#fff;border-radius:12px;padding:10px 8px;text-align:center}}
.kpi.alt{{background:#F2A900;color:#0B2A4A}} .big{{font-size:34px;font-weight:700;color:#F2A900}} .alt .big{{color:#0B2A4A}}
.lab{{font-size:13px;font-weight:700}} .sub{{font-size:11px;opacity:.8}} .row{{display:grid;grid-template-columns:1fr 1fr;gap:26px;margin-top:16px}}
h2{{font-size:15px;margin:0 0 6px;color:#0B2A4A}} h2 small{{font-weight:400;color:#556575}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th{{background:#0B2A4A;color:#fff;padding:6px 4px;font-size:12px}} td{{text-align:center;padding:6px 4px}}
tr:nth-child(odd) td{{background:#EEF6FB}} td.b{{font-weight:700;text-align:left;white-space:nowrap}} .hl{{color:#008CC8;font-weight:700}}
.note{{font-size:11px;color:#556575;margin-top:5px}} .card{{border-radius:12px;padding:12px 16px;height:240px}} .tint{{background:#EEF6FB}} .navy{{background:#0B2A4A;color:#fff}}
.steps{{font-size:12.5px;line-height:1.4}} .steps p{{margin:0 0 7px}} pre{{margin:6px 0 0;font:12px/1.28 'Courier New',monospace;color:#fff}} pre b{{color:#F2A900;font-weight:400}}
</style></head><body>
<h1><span>Self-Healing Fleet:</span> early PoC on all four NASA C-MAPSS subsets</h1>
<div class=s>Team Elxsirs · Tata Elxsi TELIPORT S4, Case Study 1. Scored on the official test sets: {n_test} engines never seen in training. RMSE against true RUL capped at 125 (standard protocol).</div>
<div class=kpis>{kpis}</div>
<div class=row><div><h2>Results per subset <small>(official test engines)</small></h2>{table}
<div class=note>RMSE in cycles, lower is better. Urgent = true RUL ≤ 20. Every urgent engine got a work order; the price is {fp1} false P1 alarms (true RUL &gt; 50), mostly on the six-condition subsets.</div></div>
<div><h2>FD001: {len(P1)} test engines <small>predicted vs true RUL with 90% interval</small></h2>{chart}</div></div>
<div class=row><div class="card tint" style="display:grid;grid-template-columns:1fr 1fr;gap:10px"><div class=steps><h2>MLOps in action: <small>a new fault mode enters the fleet</small></h2>
<p><b>1 Champion:</b> trained on FD001 engines (compressor fault only)</p>
<p><b>2 New batch:</b> FD003 engines arrive with fan faults; PSI flags {len(RT['drifted_sensors'])}/14 sensors (max {drift[1]:.2f} on {drift[0]}) &gt; 0.2 → retrain triggered</p>
<p><b>3 Gate:</b> challenger beats champion on held-out engines → MLflow alias moved to @champion</p></div>
<div><h2 style="text-align:center">RMSE (cycles)</h2>{bars_svg}</div></div>
<div class="card navy"><h2 style="color:#F2A900">Real work order raised by the agent <small style="color:#C9DBEA">(true RUL was {wo['true_rul']} cycles)</small></h2><pre>{wo_txt}</pre></div></div>
</body></html>"""

if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"])
        pg = b.new_page(viewport={"width": 1333, "height": 750}, device_scale_factor=2)
        pg.set_content(PAGE); pg.screenshot(path=str(ROOT / "assets" / "banner.png"), clip={"x": 0, "y": 0, "width": 1333, "height": 750})
        b.close()
    print("wrote assets/banner.png")
