# Self-Healing Fleet: predictive maintenance PoC

Early proof of concept for **Tata Elxsi TELIPORT Season 4, Case Study 1** (Team Elxsirs, AMU).

A closed loop on NASA C-MAPSS turbofan data: sensor stream → clean and normalise → detect degradation →
predict Remaining Useful Life (RUL) with a confidence interval → a rule-based triage agent (no LLM) raises
maintenance work orders, with humans handling only the exceptions. Every run is tracked in MLflow.

## Pipeline

| Stage | What it does |
|---|---|
| Ingest + validate | Loads FD001–FD004; schema and range checks; out-of-range data is flagged and sent to human review |
| Clean | Per-engine interpolation for gaps, Hampel spike filter, EWMA smoothing |
| Normalise | Drops 7 flat sensors; K-Means (k = 6) on operating settings → z-score per regime (fit on train only) |
| Features | 30-cycle windows: rolling mean, std, slope, EWMA, time-in-regime counts |
| Anomaly | Isolation Forest trained on the first 25% of each engine's life → degradation index 0–1 |
| RUL | LightGBM (window features) + 1D-CNN (30 × 14 windows), inverse-error weighted ensemble; RUL capped at 125 |
| Confidence | Split-conformal 90% interval, binned by predicted RUL (tight near failure) |
| Triage agent | P1 / P2 / P3 / Normal / Human review rules on the interval's lower bound, degradation and disagreement → JSON work orders |
| MLOps | MLflow tracking + model registry; PSI drift check → retrain → champion/challenger gate |

## Results (official NASA test sets, never seen in training)

RMSE in cycles against true RUL capped at 125 (standard protocol). "Urgent" = true RUL ≤ 20.

| Subset | Test engines | LightGBM | 1D-CNN | Ensemble | NASA score | 90% interval coverage | Urgent engines flagged | Human review |
|---|---|---|---|---|---|---|---|---|
| FD001 | 100 | 12.4 | 13.7 | **12.4** | 263 | 93% | 16/16 | 0% |
| FD002 | 259 | 18.9 | 23.0 | **18.1** | 1,716 | 91% | 53/53 | 25% |
| FD003 | 100 | 13.0 | 14.5 | **12.7** | 300 | 92% | 14/14 | 3% |
| FD004 | 248 | 24.8 | 28.0 | **24.1** | 6,010 | 88% | 40/40 | 50% |

**Retraining demo** (`retrain_demo.py`): a champion trained on FD001 (compressor fault only) meets FD003 engines with a
new fan fault. PSI flags 10/14 sensors as drifted (> 0.2), a challenger is retrained, and FD003 RMSE drops
from 21.2 to 12.0 with FD001 unchanged (11.3 → 11.4), so it is promoted to `@champion`.

## Run it

```bash
pip install -r requirements.txt
python pipeline.py            # all four subsets, ~7 min on 2 CPU cores (python pipeline.py FD001 for one)
python retrain_demo.py        # drift → retrain → champion/challenger gate
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Outputs land in `out/`: `metrics.json`, `pred_FDxxx.csv` (predictions + intervals + severity per test engine),
`workorders_FDxxx.json` (agent output), `health_FDxxx.csv` (degradation trajectories), `retrain_demo.json`.

## Known limitations / next steps

- FD002/FD004 (six operating conditions) have wide intervals, so the agent escalates many engines to human review (50% on FD004). Reducing this is the main Round 2 target.
- The 1D-CNN is weaker than LightGBM on its own; the ensemble weights it accordingly.
- Not yet built: Prefect orchestration, Streamlit dashboard, class-weighted loss for the critical window, live streaming replay.
- The retraining demo uses LightGBM only.

## Data

`data/` holds the NASA C-MAPSS Turbofan Engine Degradation Simulation files (Saxena et al., 2008), from the NASA Prognostics
Center of Excellence data repository.
