# Nightly improvement run: rules for unattended sessions

You are running unattended, in a fresh container, on the repo `TabishShahMohsin/self-healing-fleet`
(NASA C-MAPSS remaining-useful-life PoC for Tata Elxsi TELIPORT S4). Goal of one night: try ONE improvement idea,
keep it only if cross-validation says it is better, record everything, push. Nobody will answer questions: decide, and log why.

## Hard rules
1. **Never use the NASA test sets to choose anything.** Accept/reject is decided by `python nightly/gate.py` (cross-validation only).
   Test numbers are produced only by `python run_final.py`, and only after an idea has been accepted.
2. Never force-push, never rewrite history, never edit `data/`, never delete `out/experiments.csv` rows (append only).
3. Do not touch the submission deck/PDF (it is not in the repo). Do not change the scheduled task or its time.
4. Keep the repo in a working state: if anything fails midway, `git checkout -- .` the code you changed, log the failure
   in `nightly/LOG.md`, and still push the log + experiment rows.
5. One idea per night, one-hour budget for experiments (LightGBM CV is ~1 min per subset). CNN cross-validation
   takes ~35 min for all four subsets, so only try CNN ideas when no LightGBM idea is left, and run subsets in parallel (2 cores).

## Steps
1. `pip install --break-system-packages -r requirements.txt playwright` (Chromium is preinstalled; never run `playwright install`).
2. Read `nightly/BACKLOG.md`, `nightly/LOG.md`, `nightly/champion.json`, tail of `out/experiments.csv`.
3. Pick the first backlog item whose status is `todo`. Mark it `running` in BACKLOG.md.
4. Implement it as a new entry in the `CFG` dict of `experiments.py` (or a small option in `exp_lib.py` that defaults to the old behaviour).
   Run all four subsets: `python experiments.py NAME --fds=FD001,FD002,FD003,FD004` (set `MLFLOW_DISABLE_AGENT_HINT=1`).
5. `python nightly/gate.py NAME` prints ACCEPT or REJECT (exit code 0 / 1).
6. **REJECT** (or exit 2 after one retry): set the item to `rejected` with the numbers in BACKLOG.md, add a dated line to
   `nightly/LOG.md`, commit `out/experiments.csv`, BACKLOG.md, LOG.md and your code, push, stop.
7. **ACCEPT**: adopt the idea in `run_final.py` (set the constant / groups; bump `FEATURES_VERSION` or `CNN_VERSION` if
   feature code or CNN code changed so caches are not reused), then
   - `python run_final.py --oof-only` then `python nightly/gate.py --ens`. If this is REJECT, revert the adoption and treat as rejected.
   - otherwise `python run_final.py` (all four subsets, one frozen test evaluation), `python retrain_demo.py`,
     `python assets/make_banner.py`, `python nightly/gate.py --refresh-champion`.
   - Update the README: results table (copy from `out/metrics.json`), the iteration log row, "Tried and dropped" if relevant. Numbers in the README must match `out/metrics.json` exactly; re-read both before committing.
   - Commit everything including the config-keyed `out/cache/final_oof_*.npy` files, with a message like `Nightly: <idea> (CV RMSE a -> b)`.
8. Push to `main` (`git pull --rebase` first if needed). Update `nightly/LOG.md` with: date, idea, CV before/after, accepted/rejected, test RMSE only if a full run happened.
9. Final reply (it is the only thing the user reads in the morning): 3 lines max: idea, verdict, key numbers.

## Where things are
- `exp_lib.py` features + LightGBM CV; `exp_cnn.py` CNN CV; `experiments.py` experiment configs; `run_final.py` the full pipeline;
  `pipeline.py` loading/cleaning/conformal/triage primitives; `nightly/gate.py` the accept/reject rule; `assets/make_banner.py` README banner.
- Metric: RMSE (cycles) of predicted vs true RUL capped at 125, at random truncation points of held-out training engines.
