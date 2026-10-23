"""CNN experiments: engine-grouped 5-fold CV, windows gathered on the fly.
   python exp_cnn.py MODE FD [--epochs=40] [--seed=0] [--wd=1e-4] [--width=32] [--ewma=0.3]
   MODE: x (sensors only, old) | d (baseline deltas) | d+age | gated (deltas gated by regime one-hot + age)"""
import os, sys, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn
import exp_lib as E, pipeline as P
from pipeline import SENS, OUT, WIN

torch.set_num_threads(int(os.environ.get("TT", "1")))
CAP = 125


def build_arrays(tr, k, mode):
    assert (np.diff(tr["unit"].values) >= 0).all()
    blocks, base, off = [], np.zeros(len(tr), dtype=np.int64), 0
    pos = 0
    for u, d in tr.groupby("unit", sort=False):
        X = d[SENS].values.astype(np.float64); T = len(X)
        cs = np.cumsum(X, 0); n0 = np.minimum(np.arange(1, T + 1), 15)[:, None]
        b = np.where(np.arange(T)[:, None] < 15, cs / n0, cs[min(15, T) - 1] / min(15, T))
        D = X if mode == "x" else X - b
        parts = [D]
        if mode in ("d+age", "gated"):
            parts.append((d["cycle"].values / 300.0)[:, None])
        if mode == "gated":
            oh = np.eye(k)[d["regime"].values]
            parts += [oh, (D[:, :, None] * oh[:, None, :]).reshape(T, -1)] if k > 1 else []
        M = np.hstack(parts).astype(np.float32)
        blocks.append(np.vstack([np.repeat(M[:1], WIN - 1, 0), M]))
        base[pos:pos + T] = off + np.arange(T); pos += T; off += T + WIN - 1
    return torch.from_numpy(np.vstack(blocks)), base


def make_net(C, width=32, drop=0.3):
    return nn.Sequential(
        nn.Conv1d(C, width, 5, padding=2), nn.Tanh(),
        nn.Conv1d(width, width, 5, padding=2), nn.Tanh(),
        nn.Conv1d(width, width, 3, padding=1), nn.Tanh(),
        nn.Conv1d(width, 8, 3, padding=1), nn.Tanh(),
        nn.Flatten(), nn.Dropout(drop), nn.Linear(8 * WIN, 100), nn.Tanh(), nn.Linear(100, 1))


AR = torch.arange(WIN)


def gather(A, base, idx):
    return A[torch.from_numpy(base[idx])[:, None] + AR].transpose(1, 2)


def fit(A, base, y, w, idx, epochs=40, seed=0, wd=1e-4, width=32, bs=256):
    torch.manual_seed(seed)
    net = make_net(A.shape[1], width)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    yt = torch.from_numpy(y.astype(np.float32))[:, None] / CAP; wt = torch.from_numpy(w.astype(np.float32))[:, None]
    for ep in range(epochs):
        net.train(); perm = np.random.default_rng(seed * 1000 + ep).permutation(idx)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = (wt[b] * (net(gather(A, base, b)) - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net.eval()


def predict(net, A, base, idx):
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            out.append(net(gather(A, base, idx[i:i + 4096])).numpy().ravel())
    return np.concatenate(out) * CAP


def cv_cnn(fd, mode, epochs=40, seed=0, wd=1e-4, width=32, ewma=0.3, crit_w=0.0, n_folds=5, cap=CAP):
    d = E.prep(fd, ewma); tr = d["tr"]
    A, base = build_arrays(tr, d["k"], mode)
    y = np.minimum(tr["RUL"].values, cap); w = 1.0 + crit_w * (tr["RUL"].values <= P.CRIT)
    units = tr["unit"].unique(); oof = np.zeros(len(tr))
    for f, (tr_u, va_u) in enumerate(E.folds_for(units, n_folds)):
        m_tr = tr["unit"].isin(tr_u).values
        net = fit(A, base, y, w, np.where(m_tr)[0], epochs, seed + f, wd, width)
        va = np.where(~m_tr)[0]
        oof[va] = predict(net, A, base, va)
        print(f"  fold {f} done", flush=True)
    return tr, oof


if __name__ == "__main__":
    mode, fd = sys.argv[1], sys.argv[2]
    kw = {a.split("=")[0][2:]: float(a.split("=")[1]) for a in sys.argv[3:] if a.startswith("--")}
    kw = {k: (int(v) if k in ("epochs", "seed", "width") else v) for k, v in kw.items()}
    t = time.time()
    tr, oof = cv_cnn(fd, mode, **kw)
    m = E.score(tr, oof, E.cut_rows(tr))
    name = f"C_{mode}" + "".join(f"_{k}{v}" for k, v in kw.items())
    rec = dict(exp=name, fd=fd, n_feat=0, secs=round(time.time() - t), **{k: round(v, 3) for k, v in m.items()})
    print(json.dumps(rec), flush=True)
    res_f = OUT / "experiments.csv"
    pd.DataFrame([rec]).to_csv(res_f, mode="a", header=not res_f.exists(), index=False)
    np.save(OUT / "cache" / f"oof_{name}_{fd}.npy", oof)
