import argparse
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

GROUPS = {
    "engagement": ["FLOW", "TIME_FOR_PASSION", "ACHIEVEMENT", "TODO_COMPLETED", "LIVE_VISION"],
    "calm":       ["DAILY_STRESS", "DAILY_SHOUTING", "LOST_VACATION"],
    "giving":     ["SUPPORTING_OTHERS", "DONATION", "PERSONAL_AWARDS", "CORE_CIRCLE"],
    "active":     ["FRUITS_VEGGIES", "DAILY_STEPS", "PLACES_VISITED", "WEEKLY_MEDITATION"],
    "social":     ["SOCIAL_NETWORK"],
}
NEGATIVE = ["DAILY_STRESS", "DAILY_SHOUTING", "LOST_VACATION"]
CONTEXT = ["SUFFICIENT_INCOME", "BMI_RANGE", "AGE", "GENDER"]
TARGET = "WORK_LIFE_BALANCE_SCORE"

FEATURES = [c for cols in GROUPS.values() for c in cols]
DIMS = list(GROUPS)

SLEEP = "SLEEP_HOURS"
SLEEP_TARGET = 8.0
REST_WEIGHT = 0.3

DIM_WEIGHTS = {"social": 0.3}
WEIGHTS = np.array([DIM_WEIGHTS.get(d, 1.0) for d in DIMS])


def columns_of(dim):
    return [FEATURES.index(c) for c in GROUPS[dim]]


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def _clean(path):
    df = pd.read_csv(path)
    for c in FEATURES + [SLEEP, TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=FEATURES + [SLEEP, TARGET]).reset_index(drop=True)


def load_context(path):
    return _clean(path)[CONTEXT]


def load(path):
    df = _clean(path)
    X = df[FEATURES].astype(float).copy()
    X[NEGATIVE] = -X[NEGATIVE]
    return X, df[SLEEP].to_numpy(dtype=float), df[TARGET].to_numpy()


def rest_score(sleep):
    return -np.abs(sleep - SLEEP_TARGET)


def sleep_table(sleep, y):
    table = (pd.DataFrame({"sleep": sleep, "score": y})
             .groupby("sleep")["score"].agg(["mean", "count"]))
    print("\nMean WORK_LIFE_BALANCE_SCORE by SLEEP_HOURS (train):")
    print(table.round(1).to_string())


def cronbach_alpha(Z):
    k = Z.shape[1]
    if k < 2:
        return float("nan")
    return k / (k - 1) * (1 - Z.var(axis=0, ddof=1).sum() / Z.sum(axis=1).var(ddof=1))


def recon_mse(model, Z):
    return float(np.mean((Z - model.inverse_transform(model.transform(Z))) ** 2))


def baselines(Ztr, Zte):
    print("\n== Linear baselines (5 components) ==")
    pca = PCA(n_components=5, random_state=0).fit(Ztr)
    print(f"PCA test reconstruction MSE: {recon_mse(pca, Zte):.4f} "
          f"(explained variance {pca.explained_variance_ratio_.sum():.2%})")

    fa = FactorAnalysis(n_components=5, rotation="varimax", random_state=0).fit(Ztr)
    loadings = pd.DataFrame(fa.components_.T, index=FEATURES,
                            columns=[f"F{i+1}" for i in range(5)])
    print("\nFactor-analysis loadings (|loading| > 0.3 shown):")
    print(loadings.where(loadings.abs() > 0.3).round(2).astype(object).fillna(""))

    print("\nCronbach's alpha per dimension (internal consistency, >0.6 is ok):")
    for d in DIMS:
        print(f"  {d:<11} {cronbach_alpha(Ztr[:, columns_of(d)]):.2f}")
    return pca, fa


def build_models():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PosLinear(nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            w = nn.Linear(n_in, n_out).weight.detach().abs() + 1e-3
            self.raw_weight = nn.Parameter(torch.log(torch.expm1(w)))
            self.bias = nn.Parameter(torch.zeros(n_out))

        def forward(self, x):
            return F.linear(x, F.softplus(self.raw_weight), self.bias)

    class GroupedAE(nn.Module):
        def __init__(self, hidden=8):
            super().__init__()
            self.slices = [torch.tensor(columns_of(d)) for d in DIMS]
            self.encoders = nn.ModuleList(
                nn.Sequential(PosLinear(len(GROUPS[d]), hidden), nn.Tanh(), PosLinear(hidden, 1))
                for d in DIMS)
            self.decoder = nn.Sequential(nn.Linear(len(DIMS), 32), nn.ReLU(),
                                         nn.Linear(32, len(FEATURES)))

        def encode(self, x):
            return torch.cat([enc(x[:, s]) for enc, s in zip(self.encoders, self.slices)], 1)

        def forward(self, x):
            z = self.encode(x)
            return self.decoder(z), z

    class FreeAE(nn.Module):
        def __init__(self, k=5):
            super().__init__()
            d = len(FEATURES)
            self.enc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, k))
            self.decoder = nn.Sequential(nn.Linear(k, 32), nn.ReLU(), nn.Linear(32, d))

        def encode(self, x):
            return self.enc(x)

        def forward(self, x):
            z = self.encode(x)
            return self.decoder(z), z

    return GroupedAE, FreeAE


def reconstruction_error(model, x):
    return ((model(x)[0] - x) ** 2).mean()


def train(model_class, Ztr, Zval, seed, epochs=300, patience=20, lr=1e-3, wd=1e-4):
    import torch
    torch.manual_seed(seed)
    model = model_class()
    xtr = torch.tensor(Ztr, dtype=torch.float32)
    xval = torch.tensor(Zval, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for batch in torch.randperm(len(xtr)).split(256):
            noisy = xtr[batch] + 0.1 * torch.randn_like(xtr[batch])
            loss = ((model(noisy)[0] - xtr[batch]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val = reconstruction_error(model, xval).item()
        if val < best - 1e-5:
            best, bad_epochs = val, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best


def encode(model, Z):
    import torch
    with torch.no_grad():
        return model.encode(torch.tensor(Z, dtype=torch.float32)).numpy()


def orient(sub, Z):
    return np.array([np.sign(corr(sub[:, j], Z[:, columns_of(d)].mean(1)))
                     for j, d in enumerate(DIMS)])


def train_grouped(GroupedAE, Ztr, Zval, Zte, seeds):
    print("\n== Grouped autoencoder ==")
    runs = []
    for seed in range(seeds):
        model, val = train(GroupedAE, Ztr, Zval, seed=seed)
        sub = encode(model, Zte)
        runs.append((model, val, sub * orient(sub, Zte)))
        print(f"seed {seed}: val MSE {val:.4f}")

    model, _, sub_te = min(runs, key=lambda r: r[1])
    signs = orient(encode(model, Ztr), Ztr)

    print("Stability (corr of each subscore between seed 0 and seed 1):")
    if len(runs) > 1:
        for j, d in enumerate(DIMS):
            print(f"  {d:<11} {corr(runs[0][2][:, j], runs[1][2][:, j]):.3f}")
    return model, sub_te, signs


def compare_with_free(FreeAE, model, pca, fa, Ztr, Zval, Zte, sub_te):
    import torch
    xte = torch.tensor(Zte, dtype=torch.float32)
    with torch.no_grad():
        print(f"Grouped AE test reconstruction MSE: {reconstruction_error(model, xte):.4f}"
              f"  (PCA: {recon_mse(pca, Zte):.4f})")

    free, _ = train(FreeAE, Ztr, Zval, seed=0)
    with torch.no_grad():
        print(f"Free AE test reconstruction MSE:    {reconstruction_error(free, xte):.4f}")

    free_sub = encode(free, Zte)
    fa_sub = fa.transform(Zte)
    print("\nMax |corr| between each grouped subscore and any free-AE unit / FA factor:")
    for j, d in enumerate(DIMS):
        best_free = max(abs(corr(sub_te[:, j], free_sub[:, k])) for k in range(5))
        best_fa = max(abs(corr(sub_te[:, j], fa_sub[:, k])) for k in range(5))
        print(f"  {d:<11} free AE {best_free:.2f}   FA {best_fa:.2f}")


def combine(sub, rest, stats, dim_w=WEIGHTS, rest_w=REST_WEIGHT):
    z = np.column_stack([(sub - stats["sub_mean"]) / stats["sub_std"],
                         (rest - stats["rest_mean"]) / stats["rest_std"]])
    w = np.r_[dim_w, rest_w]
    return z @ w / w.sum()


def report_validation(index_te, sub_te, rest_te, s_te, yte, stats):
    equal_weights = combine(sub_te, rest_te, stats, np.ones(len(DIMS)), 1.0)

    print("\n== Validation ==")
    print("weights: " + ", ".join(f"{d} {w:g}" for d, w in zip(DIMS, WEIGHTS))
          + f", rest {REST_WEIGHT:g}")
    print(f"corr(index, dataset's {TARGET}) on test: {corr(index_te, yte):.3f}"
          f"   (all weights 1: {corr(equal_weights, yte):.3f})")

    print("corr of each subscore with the dataset score:")
    for j, d in enumerate(DIMS):
        print(f"  {d:<11} {corr(sub_te[:, j], yte):.3f}")
    print(f"  {'rest':<11} {corr(rest_te, yte):.3f}"
          f"   (raw sleep hours: {corr(s_te, yte):.3f})")

    print("corr of rest with each autoencoder subscore (low = adds new information):")
    for j, d in enumerate(DIMS):
        print(f"  {d:<11} {corr(sub_te[:, j], rest_te):.3f}")


def report_context(csv, Xte, index_te, ytr, yte):
    ctx = load_context(csv).loc[Xte.index].reset_index(drop=True)
    y_pct = np.interp(yte, np.sort(ytr), np.linspace(0, 100, len(ytr)))

    print("\n== Context groups (not used by the index) ==")
    print("mean index  |  mean dataset-score percentile  |  n   (test set)")
    for c in CONTEXT:
        for group, rows in ctx.groupby(c).groups.items():
            print(f"  {c:<17} {str(group):<13} {index_te[rows].mean():5.1f}  |  "
                  f"{y_pct[rows].mean():5.1f}  |  {len(rows)}")


def export(model, signs, scaler, stats, grid, out):
    import torch

    class Exportable(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            self.register_buffer("signs", torch.tensor(signs, dtype=torch.float32))

        def forward(self, z):
            return self.m.encode(z) * self.signs

    try:
        torch.onnx.export(Exportable(model).eval(), torch.zeros(1, len(FEATURES)),
                          f"model/{out}.onnx", input_names=["z"], output_names=["subscores"],
                          dynamic_axes={"z": {0: "batch"}, "subscores": {0: "batch"}})
        print(f"\nSaved model/{out}.onnx")
    except Exception as e:
        print(f"\nONNX export skipped: {e}")

    torch.save(model.state_dict(), f"model/{out}.pt")
    preprocess = {
        "features": FEATURES,
        "negative": NEGATIVE,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "sub_mean": stats["sub_mean"].tolist(),
        "sub_std": stats["sub_std"].tolist(),
        "signs": signs.tolist(),
        "percentile_grid": grid.tolist(),
        "dims": DIMS,
        "dim_weights": WEIGHTS.tolist(),
        "sleep": {
            "feature": SLEEP,
            "target_hours": SLEEP_TARGET,
            "rest_mean": float(stats["rest_mean"]),
            "rest_std": float(stats["rest_std"]),
            "weight": REST_WEIGHT,
            "formula": "rest = -abs(sleep_hours - target_hours)",
        },
    }
    with open(f"model/{out}_preprocess.json", "w") as f:
        json.dump(preprocess, f, indent=1)
    print(f"Saved model/{out}.pt and model/{out}_preprocess.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="wellbeing_index")
    args = ap.parse_args()

    X, s, y = load(args.csv)
    print(f"{len(X)} usable rows, {len(FEATURES)} autoencoder items + {SLEEP}")
    Xtr, Xte, s_tr, s_te, ytr, yte = train_test_split(X, s, y, test_size=0.2, random_state=0)
    Xtr, Xval, s_tr, _, ytr, _ = train_test_split(Xtr, s_tr, ytr, test_size=0.15, random_state=0)
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zval, Zte = (scaler.transform(a) for a in (Xtr, Xval, Xte))

    pca, fa = baselines(Ztr, Zte)
    sleep_table(s_tr, ytr)
    if args.baseline_only:
        return

    GroupedAE, FreeAE = build_models()
    model, sub_te, signs = train_grouped(GroupedAE, Ztr, Zval, Zte, args.seeds)
    compare_with_free(FreeAE, model, pca, fa, Ztr, Zval, Zte, sub_te)

    sub_tr = encode(model, Ztr) * signs
    rest_tr, rest_te = rest_score(s_tr), rest_score(s_te)
    stats = {"sub_mean": sub_tr.mean(0), "sub_std": sub_tr.std(0),
             "rest_mean": rest_tr.mean(), "rest_std": rest_tr.std()}

    raw_tr, raw_te = combine(sub_tr, rest_tr, stats), combine(sub_te, rest_te, stats)
    grid = np.percentile(raw_tr, np.arange(101))
    index_te = np.interp(raw_te, grid, np.arange(101))

    report_validation(index_te, sub_te, rest_te, s_te, yte, stats)
    report_context(args.csv, Xte, index_te, ytr, yte)
    export(model, signs, scaler, stats, grid, args.out)


if __name__ == "__main__":
    main()
