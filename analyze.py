#!/usr/bin/env python3
"""
analyze.py RUNS_DIR  - aggregate multi-seed results.
Outputs (inside RUNS_DIR):
  curves_<env>.png   greedy-evaluation return vs env steps, mean +/- 95% bootstrap CI over seeds
  summary.csv        per (env, agent): final return, area under curve (AUC), steps-to-threshold
Thresholds: LunarLander 200 ("solved"), MountainCar -110 ("solved").
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = sys.argv[1] if len(sys.argv) > 1 else "runs"
THRESH = {"LunarLander-v3": 200.0, "MountainCar-v0": -110.0, "Acrobot-v1": -100.0}
rng = np.random.default_rng(0)

def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) < 2: return (np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)

rows = []
for env in sorted(os.listdir(root)):
    edir = os.path.join(root, env)
    if not os.path.isdir(edir): continue
    groups = {}
    for f in glob.glob(os.path.join(edir, "*_eval.csv")):
        name = re.sub(r"_s\d+_eval\.csv$", "", os.path.basename(f))
        df = pd.read_csv(f)
        if len(df): groups.setdefault(name, []).append(df)
    if not groups: continue
    plt.figure(figsize=(6.4, 4))
    for name, dfs in sorted(groups.items()):
        L = min(len(d) for d in dfs)
        steps = dfs[0]["step"].values[:L]
        M = np.stack([d["mean_return"].values[:L] for d in dfs])     # seeds x evals
        mean = M.mean(0); ci = np.array([boot_ci(M[:, j]) for j in range(L)])
        plt.plot(steps, mean, label=f"{name} (n={len(dfs)})")
        if len(dfs) > 1: plt.fill_between(steps, ci[:, 0], ci[:, 1], alpha=0.2)
        final = M[:, -3:].mean(1)                     # mean of last 3 evals per seed
        auc = M.mean(1)                               # mean eval return over training per seed
        thr = THRESH.get(env)
        hit = []
        for srow in M:
            k = np.argmax(srow >= thr) if thr is not None and (srow >= thr).any() else None
            hit.append(steps[k] if k is not None else np.nan)
        topo = [json.load(open(f)) for f in glob.glob(os.path.join(edir, name + "_s*_topo.json"))]
        r = dict(env=env, agent=name, seeds=len(dfs), steps=int(steps[-1]),
                 final_mean=final.mean(), final_ci_lo=boot_ci(final)[0], final_ci_hi=boot_ci(final)[1],
                 auc_mean=auc.mean(), auc_ci_lo=boot_ci(auc)[0], auc_ci_hi=boot_ci(auc)[1],
                 seeds_reaching_thr=int(np.sum(~np.isnan(hit))),
                 median_steps_to_thr=np.nanmedian(hit) if np.any(~np.isnan(hit)) else np.nan)
        for k in ["topographic_error", "prototypes_used_frac", "usage_perplexity"]:
            if topo: r[k] = np.mean([t[k] for t in topo])
        rows.append(r)
    if env in THRESH: plt.axhline(THRESH[env], ls="--", c="gray", lw=0.8)
    plt.xlabel("environment steps"); plt.ylabel("greedy evaluation return")
    plt.title(f"{env}: mean and 95% CI over seeds"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(root, f"curves_{env}.png"), dpi=150); plt.close()

out = pd.DataFrame(rows)
out.to_csv(os.path.join(root, "summary.csv"), index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print(out.round(3).to_string(index=False))
