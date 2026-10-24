#!/usr/bin/env python3
"""
reproduce_tables.py [RESULTS_DIR]  - recompute every number in the paper's tables from the per-seed logs.

Final return   : mean of the last three greedy evaluations of each seed, averaged over seeds
Mean over training: mean of all greedy evaluations of each seed, averaged over seeds
CI             : 95% bootstrap confidence interval of the mean over seeds (4000 resamples, seed 0)
Map metrics    : from the *_topo.json file written at the end of each run
"""
import glob, json, os, sys
import numpy as np, pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "results"
rng = np.random.default_rng(0)

def ci(x):
    x = np.asarray(x, float)
    m = [rng.choice(x, len(x)).mean() for _ in range(4000)]
    return np.percentile(m, [2.5, 97.5])

def fmt(x):
    lo, hi = ci(x)
    return f"{np.mean(x):7.1f} [{lo:7.1f}, {hi:7.1f}]"

def evals(pattern):
    fs = sorted(glob.glob(os.path.join(R, pattern)))
    return [pd.read_csv(f) for f in fs]

def final(E): return np.array([e.mean_return.values[-3:].mean() for e in E])
def auc(E): return np.array([e.mean_return.mean() for e in E])
def reached(E, thr): return sum(bool((e.mean_return >= thr).any()) for e in E)

def topo(pattern):
    T = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, pattern)))]
    return {k: np.array([t[k] for t in T]) for k in T[0] if k != "num_prototypes"}, len(T)

def line(name, E, thr=None):
    s = f"  {name:34s} n={len(E):2d}  final {fmt(final(E))}  mean-over-training {fmt(auc(E))}"
    if thr is not None: s += f"  reached {reached(E, thr)}/{len(E)}"
    print(s)

print("=== Table 2: control performance ===")
line("LunarLander DQN", evals("main/LunarLander-v3/dqn_egreedy_s*_eval.csv"), 200)
line("LunarLander ATIN", evals("main/LunarLander-v3/atin_v2_egreedy_s*_eval.csv"), 200)
line("LunarLander DDQN (untuned)", evals("ddqn/LunarLander-v3/ddqn_dueling_s*_eval.csv"), 200)
line("LunarLander Proto (no grid)", evals("proto_control/LunarLander-v3/proto_egreedy_s*_eval.csv"), 200)
line("MountainCar DQN", evals("main/MountainCar-v0/dqn_egreedy_s*_eval.csv"), -110)
line("MountainCar ATIN", evals("main/MountainCar-v0/atin_v2_egreedy_s*_eval.csv"), -110)
for ag in ["dqn", "atin_v2"]:   # Acrobot: online return of training episodes in the last 20k steps
    v = [pd.read_csv(f).query("step > 80000")["return"].mean()
         for f in sorted(glob.glob(os.path.join(R, f"acrobot/Acrobot-v1/{ag}_egreedy_s*_train.csv")))]
    print(f"  {'Acrobot ' + ag + ' (online, last 20k)':34s} n={len(v):2d}  final {fmt(v)}")
line("Pendulum SAC", evals("pendulum/sac_s*_eval.csv"), -200)
line("Pendulum SAC+ATIN", evals("pendulum/atin_s*_eval.csv"), -200)
for ag in ["dqn", "atin_v2"]:
    g = [int((pd.read_csv(f).length < 200).sum()) for f in sorted(glob.glob(os.path.join(R, f"main/MountainCar-v0/{ag}_egreedy_s*_train.csv")))]
    print(f"  MountainCar {ag}: training episodes reaching the goal per seed {g}")

print("\n=== Table 3: map quality (mean +- s.d.; MountainCar also median) ===")
for name, pat in [("ATIN-v1 LunarLander 30k", "v1_vs_v2_30k/atin_v1_s*_topo.json"),
                  ("ATIN LunarLander 30k", "v1_vs_v2_30k/atin_v2_s*_topo.json"),
                  ("ATIN-v1 LunarLander 200k", "v1_200k/atin_v1_s*_topo.json"),
                  ("ATIN LunarLander 200k", "main/LunarLander-v3/atin_v2_egreedy_s*_topo.json"),
                  ("ATIN MountainCar 150k", "main/MountainCar-v0/atin_v2_egreedy_s*_topo.json"),
                  ("SAC+ATIN Pendulum 30k", "pendulum/atin_s*_topo.json"),
                  ("ATIN Acrobot 100k", "acrobot/Acrobot-v1/atin_v2_egreedy_s*_topo.json"),
                  ("Proto (no grid) LunarLander 200k", "proto_control/LunarLander-v3/proto_egreedy_s*_topo.json")]:
    t, n = topo(pat)
    te = t["topographic_error"]
    s = (f"  {name:34s} n={n:2d}  topo-error {te.mean():.3f}+-{te.std():.3f} (median {np.median(te):.3f})"
         f"  used {t['prototypes_used_frac'].mean():.0%}  effective {t['usage_perplexity'].mean():.0f}+-{t['usage_perplexity'].std():.0f}")
    if "mean_max_assignment" in t: s += f"  largest-weight {t['mean_max_assignment'].mean():.3f}"
    print(s)

print("\n=== Table 4: four-term vs two-term objective (seeds 1-5) ===")
for env, four, two in [("LunarLander-v3", "main/LunarLander-v3/atin_v2_egreedy_s{}", "two_term/LunarLander-v3/atin2term_egreedy_s{}"),
                       ("MountainCar-v0", "main/MountainCar-v0/atin_v2_egreedy_s{}", "two_term/MountainCar-v0/atin2term_egreedy_s{}"),
                       ("Pendulum-v1", "pendulum/atin_s{}", "two_term/Pendulum-v1/atin2term_s{}")]:
    for lab, p in [("four-term", four), ("two-term", two)]:
        E = [pd.read_csv(os.path.join(R, p.format(s) + "_eval.csv")) for s in range(1, 6)]
        T = [json.load(open(os.path.join(R, p.format(s) + "_topo.json"))) for s in range(1, 6)]
        print(f"  {env:15s} {lab:9s} topo-error {np.mean([t['topographic_error'] for t in T]):.3f}"
              f"  used {np.mean([t['prototypes_used_frac'] for t in T]):.0%}  final {fmt(final(E))}")

print("\n=== Table 5: ablation, LunarLander 200k, seeds 1-3 ===")
rows = [("full objective", "main/LunarLander-v3/atin_v2_egreedy_s{}")] + \
       [(v, "ablation_200k/" + v + "_s{}") for v in ["no_som", "no_assign", "no_temp", "no_fwd", "plus_div"]]
for name, p in rows:
    T = [json.load(open(os.path.join(R, p.format(s) + "_topo.json"))) for s in (1, 2, 3)]
    F = [pd.read_csv(os.path.join(R, p.format(s) + "_eval.csv")).mean_return.values[-3:].mean() for s in (1, 2, 3)]
    te = np.array([t["topographic_error"] for t in T]); pp = np.array([t["usage_perplexity"] for t in T])
    print(f"  {name:15s} topo-error {te.mean():.3f}+-{te.std():.3f}  used {np.mean([t['prototypes_used_frac'] for t in T]):.0%}"
          f"  effective {pp.mean():.0f}+-{pp.std():.0f}  largest-weight {np.mean([t['mean_max_assignment'] for t in T]):.3f}"
          f"  final {np.mean(F):.0f}+-{np.std(F):.0f}")

print("\n=== Table 6: MountainCar exploration (bonus in reward, seeds 1-5) ===")
for name, p in [("ATIN prototype counts, beta 1", "explore_count/MountainCar-v0/atin_count_b1_s{}"),
                ("SimHash counts, beta 1", "explore_count/MountainCar-v0/simhash_count_b1_s{}"),
                ("SimHash counts, beta 3", "explore_count/MountainCar-v0/simhash_count_b3_s{}"),
                ("Proto kNN novelty, beta 3", "proto_explore/MountainCar-v0/proto_knn_b3_s{}")]:
    g = [int((pd.read_csv(os.path.join(R, p.format(s) + "_train.csv")).length < 200).sum()) for s in range(1, 6)]
    E = [pd.read_csv(os.path.join(R, p.format(s) + "_eval.csv")) for s in range(1, 6)]
    print(f"  {name:30s} found goal {sum(x > 0 for x in g)}/5  goal episodes {g}  final {fmt(final(E))}")

print("\n=== Section 5.5 / Table 7: diagnosis and change detection (5 controllers) ===")
D = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "diagnosis/atin_s*_results.json")))]
for k in ["auroc_map_cells", "auroc_kmeans_cells", "morans_I"]:
    v = [d["diagnosis"][k] for d in D]
    print(f"  {k:22s} {np.mean(v):.3f}+-{np.std(v):.3f}")
print(f"  Moran p<0.05 in {sum(d['diagnosis']['morans_p'] < 0.05 for d in D)}/{len(D)} controllers;"
      f" crash episodes per controller {[d['diagnosis']['crash_episodes'] for d in D]}")
dets = ["proto_dist", "fwd_surprise", "knn_raw", "knn_latent", "dyn_model"]
print("  condition      " + "  ".join(f"{d:>13s}" for d in dets))
for c in ["wind", "low_gravity", "sensor_noise", "engine_fault"]:
    print(f"  {c:14s} " + "  ".join(f"{np.mean([x['detection'][c][d] for x in D]):.2f}+-{np.std([x['detection'][c][d] for x in D]):.2f}".rjust(13) for d in dets))
