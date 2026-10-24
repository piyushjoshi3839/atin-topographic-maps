#!/usr/bin/env python3
"""
diagnose.py MODEL.pt OUT_PREFIX  - failure diagnosis and change detection with a trained LunarLander ATIN controller.

All settings below were fixed before any result was seen.

A) Failure diagnosis (nominal LunarLander, 300 episodes, 10% random actions to create some crashes)
   - "pre-crash" states = last 30 steps of episodes that end in a crash (reward -100)
   - per map cell, pre-crash rate is estimated on even episodes, tested on odd episodes:
       AUROC of "cell crash rate" as a predictor of pre-crash states (state level)
   - same with k-means cells (225 clusters, raw standardised state) as an unordered reference
   - Moran's I of cell crash rates on the grid (8-neighbourhood, cells with >=10 states),
       with a permutation p-value: is risk spatially contiguous on the map?
B) Change detection
   - reference: 100 nominal episodes (5% random actions); test: 100 new nominal episodes and
     100 episodes in each changed condition: wind, low gravity, sensor noise, engine fault
   - episode score = mean per-step score over the first 50 steps; AUROC nominal-vs-changed
   - detectors: ATIN prototype distance, ATIN forward-model surprise, kNN distance in raw
     state, kNN distance in encoder features, separate dynamics-model error (MLP on reference)
Output: OUT_PREFIX_results.json, OUT_PREFIX_cells.npz
"""
import sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, gymnasium as gym
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor

model_path, out = sys.argv[1:3]
import os; os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
torch.set_num_threads(1)
G, ZD, PD, NA = 15, 20, 16, 4

def mlp(s):
    L = []
    for a, b in zip(s[:-1], s[1:]): L += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*L[:-1])

ck = torch.load(model_path, map_location="cpu")
enc, qh = mlp([8, 128, 64, ZD]), mlp([ZD, 256, 256, NA])
enc.load_state_dict({k[8:]: v for k, v in ck["net"].items() if k.startswith("encoder.")})
qh.load_state_dict({k[2:]: v for k, v in ck["net"].items() if k.startswith("q.")})
A = ck["atin"]
proj = nn.Linear(ZD, PD); proj.load_state_dict({"weight": A["proj.weight"], "bias": A["proj.bias"]})
fwd = mlp([ZD + NA, 64, PD]); fwd.load_state_dict({k[4:]: v for k, v in A.items() if k.startswith("fwd.")})
P = F.normalize(A["prototypes"], dim=-1)

CONDITIONS = {
    "nominal": dict(env={}, obs_noise=0.0, fault=0.0),
    "wind": dict(env=dict(enable_wind=True, wind_power=15.0, turbulence_power=1.5), obs_noise=0.0, fault=0.0),
    "low_gravity": dict(env=dict(gravity=-4.0), obs_noise=0.0, fault=0.0),
    "sensor_noise": dict(env={}, obs_noise=0.1, fault=0.0),
    "engine_fault": dict(env={}, obs_noise=0.0, fault=0.3),
}

def rollout(cond, n_eps, seed0, rand_p):
    c = CONDITIONS[cond]; env = gym.make("LunarLander-v3", **c["env"]); rng = np.random.default_rng(seed0)
    S, A_, S2, EP, T, OUT = [], [], [], [], [], []
    for e in range(n_eps):
        o, _ = env.reset(seed=seed0 + e); seen = o + c["obs_noise"] * rng.standard_normal(8).astype(np.float32)
        done, t, last_r = False, 0, 0.0
        while not done:
            with torch.no_grad():
                a = int(qh(enc(torch.as_tensor(seen, dtype=torch.float32)[None])).argmax())
            if rng.random() < rand_p: a = int(rng.integers(NA))
            a_exec = 0 if (c["fault"] > 0 and a in (1, 2, 3) and rng.random() < c["fault"]) else a  # engine fault: command lost
            o2, r, te, tr, _ = env.step(a_exec); done = te or tr; last_r = r
            seen2 = o2 + c["obs_noise"] * rng.standard_normal(8).astype(np.float32)
            S.append(seen); A_.append(a); S2.append(seen2); EP.append(e); T.append(t)
            seen = seen2; t += 1
        OUT.append("crash" if last_r == -100 else ("landed" if last_r == 100 else "timeout"))
    return dict(S=np.array(S, np.float32), A=np.array(A_), S2=np.array(S2, np.float32), EP=np.array(EP), T=np.array(T), OUT=OUT)

@torch.no_grad()
def feats(d):
    s = torch.as_tensor(d["S"]); s2 = torch.as_tensor(d["S2"])
    z, z2 = enc(s), enc(s2)
    u, u2 = F.normalize(proj(z), dim=-1), F.normalize(proj(z2), dim=-1)
    cos = u @ P.t()
    pred = F.normalize(fwd(torch.cat([z, F.one_hot(torch.as_tensor(d["A"]), NA).float()], -1)), dim=-1)
    return dict(z=z.numpy(), cell=cos.argmax(-1).numpy(), proto_dist=(1 - cos.max(-1).values).numpy(),
                fwd_surprise=(1 - (pred * u2).sum(-1)).numpy())

res = {}
# ---------------- A) failure diagnosis
D = rollout("nominal", 300, 50_000, 0.10); f = feats(D)
crash_ep = np.array([o == "crash" for o in D["OUT"]])
ep_len = np.bincount(D["EP"])
pre = crash_ep[D["EP"]] & (D["T"] >= ep_len[D["EP"]] - 30)
train, test = (D["EP"] % 2 == 0), (D["EP"] % 2 == 1)

def cell_auc(cells, ncell):
    n = np.bincount(cells[train], minlength=ncell); k = np.bincount(cells[train], weights=pre[train], minlength=ncell)
    rate = (k + pre[train].mean()) / (n + 1)                 # shrink towards the base rate
    return float(roc_auc_score(pre[test], rate[cells[test]])) if pre[test].any() and (~pre[test]).any() else None, rate, n

auc_map, rate_map, n_map = cell_auc(f["cell"], G * G)
mu, sd = D["S"][train].mean(0), D["S"][train].std(0) + 1e-6
km = KMeans(G * G, n_init=3, random_state=0).fit((D["S"][train] - mu) / sd)
auc_km, _, _ = cell_auc(km.predict((D["S"] - mu) / sd), G * G)

def morans_i(vals, ok, rng):
    idx = np.where(ok)[0]; r, c = idx // G, idx % G
    W = ((np.abs(r[:, None] - r[None]) <= 1) & (np.abs(c[:, None] - c[None]) <= 1)).astype(float); np.fill_diagonal(W, 0)
    def I(x):
        x = x - x.mean(); return len(x) / W.sum() * (x @ W @ x) / (x @ x)
    x = vals[idx]; i0 = I(x); perm = np.array([I(rng.permutation(x)) for _ in range(2000)])
    return float(i0), float((perm >= i0).mean())

full_n = np.bincount(f["cell"], minlength=G * G); full_k = np.bincount(f["cell"], weights=pre, minlength=G * G)
full_rate = np.where(full_n > 0, full_k / np.maximum(full_n, 1), np.nan)
mI, mp = morans_i(np.nan_to_num(full_rate), full_n >= 10, np.random.default_rng(0))
res["diagnosis"] = dict(episodes=300, crash_episodes=int(crash_ep.sum()), prec_states=int(pre.sum()),
                        auroc_map_cells=auc_map, auroc_kmeans_cells=auc_km, morans_I=mI, morans_p=mp,
                        cells_used=int((full_n >= 10).sum()))
np.savez(out + "_cells.npz", rate=full_rate, n=full_n, S=D["S"], cell=f["cell"], pre=pre)

# ---------------- B) change detection
R = rollout("nominal", 100, 20_000, 0.05); fr = feats(R)
mu_r, sd_r = R["S"].mean(0), R["S"].std(0) + 1e-6
sub = np.random.default_rng(0).choice(len(R["S"]), min(20_000, len(R["S"])), replace=False)
nn_raw = NearestNeighbors(n_neighbors=5).fit(((R["S"] - mu_r) / sd_r)[sub])
zmu, zsd = fr["z"].mean(0), fr["z"].std(0) + 1e-6
nn_lat = NearestNeighbors(n_neighbors=5).fit(((fr["z"] - zmu) / zsd)[sub])
X = np.hstack([(R["S"] - mu_r) / sd_r, np.eye(NA)[R["A"]]]); Y = (R["S2"] - R["S"]) / sd_r
dyn = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, random_state=0, early_stopping=True).fit(X[sub], Y[sub])
res_sd = (Y[sub] - dyn.predict(X[sub])).std(0) + 1e-6

def scores(d):
    fd = feats(d)
    sr = (d["S"] - mu_r) / sd_r
    out = dict(proto_dist=fd["proto_dist"], fwd_surprise=fd["fwd_surprise"],
               knn_raw=nn_raw.kneighbors(sr)[0][:, -1], knn_latent=nn_lat.kneighbors((fd["z"] - zmu) / zsd)[0][:, -1])
    Xd = np.hstack([sr, np.eye(NA)[d["A"]]]); Yd = (d["S2"] - d["S"]) / sd_r
    out["dyn_model"] = np.linalg.norm((Yd - dyn.predict(Xd)) / res_sd, axis=1)
    early = d["T"] < 50
    return {k: np.array([v[(d["EP"] == e) & early].mean() for e in np.unique(d["EP"])]) for k, v in out.items()}

base = rollout("nominal", 100, 30_000, 0.05); sb = scores(base)
res["detection"] = {}; res["outcomes"] = {"nominal": {o: base["OUT"].count(o) for o in set(base["OUT"])}}
for cond in ["wind", "low_gravity", "sensor_noise", "engine_fault"]:
    d = rollout(cond, 100, 40_000, 0.05); sc = scores(d)
    res["outcomes"][cond] = {o: d["OUT"].count(o) for o in set(d["OUT"])}
    res["detection"][cond] = {k: float(roc_auc_score(np.r_[np.zeros(len(sb[k])), np.ones(len(sc[k]))], np.r_[sb[k], sc[k]])) for k in sb}
json.dump(res, open(out + "_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
