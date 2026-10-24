#!/usr/bin/env python3
"""
atin.py - DQN experiments for "Topographic state maps for inspecting deep reinforcement learning controllers".

Agents (--agent):
  dqn     : plain DQN (encoder + Q-head). The control: same network as ATIN, no auxiliary losses.
            With --double / --dueling it becomes the DDQN reference.
  atin_v2 : ATIN, the topographic head of the paper:
              - assignments on the unit sphere (cosine similarity / temperature)
              - L_assign : sharp per-state assignment + balanced prototype usage
              - L_som    : SOM neighbourhood loss (winner and its grid neighbours move to the state)
              - L_temp   : grid-position smoothness on the true (s_t, s_t+1) pair of each transition
              - L_fwd    : action-conditioned forward model on the true (s_t, s_t+1) pair
              - L_div    : squared-cosine diversity (off by default, --w-div)
            Terms can be switched off for ablations (--no-som, --no-assign, --no-temp, --no-fwd).
  atin_v1 : the original nine-term objective analysed in Sec. 3.5 of the paper (it collapses).
  proto   : Proto-RL-style baseline: unordered prototypes trained with a SwAV/Sinkhorn loss on
            (s_t, s_t+1) pairs, no grid and no SOM loss.

Exploration (--explore):
  egreedy : epsilon-greedy, linear decay over environment steps
  ucb     : epsilon-greedy + count bonus at action selection (counts over prototypes or SimHash)
  bonus   : count bonus beta/sqrt(N) added to the reward (#Exploration style); --counter picks
            ATIN prototypes (proto) or SimHash buckets of the raw state (simhash)
  knn     : Proto-RL-style novelty bonus (distance to k-th nearest recent state), --agent proto only

Outputs (all measured in environment steps):
  <out>_train.csv : step, episode, return, length   (one row per training episode)
  <out>_eval.csv  : step, mean_return, std_return    (greedy evaluation every --eval-every steps)
  <out>_topo.json : map metrics at the end of training (prototype-based agents)
  <out>_args.json, <out>_model.pt

Example:
  python atin.py --env LunarLander-v3 --agent atin_v2 --seed 1 --total-steps 200000 \
                 --out runs/main/LunarLander-v3/atin_v2_egreedy_s1
"""
import argparse, copy, csv, json, math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--env", default="LunarLander-v3")
p.add_argument("--agent", default="atin_v2", choices=["dqn", "atin_v1", "atin_v2", "proto"])
p.add_argument("--explore", default="egreedy", choices=["egreedy", "ucb", "bonus", "knn"],
               help="ucb: count bonus added at action selection; bonus: count bonus added to the reward (#Exploration-style)")
p.add_argument("--counter", default="auto", choices=["auto", "proto", "simhash"],
               help="what to count: ATIN prototypes or SimHash buckets of the raw state (auto: proto for ATIN, simhash for DQN)")
p.add_argument("--bonus-beta", type=float, default=0.1, help="intrinsic reward scale for --explore bonus")
p.add_argument("--hash-bits", type=int, default=None, help="SimHash bits (default: ~log2 of the number of prototypes)")
p.add_argument("--knn-k", type=int, default=3, help="k for the Proto-RL-style kNN novelty bonus (--explore knn)")
p.add_argument("--knn-queue", type=int, default=1024, help="recent states kept for the kNN novelty bonus")
p.add_argument("--double", action="store_true", help="double DQN target")
p.add_argument("--dueling", action="store_true", help="dueling value head")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--total-steps", type=int, default=200_000)
p.add_argument("--out", default="runs/run")
p.add_argument("--device", default="cpu")
p.add_argument("--threads", type=int, default=1)
# DQN (same for every agent)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--buffer", type=int, default=200_000)
p.add_argument("--learning-starts", type=int, default=1_000)
p.add_argument("--train-every", type=int, default=1)
p.add_argument("--target-update", type=int, default=1_000)
p.add_argument("--eps-start", type=float, default=1.0)
p.add_argument("--eps-end", type=float, default=0.01)
p.add_argument("--eps-decay-steps", type=int, default=50_000)
p.add_argument("--eval-every", type=int, default=10_000)
p.add_argument("--eval-episodes", type=int, default=5)
# network sizes (None -> per-environment defaults, see ENV_DEFAULTS)
p.add_argument("--z-dim", type=int, default=None)
p.add_argument("--grid", type=int, default=None, help="grid side; N = grid*grid prototypes")
# ATIN common
p.add_argument("--aux-weight", type=float, default=0.5)
# ATIN v2
p.add_argument("--proj-dim", type=int, default=16)
p.add_argument("--tau", type=float, default=0.1, help="assignment temperature")
p.add_argument("--sigma-start", type=float, default=None, help="SOM radius at start (default grid/2)")
p.add_argument("--sigma-end", type=float, default=1.0)
p.add_argument("--sigma-anneal-steps", type=int, default=100_000)
p.add_argument("--w-som", type=float, default=3.0)
p.add_argument("--w-assign", type=float, default=1.0)
p.add_argument("--w-div", type=float, default=0.0, help="off by default: conflicts with topology; balanced usage already prevents collapse")
p.add_argument("--w-temp", type=float, default=1.0)
p.add_argument("--w-fwd", type=float, default=1.0)
for t in ["som", "assign", "div", "temp", "fwd"]:
    p.add_argument(f"--no-{t}", action="store_true")
# UCB
p.add_argument("--ucb-c", type=float, default=0.75)
p.add_argument("--ucb-decay", type=float, default=0.9999, help="per-step count decay (1.0 = none)")
args = p.parse_args()

ENV_DEFAULTS = {"LunarLander-v3": dict(z_dim=20, grid=15), "MountainCar-v0": dict(z_dim=8, grid=10),
                "Acrobot-v1": dict(z_dim=16, grid=10)}
d = ENV_DEFAULTS.get(args.env, dict(z_dim=16, grid=10))
args.z_dim = args.z_dim or d["z_dim"]
args.grid = args.grid or d["grid"]
if args.sigma_start is None:
    args.sigma_start = args.grid / 2.0
if args.counter == "auto":
    args.counter = "simhash" if args.agent == "dqn" else "proto"
if args.explore == "knn" and args.agent != "proto":
    raise SystemExit("--explore knn is the Proto-RL-style bonus; use --agent proto")
if args.counter == "proto" and args.agent == "dqn":
    raise SystemExit("prototype counts need an ATIN agent; use --counter simhash")
if args.hash_bits is None:
    args.hash_bits = int(round(math.log2(args.grid ** 2)))

torch.set_num_threads(args.threads)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device(args.device)


# --------------------------------------------------------------------------------------
# Networks (identical for all agents)
# --------------------------------------------------------------------------------------
def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class DuelingHead(nn.Module):
    def __init__(self, z_dim, n_actions):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(z_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.v, self.a = nn.Linear(256, 1), nn.Linear(256, n_actions)

    def forward(self, z):
        h = self.body(z); a = self.a(h)
        return self.v(h) + a - a.mean(-1, keepdim=True)


class QAgent(nn.Module):
    """encoder (obs -> z) + Q-head (z -> Q)."""
    def __init__(self, obs_dim, n_actions, z_dim, dueling=False):
        super().__init__()
        self.encoder = mlp([obs_dim, 128, 64, z_dim])
        self.q = DuelingHead(z_dim, n_actions) if dueling else mlp([z_dim, 256, 256, n_actions])

    def forward(self, obs):
        return self.q(self.encoder(obs))


def grid_coords(g):
    ii, jj = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    return torch.stack([ii.flatten(), jj.flatten()], 1).float()  # (N,2)


# --------------------------------------------------------------------------------------
# ATIN v1: original loss, ported unchanged (random-neighbour "temporal" pairs etc.)
# --------------------------------------------------------------------------------------
class AtinV1(nn.Module):
    def __init__(self, z_dim, g, beta_start, beta_end=10.0, beta_anneal=50_000):
        super().__init__()
        self.N = g * g
        self.prototypes = nn.Parameter(torch.randn(self.N, z_dim) * 0.05)
        self.proto_values = nn.Parameter(torch.zeros(self.N))
        c = grid_coords(g)
        self.register_buffer("grid_d2_init", ((c[:, None] - c[None]) ** 2).sum(-1))
        self.proto_coords = nn.Parameter(c.clone())
        self.forward_model = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(), nn.Linear(z_dim, z_dim))
        self.neigh_transform = nn.Linear(z_dim, z_dim, bias=False)
        self.b0, self.b1, self.banneal, self.step = beta_start, beta_end, beta_anneal, 0
        self.beta = beta_start

    def assign(self, z):
        d2 = ((z[:, None] - self.prototypes[None]) ** 2).sum(-1)
        return F.softmax(-self.beta * d2, -1), d2

    def loss(self, z, z_next, act, reward):
        # NOTE: z_next / act unused on purpose -> original code used z[t] vs z[t+1] in the batch
        q, d2 = self.assign(z)
        coords = self.proto_coords
        kd2 = ((coords[:, None] - coords[None]) ** 2).sum(-1)
        K = torch.exp(-kd2 / 2.0)
        smooth = F.mse_loss(q[1:], q[:-1])
        ent = -(q * torch.log(q + 1e-8)).sum(-1).mean()
        pn = F.normalize(self.prototypes, dim=-1)
        div = (pn @ pn.t() - torch.eye(self.N)).mean()
        fwd = F.mse_loss(self.forward_model(z[:-1]), z[1:])
        attr = (q * d2).sum(-1).mean()
        pd2 = ((self.prototypes[:, None] - self.prototypes[None]) ** 2).sum(-1)
        topo = (K * pd2).sum() / (K.sum() + 1e-12)
        ec = q @ coords
        gsm = F.mse_loss(ec[1:], ec[:-1])
        rl = F.mse_loss(q @ self.proto_values, reward)
        agg = (K / (K.sum(1, keepdim=True) + 1e-12)) @ self.neigh_transform(self.prototypes)
        msg = F.mse_loss(self.prototypes, agg)
        creg = F.mse_loss(kd2, self.grid_d2_init)
        L = (0.8 * smooth - 0.5 * ent + 0.1 * div + 0.5 * fwd + 0.7 * attr + 0.25 * topo
             + 0.6 * gsm + 0.5 * rl + 0.81 * msg + 0.094 * creg)
        self.step += 1
        self.beta = self.b0 + (self.b1 - self.b0) * min(1.0, self.step / self.banneal)
        return L, {}

    @torch.no_grad()
    def winners(self, z):
        q, _ = self.assign(z)
        return q.topk(2, -1).indices, q


# --------------------------------------------------------------------------------------
# ATIN v2: fixed losses
# --------------------------------------------------------------------------------------
class AtinV2(nn.Module):
    def __init__(self, z_dim, g, n_actions, proj_dim, tau):
        super().__init__()
        self.g, self.N, self.tau, self.nA = g, g * g, tau, n_actions
        self.proj = nn.Linear(z_dim, proj_dim)                      # ATIN head on top of encoder
        self.prototypes = nn.Parameter(F.normalize(torch.randn(self.N, proj_dim), dim=-1))
        c = grid_coords(g)
        self.register_buffer("coords", c)
        self.register_buffer("grid_d2", ((c[:, None] - c[None]) ** 2).sum(-1))
        self.fwd = mlp([z_dim + n_actions, 64, proj_dim])           # (z_t, a_t) -> u_{t+1}
        self.sigma = args.sigma_start
        self.step = 0

    def embed(self, z):
        return F.normalize(self.proj(z), dim=-1)

    def protos(self):
        return F.normalize(self.prototypes, dim=-1)

    def logits(self, u):
        return (u @ self.protos().t()) / self.tau

    def loss(self, z, z_next, act, reward):
        u, P = self.embed(z), self.protos()
        cos = u @ P.t()                                    # (B,N)
        q = F.softmax(cos / self.tau, -1)
        logs = {}
        L = torch.zeros((), device=z.device)

        if not args.no_assign:   # sharp per state, balanced over the batch
            h_state = -(q * torch.log(q + 1e-8)).sum(-1).mean()
            qbar = q.mean(0)
            h_marg = -(qbar * torch.log(qbar + 1e-8)).sum()
            l = h_state - h_marg
            L = L + args.w_assign * l; logs["assign"] = l.item()

        if not args.no_som:      # SOM: winner + grid neighbours move toward the state
            win = cos.argmax(-1)
            K = torch.exp(-self.grid_d2[win] / (2 * self.sigma ** 2))   # (B,N)
            K = K / K.sum(-1, keepdim=True)
            l = (K * (1 - cos)).sum(-1).mean()
            L = L + args.w_som * l; logs["som"] = l.item()

        if not args.no_div and args.w_div > 0:      # squared cosine: +u/-u no longer cheats
            S = P @ P.t()
            l = (S - torch.eye(self.N, device=z.device)).pow(2).sum() / (self.N * (self.N - 1))
            L = L + args.w_div * l; logs["div"] = l.item()

        if not args.no_temp or not args.no_fwd:
            with torch.no_grad():
                u_next = self.embed(z_next)                # target: same transition's next state
        if not args.no_temp:     # expected grid position should move little between s_t and s_t+1
            q_next = F.softmax((self.embed(z_next) @ P.t()) / self.tau, -1)
            l = ((q @ self.coords - q_next @ self.coords) ** 2).sum(-1).mean() / (self.g ** 2)
            L = L + args.w_temp * l; logs["temp"] = l.item()

        if not args.no_fwd:      # action-conditioned forward model, true pairs
            a1 = F.one_hot(act, self.nA).float()
            pred = F.normalize(self.fwd(torch.cat([z, a1], -1)), dim=-1)
            l = (1 - (pred * u_next).sum(-1)).mean()
            L = L + args.w_fwd * l; logs["fwd"] = l.item()

        self.step += 1
        fr = min(1.0, self.step / args.sigma_anneal_steps)
        self.sigma = args.sigma_start * (args.sigma_end / args.sigma_start) ** fr   # exponential anneal
        return L, logs

    @torch.no_grad()
    def winners(self, z):
        cos = self.embed(z) @ self.protos().t()
        return cos.topk(2, -1).indices, F.softmax(cos / self.tau, -1)


# --------------------------------------------------------------------------------------
# Proto-RL-style baseline (adapted to state vectors): unordered prototypes, no grid / SOM.
# SwAV-style loss: the online prediction for s_t must match the Sinkhorn-balanced prototype
# assignment of s_{t+1} (stop-grad), as in Proto-RL (Yarats et al., 2021). Same number of
# prototypes, embedding size and temperature as ATIN so that only the grid/SOM differs.
# --------------------------------------------------------------------------------------
class ProtoRL(nn.Module):
    def __init__(self, z_dim, g, proj_dim, tau, sk_eps=0.05, sk_iters=3):
        super().__init__()
        self.g, self.N, self.tau, self.sk_eps, self.sk_iters = g, g * g, tau, sk_eps, sk_iters
        self.proj = nn.Linear(z_dim, proj_dim)
        self.pred = nn.Sequential(nn.Linear(proj_dim, 64), nn.ReLU(), nn.Linear(64, proj_dim))
        self.prototypes = nn.Parameter(F.normalize(torch.randn(self.N, proj_dim), dim=-1))

    def embed(self, z):
        return F.normalize(self.proj(z), dim=-1)

    def protos(self):
        return F.normalize(self.prototypes, dim=-1)

    @torch.no_grad()
    def sinkhorn(self, scores):
        Q = torch.exp((scores - scores.max()) / self.sk_eps).t()      # (N,B)
        Q /= Q.sum()
        N, Bn = Q.shape
        for _ in range(self.sk_iters):
            Q /= Q.sum(1, keepdim=True); Q /= N
            Q /= Q.sum(0, keepdim=True); Q /= Bn
        return (Q * Bn).t()                                         # (B,N), rows sum to 1

    def loss(self, z, z_next, act, reward):
        P = self.protos()
        u_pred = F.normalize(self.pred(self.embed(z)), dim=-1)
        logp = F.log_softmax(u_pred @ P.t() / self.tau, -1)
        with torch.no_grad():
            target = self.sinkhorn(self.embed(z_next) @ P.t())
        l = -(target * logp).sum(-1).mean()
        return l, {"swav": l.item()}

    @torch.no_grad()
    def winners(self, z):
        cos = self.embed(z) @ self.protos().t()
        return cos.topk(2, -1).indices, F.softmax(cos / self.tau, -1)


# --------------------------------------------------------------------------------------
# Topology metrics (answers "show it quantitatively")
# --------------------------------------------------------------------------------------
@torch.no_grad()
def topo_metrics(atin, z, g):
    top2, q = atin.winners(z)
    w1, w2 = top2[:, 0], top2[:, 1]
    r1, c1, r2, c2 = w1 // g, w1 % g, w2 // g, w2 % g
    neighbour = (torch.maximum((r1 - r2).abs(), (c1 - c2).abs()) <= 1).float()
    usage = torch.bincount(w1, minlength=g * g).float()
    pu = usage / usage.sum()
    perp = torch.exp(-(pu[pu > 0] * pu[pu > 0].log()).sum()).item()
    return {
        "topographic_error": 1 - neighbour.mean().item(),   # top-2 winners not grid neighbours
        "prototypes_used_frac": (usage > 0).float().mean().item(),
        "usage_perplexity": perp,                           # effective number of prototypes used
        "num_prototypes": g * g,
        "mean_max_assignment": q.max(-1).values.mean().item(),
    }


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def main():
    env, eval_env = gym.make(args.env), gym.make(args.env)
    obs_dim, nA = env.observation_space.shape[0], env.action_space.n
    net = QAgent(obs_dim, nA, args.z_dim, args.dueling).to(dev)
    tgt = copy.deepcopy(net).requires_grad_(False)          # target = encoder + Q-head copy
    atin = None
    if args.agent == "atin_v1":
        atin = AtinV1(args.z_dim, args.grid, beta_start=1.0 if "Lunar" in args.env else 5.0).to(dev)
    elif args.agent == "atin_v2":
        atin = AtinV2(args.z_dim, args.grid, nA, args.proj_dim, args.tau).to(dev)
    elif args.agent == "proto":
        atin = ProtoRL(args.z_dim, args.grid, args.proj_dim, args.tau).to(dev)
    recent = np.zeros((args.knn_queue, obs_dim), np.float32); n_recent = 0
    params = list(net.parameters()) + (list(atin.parameters()) if atin else [])
    opt = torch.optim.Adam(params, lr=args.lr)              # ONE optimizer, each param stepped once

    B = args.buffer
    S = np.zeros((B, obs_dim), np.float32); S2 = np.zeros_like(S)
    A = np.zeros(B, np.int64); R = np.zeros(B, np.float32); D = np.zeros(B, np.float32)
    ptr = size = 0

    counting = args.explore in ("ucb", "bonus")
    if counting and args.counter == "simhash":   # #Exploration: sign of a fixed random projection of the scaled state
        hi = np.where(np.isfinite(env.observation_space.high), np.abs(env.observation_space.high), 1.0)
        lo = np.where(np.isfinite(env.observation_space.low), np.abs(env.observation_space.low), 1.0)
        scale = np.maximum(np.maximum(hi, lo), 1e-6).astype(np.float32)
        Ahash = np.random.default_rng(1000 + args.seed).standard_normal((args.hash_bits, obs_dim)).astype(np.float32)
        pow2 = 2 ** np.arange(args.hash_bits)
        n_cells = 2 ** args.hash_bits
    else:
        n_cells = args.grid ** 2

    def cell_of(o, z):
        if args.counter == "simhash":
            return int(((Ahash @ (o / scale)) > 0) @ pow2)
        return int(atin.winners(z)[0][0, 0])

    counts = np.zeros((n_cells, nA)) if counting else None
    total = 0.0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ftr = open(args.out + "_train.csv", "w", newline=""); wtr = csv.writer(ftr)
    wtr.writerow(["step", "episode", "return", "length"])
    fev = open(args.out + "_eval.csv", "w", newline=""); wev = csv.writer(fev)
    wev.writerow(["step", "mean_return", "std_return"])

    def evaluate():
        rets = []
        for k in range(args.eval_episodes):
            o, _ = eval_env.reset(seed=10_000 + 100 * args.seed + k)
            done, ret = False, 0.0
            while not done:
                with torch.no_grad():
                    a = int(net(torch.as_tensor(o, device=dev)[None]).argmax())
                o, r, te, tr, _ = eval_env.step(a); ret += r; done = te or tr
            rets.append(ret)
        return float(np.mean(rets)), float(np.std(rets))

    obs, _ = env.reset(seed=args.seed)
    ep, ep_ret, ep_len, t0 = 0, 0.0, 0, time.time()
    for step in range(1, args.total_steps + 1):
        eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * step / args.eps_decay_steps)
        ot = torch.as_tensor(obs, device=dev)[None]
        with torch.no_grad():
            z = net.encoder(ot); qv = net.q(z)[0].cpu().numpy()
            win = cell_of(obs, z) if counts is not None else None
        if random.random() < eps:
            a = random.randrange(nA)
        elif args.explore == "ucb":
            bonus = args.ucb_c * np.sqrt(math.log(total + 1) / (counts[win] + 1))   # >0 for unseen
            a = int(np.argmax(qv + bonus))
        else:
            a = int(np.argmax(qv))
        if counts is not None:
            counts *= args.ucb_decay; total = total * args.ucb_decay + 1
            counts[win, a] += 1

        o2, r, te, tr, _ = env.step(a)
        r_store = r
        if args.explore == "bonus":            # intrinsic reward from the visit count of the current cell
            r_store = r + args.bonus_beta / math.sqrt(counts[win].sum())
        S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = obs, a, r_store, o2, float(te)  # bootstrap through time-limit
        recent[n_recent % args.knn_queue] = o2; n_recent += 1
        ptr = (ptr + 1) % B; size = min(size + 1, B)
        obs = o2; ep_ret += r; ep_len += 1
        if te or tr:
            ep += 1; wtr.writerow([step, ep, round(ep_ret, 3), ep_len]); ftr.flush()
            obs, _ = env.reset(); ep_ret, ep_len = 0.0, 0

        if step >= args.learning_starts and step % args.train_every == 0:
            idx = np.random.randint(0, size, args.batch_size)
            s = torch.as_tensor(S[idx], device=dev); s2 = torch.as_tensor(S2[idx], device=dev)
            a_b = torch.as_tensor(A[idx], device=dev); r_b = torch.as_tensor(R[idx], device=dev)
            d_b = torch.as_tensor(D[idx], device=dev)
            if args.explore == "knn" and n_recent > args.knn_k:   # Proto-RL-style novelty: k-th NN distance of s' among recent states,
                with torch.no_grad():                             # computed at update time with the current embedding
                    qs = torch.as_tensor(recent[:min(n_recent, args.knn_queue)], device=dev)
                    e2, eq = atin.embed(net.encoder(s2)), atin.embed(net.encoder(qs))
                    dk = torch.cdist(e2, eq).topk(args.knn_k + 1, largest=False).values[:, -1]
                    r_b = r_b + args.bonus_beta * dk
            z = net.encoder(s)
            qsa = net.q(z).gather(1, a_b[:, None]).squeeze(1)
            with torch.no_grad():
                if args.double:
                    a2 = net(s2).argmax(1, keepdim=True)
                    y = r_b + args.gamma * (1 - d_b) * tgt(s2).gather(1, a2).squeeze(1)
                else:
                    y = r_b + args.gamma * (1 - d_b) * tgt(s2).max(1).values
            loss = F.smooth_l1_loss(qsa, y)
            if atin is not None:
                z2 = net.encoder(s2)
                la, _ = atin.loss(z, z2, a_b, r_b)
                loss = loss + args.aux_weight * la
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 10.0); opt.step()

        if step % args.target_update == 0:
            tgt.load_state_dict(net.state_dict())
        if step % args.eval_every == 0:
            m, sd = evaluate(); wev.writerow([step, round(m, 3), round(sd, 3)]); fev.flush()
            print(f"[{args.agent}/{args.explore} s{args.seed}] step {step} eval {m:.1f} "
                  f"eps {eps:.3f} {(time.time()-t0)/60:.1f} min", flush=True)

    if atin is not None:
        idx = np.random.randint(0, size, min(size, 5000))
        with torch.no_grad():
            z = net.encoder(torch.as_tensor(S[idx], device=dev))
        json.dump(topo_metrics(atin, z, args.grid), open(args.out + "_topo.json", "w"), indent=1)
    json.dump(vars(args), open(args.out + "_args.json", "w"), indent=1)
    torch.save({"net": net.state_dict(), "atin": atin.state_dict() if atin else None}, args.out + "_model.pt")
    ftr.close(); fev.close()


if __name__ == "__main__":
    main()
