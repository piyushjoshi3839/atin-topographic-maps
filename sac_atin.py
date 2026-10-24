#!/usr/bin/env python3
"""
sac_atin.py - continuous-action control: SAC with and without the ATIN topographic head.

The critic encoder (obs -> z) is shared by the two Q-heads; the ATIN head (same corrected
four-term objective as atin.py) is attached to this encoder, exactly as in the DQN version.
The actor has its own MLP on the raw state, so the policy network is identical for both agents.
Forward model is conditioned on the continuous action.

Example:
  python sac_atin.py --env Pendulum-v1 --agent atin --seed 1 --total-steps 30000 --out runs_sac/Pendulum-v1/atin_s1
Outputs: <out>_train.csv, <out>_eval.csv, <out>_topo.json (ATIN), <out>_args.json
"""
import argparse, copy, csv, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import gymnasium as gym

p = argparse.ArgumentParser()
p.add_argument("--env", default="Pendulum-v1")
p.add_argument("--agent", default="atin", choices=["sac", "atin"])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--total-steps", type=int, default=30_000)
p.add_argument("--out", default="runs_sac/run")
p.add_argument("--threads", type=int, default=1)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--polyak", type=float, default=0.005)
p.add_argument("--buffer", type=int, default=100_000)
p.add_argument("--learning-starts", type=int, default=1_000)
p.add_argument("--eval-every", type=int, default=2_000)
p.add_argument("--eval-episodes", type=int, default=5)
p.add_argument("--z-dim", type=int, default=16)
p.add_argument("--grid", type=int, default=10)
p.add_argument("--proj-dim", type=int, default=16)
p.add_argument("--tau", type=float, default=0.1)
p.add_argument("--sigma-end", type=float, default=1.0)
p.add_argument("--sigma-anneal-steps", type=int, default=None, help="default: half the training budget")
p.add_argument("--aux-weight", type=float, default=0.5)
p.add_argument("--w-som", type=float, default=3.0)
p.add_argument("--w-assign", type=float, default=1.0)
p.add_argument("--w-temp", type=float, default=1.0)
p.add_argument("--w-fwd", type=float, default=1.0)
args = p.parse_args()
if args.sigma_anneal_steps is None:
    args.sigma_anneal_steps = args.total_steps // 2

torch.set_num_threads(args.threads)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)


def mlp(sizes):
    L = []
    for a, b in zip(sizes[:-1], sizes[1:]): L += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*L[:-1])


class Critic(nn.Module):
    def __init__(self, ds, da, zd):
        super().__init__()
        self.encoder = mlp([ds, 128, 64, zd])
        self.q1, self.q2 = mlp([zd + da, 256, 256, 1]), mlp([zd + da, 256, 256, 1])

    def forward(self, s, a, z=None):
        z = self.encoder(s) if z is None else z
        x = torch.cat([z, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, ds, da, scale):
        super().__init__()
        self.net = mlp([ds, 256, 256, 2 * da]); self.register_buffer("scale", scale)

    def forward(self, s, deterministic=False):
        mu, log_std = self.net(s).chunk(2, -1)
        log_std = log_std.clamp(-5, 2); std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a * self.scale, logp


class AtinHead(nn.Module):
    """Same corrected objective as atin.py (AtinV2), forward model conditioned on a continuous action."""
    def __init__(self, zd, da, g, pd, tau):
        super().__init__()
        self.g, self.N, self.tau = g, g * g, tau
        self.proj = nn.Linear(zd, pd)
        self.prototypes = nn.Parameter(F.normalize(torch.randn(self.N, pd), dim=-1))
        ii, jj = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
        c = torch.stack([ii.flatten(), jj.flatten()], 1).float()
        self.register_buffer("coords", c); self.register_buffer("grid_d2", ((c[:, None] - c[None]) ** 2).sum(-1))
        self.fwd = mlp([zd + da, 64, pd])
        self.sigma0 = g / 2.0; self.sigma = self.sigma0; self.step = 0

    def embed(self, z): return F.normalize(self.proj(z), dim=-1)
    def protos(self): return F.normalize(self.prototypes, dim=-1)

    def loss(self, z, z2, a):
        u, P = self.embed(z), self.protos()
        cos = u @ P.t(); q = F.softmax(cos / self.tau, -1)
        qbar = q.mean(0)
        l_assign = -(q * torch.log(q + 1e-8)).sum(-1).mean() + (qbar * torch.log(qbar + 1e-8)).sum()
        K = torch.exp(-self.grid_d2[cos.argmax(-1)] / (2 * self.sigma ** 2)); K = K / K.sum(-1, keepdim=True)
        l_som = (K * (1 - cos)).sum(-1).mean()
        u2 = self.embed(z2)
        q2 = F.softmax(u2 @ P.t() / self.tau, -1)
        l_temp = ((q @ self.coords - q2 @ self.coords) ** 2).sum(-1).mean() / self.g ** 2
        pred = F.normalize(self.fwd(torch.cat([z, a], -1)), dim=-1)
        l_fwd = (1 - (pred * u2.detach()).sum(-1)).mean()
        self.step += 1
        fr = min(1.0, self.step / args.sigma_anneal_steps)
        self.sigma = self.sigma0 * (args.sigma_end / self.sigma0) ** fr
        return args.w_assign * l_assign + args.w_som * l_som + args.w_temp * l_temp + args.w_fwd * l_fwd

    @torch.no_grad()
    def metrics(self, z):
        cos = self.embed(z) @ self.protos().t(); top2 = cos.topk(2, -1).indices; g = self.g
        r1, c1, r2, c2 = top2[:, 0] // g, top2[:, 0] % g, top2[:, 1] // g, top2[:, 1] % g
        nb = (torch.maximum((r1 - r2).abs(), (c1 - c2).abs()) <= 1).float()
        use = torch.bincount(top2[:, 0], minlength=self.N).float(); pu = use / use.sum()
        return {"topographic_error": 1 - nb.mean().item(), "prototypes_used_frac": (use > 0).float().mean().item(),
                "usage_perplexity": torch.exp(-(pu[pu > 0] * pu[pu > 0].log()).sum()).item(), "num_prototypes": self.N}


def main():
    env, eval_env = gym.make(args.env), gym.make(args.env)
    ds, da = env.observation_space.shape[0], env.action_space.shape[0]
    scale = torch.as_tensor(env.action_space.high, dtype=torch.float32)
    critic = Critic(ds, da, args.z_dim); critic_t = copy.deepcopy(critic).requires_grad_(False)
    actor = Actor(ds, da, scale)
    head = AtinHead(args.z_dim, da, args.grid, args.proj_dim, args.tau) if args.agent == "atin" else None
    log_alpha = torch.zeros(1, requires_grad=True); target_ent = -float(da)
    c_params = list(critic.parameters()) + (list(head.parameters()) if head else [])
    opt_c = torch.optim.Adam(c_params, lr=args.lr)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_al = torch.optim.Adam([log_alpha], lr=args.lr)

    B = args.buffer
    S = np.zeros((B, ds), np.float32); S2 = np.zeros_like(S); A = np.zeros((B, da), np.float32)
    R = np.zeros(B, np.float32); D = np.zeros(B, np.float32); ptr = size = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ftr = open(args.out + "_train.csv", "w", newline=""); wtr = csv.writer(ftr); wtr.writerow(["step", "episode", "return", "length"])
    fev = open(args.out + "_eval.csv", "w", newline=""); wev = csv.writer(fev); wev.writerow(["step", "mean_return", "std_return"])

    def evaluate():
        rets = []
        for k in range(args.eval_episodes):
            o, _ = eval_env.reset(seed=10_000 + 100 * args.seed + k); done, ret = False, 0.0
            while not done:
                with torch.no_grad(): a, _ = actor(torch.as_tensor(o)[None], deterministic=True)
                o, r, te, tr, _ = eval_env.step(a[0].numpy()); ret += r; done = te or tr
            rets.append(ret)
        return float(np.mean(rets)), float(np.std(rets))

    o, _ = env.reset(seed=args.seed); ep, ep_ret, ep_len, t0 = 0, 0.0, 0, time.time()
    for step in range(1, args.total_steps + 1):
        if step < args.learning_starts:
            a = env.action_space.sample()
        else:
            with torch.no_grad(): a = actor(torch.as_tensor(o, dtype=torch.float32)[None])[0][0].numpy()
        o2, r, te, tr, _ = env.step(a)
        S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = o, a, r, o2, float(te)
        ptr = (ptr + 1) % B; size = min(size + 1, B); o = o2; ep_ret += r; ep_len += 1
        if te or tr:
            ep += 1; wtr.writerow([step, ep, round(float(ep_ret), 3), ep_len]); ftr.flush()
            o, _ = env.reset(); ep_ret, ep_len = 0.0, 0

        if step >= args.learning_starts:
            idx = np.random.randint(0, size, args.batch_size)
            s, a_b, r_b, s2, d_b = (torch.as_tensor(x[idx]) for x in (S, A, R, S2, D))
            alpha = log_alpha.exp().detach()
            with torch.no_grad():
                a2, lp2 = actor(s2); q1t, q2t = critic_t(s2, a2)
                y = r_b + args.gamma * (1 - d_b) * (torch.min(q1t, q2t) - alpha * lp2)
            z = critic.encoder(s); q1, q2 = critic(s, a_b, z)
            lc = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            if head is not None:
                lc = lc + args.aux_weight * head.loss(z, critic.encoder(s2), a_b / scale)
            opt_c.zero_grad(); lc.backward(); opt_c.step()
            an, lp = actor(s)
            with torch.no_grad(): zc = critic.encoder(s)
            q1n, q2n = critic(s, an, zc)
            la = (alpha * lp - torch.min(q1n, q2n)).mean()
            opt_a.zero_grad(); la.backward(); opt_a.step()
            lal = -(log_alpha * (lp.detach() + target_ent)).mean()
            opt_al.zero_grad(); lal.backward(); opt_al.step()
            with torch.no_grad():
                for pt, pp in zip(critic_t.parameters(), critic.parameters()): pt.mul_(1 - args.polyak).add_(args.polyak * pp)
        if step % args.eval_every == 0:
            m, sd = evaluate(); wev.writerow([step, round(m, 3), round(sd, 3)]); fev.flush()
            print(f"[{args.agent} s{args.seed}] step {step} eval {m:.1f} {(time.time()-t0)/60:.1f} min", flush=True)

    if head is not None:
        idx = np.random.randint(0, size, min(size, 5000))
        with torch.no_grad(): z = critic.encoder(torch.as_tensor(S[idx]))
        json.dump(head.metrics(z), open(args.out + "_topo.json", "w"), indent=1)
    json.dump(vars(args), open(args.out + "_args.json", "w"), indent=1)
    torch.save({"critic": critic.state_dict(), "actor": actor.state_dict(), "head": head.state_dict() if head else None}, args.out + "_model.pt")


if __name__ == "__main__":
    main()
