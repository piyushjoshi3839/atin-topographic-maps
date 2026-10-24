# ATIN: topographic state maps for inspecting deep RL controllers

Code and per-seed results for the paper

> **Topographic state maps for inspecting deep reinforcement learning controllers**
> Atin Chowdhury, Piyush Joshi. *Submitted to Neurocomputing.*

ATIN adds a small head to the encoder of a deep reinforcement learning controller (DQN or SAC). The head
learns a **self-organising map**: every state the controller sees is assigned to a cell of a 2-D grid, and
neighbouring cells hold similar states. Colouring the grid by a state variable or by an outcome (for example,
crashes) gives a readable picture of how the controller organises its operating conditions and where it fails.

**What the paper finds**
- Intuitive objectives for this problem collapse (proofs in Sec. 3.5); a balanced-assignment loss plus a SOM
  neighbourhood loss gives ordered, fully used maps.
- The map costs nothing in control performance on LunarLander, MountainCar, Acrobot and Pendulum (5–10 seeds).
- Crash risk forms contiguous hotspots on the map.
- The map does **not** speed up learning, help exploration, or detect changed dynamics better than standard methods.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.13 on CPU. `gymnasium[box2d]` needs `swig` on some systems (`apt install swig` or `conda install swig`).

## Quick start
```bash
# DQN + ATIN on LunarLander (about 17 min on one CPU core)
python atin.py --env LunarLander-v3 --agent atin_v2 --seed 1 --total-steps 200000 --out runs/demo/atin_s1
# the same controller without ATIN
python atin.py --env LunarLander-v3 --agent dqn --seed 1 --total-steps 200000 --out runs/demo/dqn_s1
# SAC + ATIN on Pendulum (about 8 min)
python sac_atin.py --env Pendulum-v1 --agent atin --seed 1 --total-steps 30000 --out runs/demo/sac_atin_s1
```
Each run writes `_train.csv`, `_eval.csv`, `_topo.json` (map metrics), `_args.json` and `_model.pt`.

## Files
| File | Purpose |
|---|---|
| `atin.py` | DQN agents: `dqn`, `atin_v2` (ATIN), `atin_v1` (original objective), `proto` (Proto-RL-style); exploration modes `egreedy`, `ucb`, `bonus`, `knn`. Run `python atin.py -h` for all options. |
| `sac_atin.py` | SAC and SAC+ATIN for continuous actions (Pendulum) |
| `diagnose.py` | Failure diagnosis (crash-risk map) and change detection for a trained LunarLander ATIN model |
| `analyze.py` | Learning curves and summary for any results folder |
| `reproduce_tables.py` | Recomputes every number in the paper's tables from `results/` |
| `scripts/reproduce_all.sh` | Re-runs every experiment in the paper (restartable) |
| `scripts/figures/` | Scripts that draw the paper's figures |
| `results/` | Per-seed logs of every run reported in the paper (CSV/JSON, no model weights) |

## Reproducing the paper
**Check the numbers from the included logs (seconds):**
```bash
python reproduce_tables.py results
```
**Re-run everything from scratch:**
```bash
JOBS=16 bash scripts/reproduce_all.sh      # writes to runs/, then prints the tables
```
The full set is roughly 35–40 single-core CPU hours of small-MLP training (measured on a 2-core cloud machine); with 16 parallel jobs it takes a few hours.
Runs are deterministic per seed on CPU with one thread (`--threads 1`, the default), so the same seed gives the
same learning curve on the same software versions.

**Figures**
```bash
python scripts/figures/fig_curves.py fig_curves_lunar.pdf "DQN=results/main/LunarLander-v3/dqn_egreedy" \
       "ATIN=results/main/LunarLander-v3/atin_v2_egreedy" "DDQN (untuned)=results/ddqn/LunarLander-v3/ddqn_dueling"
python scripts/figures/fig_map.py LunarLander-v3 runs/main/LunarLander-v3/atin_v2_egreedy_s1_model.pt fig_map_ll.pdf
python scripts/figures/fig_map.py MountainCar-v0 runs/main/MountainCar-v0/atin_v2_egreedy_s1_model.pt fig_map_mc.pdf
python scripts/figures/fig_map_pendulum.py runs/pendulum/atin_s1_model.pt fig_map_pend.pdf
python scripts/figures/fig_risk_map.py runs/diagnosis/atin_s5_cells.npz fig_risk_map.pdf
```
The map figures need trained model files, which are not stored in `results/`; run the corresponding experiment first.

## Results folder layout
| Folder | Paper |
|---|---|
| `main/` | Table 2–3, Fig. 2: DQN vs ATIN, LunarLander and MountainCar, seeds 1–10 |
| `ddqn/`, `proto_control/` | Table 2: DDQN and Proto-RL-style baselines, LunarLander |
| `acrobot/`, `acrobot_lr6.3e-4/` | Table 2 and Sec. 5.1: Acrobot |
| `pendulum/` | Tables 2–4: SAC vs SAC+ATIN, Pendulum |
| `v1_vs_v2_30k/`, `v1_200k/` | Table 3: original vs corrected objective |
| `ablation_200k/`, `ablation_50k/`, `two_term/` | Tables 4–5: ablations |
| `explore_count/`, `proto_explore/`, `explore_ucb_action/`, `explore_tuning*/` | Table 6: exploration (tuning on seed 0 only) |
| `diagnosis/` | Sec. 5.5, Table 7: failure diagnosis and change detection |

## Citation
```bibtex
@misc{chowdhury2026atin,
  title  = {Topographic state maps for inspecting deep reinforcement learning controllers},
  author = {Chowdhury, Atin and Joshi, Piyush},
  year   = {2026},
  note   = {Submitted to Neurocomputing}
}
```

## License
MIT (see `LICENSE`).
