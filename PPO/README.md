# PPO: Proximal Policy Optimization for Latent-Space Ligand Optimization

This module implements a reinforcement learning framework that optimizes molecular structures directly in the continuous latent space of a pre-trained JT-VAE. The agent learns to navigate the latent space toward regions that decode to high-selectivity ligands for Am(III)/Eu(III) separation.

## Core Files

| File | Description |
|:------|:------------|
| `env.py` | RL environment wrapping JT-VAE decoder and XGBoost surrogate model for reward computation |
| `agent.py` | PPO agent with quantile distributional critic and multi-step trajectory support |
| `RL_train.py` | Main training script with elite replay buffer, repetition penalty, and baseline comparison |

## Architecture

```
Latent Vector z ──→ Encoder ──→ Actor ──→ Action Δz ──→ z' = z + Δz
                     │                        │
                     └──→ Critic (Quantile)   └──→ JT-VAE Decoder ──→ SMILES
                           │                                              │
                           └── Value Estimate                             └──→ XGBoost ──→ SF Prediction
                                                                                      │
                                                                              ┌───────┴───────┐
                                                                              │  Signal Reward  │
                                                                              │  Explore Bonus  │
                                                                              │  Repeat Penalty │
                                                                              └───────┬───────┘
                                                                                      │
                                                                              Composite Reward
```

## Key Components

### Environment (`env.py`)

- **`CJTVAE_XGB_Env`**: Wraps a frozen JT-VAE decoder and XGBoost surrogate model. Given a latent vector, it decodes the corresponding molecule, predicts its separation factor, and returns a composite reward.
- **Reward function** combines:
  - **Signal reward**: Soft-aggregated SF prediction over top-k similar reference molecules
  - **Exploration bonus**: Gaussian-shaped similarity reward encouraging moderate structural novelty
  - **Repetition penalty**: Logarithmic penalty based on molecule visit frequency
- **Two separate top-k queries**: Deduplicated for similarity scoring, non-deduplicated for solvent condition matching
- **Timeout-protected decoding** via signal-based alarms

### Agent (`agent.py`)

- **`MolecularRLAgent`**: Continuous-action PPO agent operating on 48-dimensional latent vectors
  - Shared encoder with LayerNorm and GELU activations
  - Actor head with self-attention and Tanh-bounded output (actions clipped to [-0.8, 0.8])
  - Learnable action log-standard deviation
- **`Critic`**: Quantile distributional value network (8 quantiles) for risk-sensitive value estimation
  - Self-attention over feature dimension
  - Supports mean and CVaR (Conditional Value at Risk) aggregation modes
- **`MultiStepPPOTrainer`**: PPO trainer with multi-step trajectory support
  - Generalized Advantage Estimation (GAE) with configurable gamma and lambda
  - Clipped policy updates with entropy regularization
  - Quantile Huber loss for critic training
  - Gradient clipping and optional early stopping via KL divergence

### Training Script (`RL_train.py`)

- **`EliteReplayBuffer`**: Stores high-reward trajectories with scaffold-based deduplication
  - Murcko scaffold extraction and Tanimoto similarity filtering
  - Reward-weighted sampling of initial latent states
  - Configurable noise for exploration around elite trajectories
- **Mixed initialization strategy**: Sigmoid-scheduled blend of random prior samples and elite replay samples
- **Similarity-aware repetition penalty**: Penalizes molecules similar to those already generated, with penalty strength scaled by Tanimoto similarity
- **Baseline integration**: Compares PPO against Random Search, Greedy Search, and Cross-Entropy Method

## Training

```bash
python RL_train.py
```

### Key Hyperparameters

| Parameter | Value | Description |
|:----------|:------|:------------|
| `LATENT_DIM` | 48 | JT-VAE latent space dimension |
| `BATCH_SIZE` | 16 | Molecules per training step |
| `ROLLOUT_STEPS` | 3 | Steps per trajectory |
| `TRAIN_ITERS` | 3 | PPO update epochs per batch |
| `GAMMA` | 0.9 | Discount factor |
| `GAE_LAMBDA` | 0.95 | GAE smoothing parameter |
| `CLIP_EPS` | 0.2 | PPO clipping range |
| `ENT_COEF` | 0.02 | Entropy regularization weight |

### Output Files

| File | Description |
|:------|:------------|
| `checkpoint_step*.pth` | Model checkpoints with agent and optimizer states |
| `training_log.xlsx` | Per-molecule records with SMILES, rewards, and SF predictions |
| `training_metrics.csv` | Per-step training metrics (losses, KL divergence, etc.) |
| `top_molecules.xlsx` | Top-50 unique molecules by reward |
| `final_training_metrics.png` | Training curves (loss, accuracy, validity) |
| `ppo_plot.png` | PPO vs. baseline comparison plot |

## Usage

```python
from env import CJTVAE_XGB_Env
from agent import MolecularRLAgent, MultiStepPPOTrainer

# Initialize environment with pre-trained models
env = CJTVAE_XGB_Env(
    cjtvae_model=cjtvae,
    xgb_model=xgb,
    device=device,
)

# Create PPO agent
agent = MolecularRLAgent(z_dim=48).to(device)

# Create trainer
ppo = MultiStepPPOTrainer(
    agent=agent,
    lr=1e-4,
    clip_eps=0.2,
    ent_coef=0.02,
    rollout_steps=3,
    device=device,
)

# Collect a trajectory and update
states, actions, logps, rewards, smiles, valid, infos = collect_rollout(
    agent, env, z0, rollout_steps=3
)
metrics = ppo.update_trajectory(states, actions, logps, rewards)
```

## Reward Function Details

The composite reward for a generated molecule is:

```
R = R_signal + R_explore - R_repeat
```

- **R_signal**: Soft aggregation of XGBoost-predicted SF values over top-k similar reference molecules, with logarithmic decay for very high SF values
- **R_explore**: Gaussian bonus centered at similarity 0.50 (σ=0.25), encouraging molecules with moderate structural distance from known ligands
- **R_repeat**: Logarithmic penalty η·log(n_smiles + 1) where η=0.1, discouraging repeated generation of the same molecules

Invalid molecules receive a fixed penalty of -0.2.
