import torch
import torch.nn as nn


# ============================================================
# Core modules for reinforcement learning-based ligand optimization:
#   - Critic: Quantile value function estimator
#   - MolecularRLAgent: Continuous-action agent in CJT-VAE latent space
#   - quantile_huber_loss: Quantile regression Huber loss
#   - MultiStepPPOTrainer: PPO trainer with multi-step trajectory updates
# ============================================================


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: [T, B] or [T], rewards at each timestep.
        values: [T+1, B] or [T+1], state values including the final state.
        dones: [T, B] or [T], terminal flags.
        gamma: Discount factor.
        lam: GAE lambda parameter.

    Returns:
        advantages: [T, B] or [T], advantage estimates.
        returns: [T, B] or [T], discounted returns.
    """
    rewards = torch.as_tensor(rewards)
    values = torch.as_tensor(values)
    dones = torch.as_tensor(dones)

    # Ensure correct dimensions
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)
        values = values.unsqueeze(1)
        dones = dones.unsqueeze(1)

    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    last_gae = 0

    # Compute backwards through time
    for t in reversed(range(T)):
        if t == T - 1:
            # Final step: use the terminal state value
            next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
        else:
            next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        # Temporal difference residual
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]

        # GAE accumulation
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

        # Discounted return
        returns[t] = advantages[t] + values[t]

    return advantages, returns


class Critic(nn.Module):
    """
    Quantile distributional critic for risk-sensitive value estimation.

    Used within MolecularRLAgent to estimate a distribution over state
    values rather than a single point estimate. Input is expected to be
    a feature tensor of shape [batch_size, hidden_dim * 2] from the
    encoder projection layer.
    """

    def __init__(self, hidden_dim, num_quantiles=16):
        """
        Args:
            hidden_dim: Hidden dimension matching the agent's encoder output.
                        The actual input dimension is 2 * hidden_dim.
            num_quantiles: Number of quantiles to output.
        """
        super().__init__()

        if hidden_dim is None:
            raise ValueError(
                "Critic.hidden_dim must not be None. "
                "Please pass hidden_dim explicitly when constructing Critic."
            )

        # Lightweight self-attention over the feature dimension
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )

        # Value network: 2 * hidden_dim → num_quantiles
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(256, num_quantiles),
        )

        # Initialize weights
        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, h):
        """
        Forward pass for the quantile critic.

        Args:
            h: Feature tensor of shape [batch_size, hidden_dim * 2] or
               [batch_size, seq_len, hidden_dim * 2].

        Returns:
            Quantile values of shape [batch_size, num_quantiles].
        """
        # Ensure h is 3D: [batch_size, seq_len, hidden_dim * 2]
        if h.dim() == 2:
            h = h.unsqueeze(1)

        batch_size = h.size(0)
        hidden_dim = h.size(2) // 2

        # Self-attention over the feature dimension
        attn_output, _ = self.attention(h, h, h)
        attn_output = attn_output.squeeze(1)

        # Value prediction
        quantiles = self.value_net(attn_output)
        quantiles = torch.clamp(quantiles, -300.0, 300.0)

        return quantiles


class MolecularRLAgent(nn.Module):
    """
    PPO agent operating on the CJT-VAE latent space with continuous actions.

    The agent encodes a latent vector z into a shared hidden representation,
    then uses separate Actor and Critic heads for policy and value estimation.
    The Actor outputs a Gaussian policy over latent displacements, and the
    Critic provides quantile-based distributional value estimates.
    """

    def __init__(
            self,
            z_dim=48,
            tree_dim=24,
            graph_dim=24,
            hidden_dim=128,
            action_dim=48,
            num_quantiles=8
    ):
        super().__init__()

        self.z_dim = z_dim
        # These fields originate from an earlier structured latent design
        # and are currently retained for interface compatibility only.
        self.tree_dim = tree_dim
        self.graph_dim = graph_dim

        # Shared encoder for feature extraction
        self.encoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Lightweight self-attention for the Actor head
        self.actor_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        # Actor mean network: outputs action mean in [-1, 1]
        self.actor_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # Actor log-standard deviation (learnable parameter)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Quantile distributional critic
        self.critic = Critic(
            hidden_dim=hidden_dim,
            num_quantiles=num_quantiles
        )

        # Projection layer for critic input (hidden_dim → 2 * hidden_dim)
        self.critic_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        # Quantile target positions
        self.register_buffer(
            "taus",
            torch.linspace(0.5 / num_quantiles, 1 - 0.5 / num_quantiles, num_quantiles)
        )

    # ---------- Actor ----------
    def act(self, z):
        """
        Sample an action from the current policy.

        Args:
            z: Latent vector of shape [B, z_dim].

        Returns:
            action: Sampled displacement of shape [B, action_dim].
            log_prob: Log probability of the sampled action [B].
        """
        h = self.encoder(z)

        # Self-attention: reshape to [B, 1, H], apply, then flatten back
        h_seq = h.unsqueeze(1)
        attn_out, _ = self.actor_attn(h_seq, h_seq, h_seq)
        h_attn = attn_out.squeeze(1)

        mu = self.actor_mu(h_attn)

        # Fixed standard deviation
        log_std = self.actor_log_std
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        action = action.clamp(-0.8, 0.8)
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob

    # ---------- Critic ----------
    def evaluate(self, z):
        """
        Estimate quantile values for a given latent state.

        Args:
            z: Latent vector of shape [B, z_dim].

        Returns:
            Quantile values of shape [B, num_quantiles].
        """
        h = self.encoder(z)

        # Project to critic input dimension
        critic_input = self.critic_proj(h)

        quantiles = self.critic(critic_input)
        return quantiles

    # ---------- Value ----------
    def value(self, z, mode="mean", alpha=0.25):
        """
        Compute a scalar value estimate from quantile predictions.

        Args:
            z: Latent vector of shape [B, z_dim].
            mode: Aggregation mode — "mean" or "cvar" (Conditional Value at Risk).
            alpha: Fraction of lower quantiles for CVaR.

        Returns:
            Scalar value estimate of shape [B].
        """
        quantiles = self.evaluate(z)

        if mode == "mean":
            return quantiles.mean(dim=-1)
        elif mode == "cvar":
            k = max(1, int(self.taus.numel() * alpha))
            return quantiles[:, :k].mean(dim=-1)
        else:
            raise ValueError(f"Unknown value mode: {mode}")


def quantile_huber_loss(pred, target, taus, kappa=1.0):
    """
    Quantile regression Huber loss.

    Args:
        pred: Predicted quantiles of shape [B, N].
        target: Target values of shape [B].
        taus: Quantile target positions of shape [N].
        kappa: Huber loss threshold.

    Returns:
        Scalar loss value.
    """
    diff = target.unsqueeze(1) - pred.unsqueeze(2)
    abs_diff = diff.abs()

    huber = torch.where(
        abs_diff <= kappa,
        0.5 * diff.pow(2),
        kappa * (abs_diff - 0.5 * kappa)
    )

    taus = taus.view(1, -1, 1)
    loss = torch.abs(taus - (diff.detach() < 0).float()) * huber

    return loss.mean()


class MultiStepPPOTrainer:
    """
    Proximal Policy Optimization trainer with multi-step trajectory support.

    Collects trajectories of length rollout_steps, computes GAE advantages,
    and performs multiple epochs of PPO updates with clipping, entropy
    regularization, and optional early stopping based on KL divergence.
    """

    def __init__(self, agent, lr=1e-4, clip_eps=0.1, vf_coef=0.5,
                 ent_coef=0.01, max_grad_norm=0.3, train_iters=4,
                 target_kl=0.01, value_mode="mean", cvar_alpha=0.25,
                 device="cpu", warmup_steps=1000,
                 gamma=0.99, gae_lambda=0.95, rollout_steps=2):
        """
        Args:
            gamma: Discount factor (default 0.99).
            gae_lambda: GAE lambda parameter (default 0.95).
            rollout_steps: Number of steps per trajectory (default 2).
        """
        self.agent = agent
        self.device = device

        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.train_iters = train_iters
        self.target_kl = target_kl
        self.value_mode = value_mode
        self.cvar_alpha = cvar_alpha

        # Multi-step trajectory parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.rollout_steps = rollout_steps

        self.warmup_steps = warmup_steps
        self.update_step = 0

        self.optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

    def update_trajectory(self, states, actions, old_logps, rewards, dones=None):
        """
        Perform a multi-step PPO update from a collected trajectory.

        Args:
            states: Latent states of shape [B, T+1, D] or list of length T+1.
            actions: Actions of shape [B, T, D] or list of length T.
            old_logps: Old log-probabilities of shape [B, T] or list of length T.
            rewards: Rewards of shape [B, T] or list of length T.
            dones: Terminal flags of shape [B, T] or list of length T (optional).

        Returns:
            Dictionary of training metrics averaged over update iterations.
        """
        # Step 1: Convert inputs to tensors
        if isinstance(states, list):
            states = torch.stack(states, dim=1)        # [B, T+1, D]
        if isinstance(actions, list):
            actions = torch.stack(actions, dim=1)      # [B, T, D]
        if isinstance(old_logps, list):
            old_logps = torch.stack(old_logps, dim=1)  # [B, T]
        if isinstance(rewards, list):
            rewards = torch.stack(rewards, dim=1)      # [B, T]

        batch_size, T, action_dim = actions.shape
        state_dim = states.shape[-1]

        # Step 2: Default dones to all False if not provided
        if dones is None:
            dones = torch.zeros(batch_size, T, device=self.device)
        elif isinstance(dones, list):
            dones = torch.stack(dones, dim=1)

        # Step 3: Compute state values (no gradient required)
        with torch.no_grad():
            values = []
            for t in range(T + 1):
                state_t = states[:, t, :]
                value_t = self.agent.value(state_t, mode=self.value_mode, alpha=self.cvar_alpha)
                value_t = torch.clamp(value_t, -300.0, 300.0)
                values.append(value_t)
            values = torch.stack(values, dim=1)  # [B, T+1]

        # Step 4: Compute GAE advantages
        # Transpose to [T, B] for the GAE function
        rewards_t = rewards.transpose(0, 1)      # [T, B]
        values_t = values.transpose(0, 1)        # [T+1, B]
        dones_t = dones.transpose(0, 1)          # [T, B]

        advantages_t, returns_t = compute_gae(
            rewards_t,
            values_t,
            dones_t,
            gamma=self.gamma,
            lam=self.gae_lambda
        )

        # Transpose back to [B, T]
        advantages = advantages_t.transpose(0, 1)
        returns = returns_t.transpose(0, 1)

        # Normalize advantages and returns
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Step 5: Flatten all tensors to [B*T, ...]
        states_flat = states[:, :-1, :].reshape(-1, state_dim).detach()
        actions_flat = actions.reshape(-1, action_dim).detach()
        old_logps_flat = old_logps.reshape(-1).detach()
        advantages_flat = advantages.reshape(-1).detach()
        returns_flat = returns.reshape(-1).detach()

        # Step 6: Initialize metrics dictionary
        metrics = {
            "policy_loss": [], "critic_loss": [], "total_loss": [],
            "entropy": [], "kl_divergence": [], "gradient_norm": [],
            "ratio_mean": [], "ratio_std": [],
            "value_mean": values.mean().item(),
            "value_std": values.std().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "mu_std": [], "action_std": [],
            "old_logp_mean": old_logps.mean().item(),
            "new_logp_mean": []
        }

        # Step 7: Multiple PPO update iterations
        for iter_idx in range(self.train_iters):
            # Forward pass (recomputed each iteration)
            h = self.agent.encoder(states_flat)
            h_seq = h.unsqueeze(1)
            attn_out, _ = self.agent.actor_attn(h_seq, h_seq, h_seq)
            h_attn = attn_out.squeeze(1)
            mu = self.agent.actor_mu(h_attn)
            metrics["mu_std"].append(mu.std().item())

            # Standard deviation
            log_std = self.agent.actor_log_std
            std = torch.exp(log_std)
            metrics["action_std"].append(std.mean().item())

            # Create action distribution
            dist = torch.distributions.Normal(mu, std)

            # New log-probabilities
            new_logp = dist.log_prob(actions_flat).sum(-1)
            metrics["new_logp_mean"].append(new_logp.mean().item())

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Importance sampling ratio
            log_ratio = new_logp - old_logps_flat
            ratio = torch.exp(log_ratio)
            ratio_mean = ratio.mean().item()
            ratio_std = ratio.std().item()

            # Clipped policy loss
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # Quantile critic loss
            quantiles = self.agent.evaluate(states_flat)
            target = returns_flat.unsqueeze(1).expand_as(quantiles)
            critic_loss = quantile_huber_loss(quantiles, target, self.agent.taus)

            # Total loss: policy + value - entropy
            loss = policy_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
            print(f'critic_loss: {critic_loss.item()} | policy_loss: {policy_loss.item()} | loss: {loss.item()}')

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.max_grad_norm
            )

            # Skip update if gradients are unstable
            if torch.isnan(grad_norm) or grad_norm > 100:
                self.optimizer.zero_grad()
                break

            self.optimizer.step()

            # KL divergence check
            approx_kl = (old_logps_flat - new_logp).mean().item()

            # Record metrics for this iteration
            metrics["policy_loss"].append(policy_loss.item())
            metrics["critic_loss"].append(critic_loss.item())
            metrics["total_loss"].append(loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["kl_divergence"].append(approx_kl)
            metrics["gradient_norm"].append(grad_norm.item())
            metrics["ratio_mean"].append(ratio_mean)
            metrics["ratio_std"].append(ratio_std)

        self.update_step += 1

        # Step 8: Aggregate metrics across iterations
        out = {}
        for k in metrics:
            if isinstance(metrics[k], list) and metrics[k]:
                out[k] = sum(metrics[k]) / len(metrics[k])
            else:
                out[k] = metrics[k]

        out["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        out["actual_iters"] = len(metrics["policy_loss"])
        out["update_step"] = self.update_step
        out["trajectory_length"] = T
        out["gamma"] = self.gamma
        out["gae_lambda"] = self.gae_lambda

        return out
