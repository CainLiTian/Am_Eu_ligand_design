import torch
import torch.nn as nn


# ============================================================
# 本文件定义了强化学习所需的核心模块：
#   - Critic：分位数价值函数估计器
#   - MolecularRLAgent：在 CJT-VAE 潜空间上做连续动作的 Agent
#   - quantile_huber_loss：分位数回归 Huber 损失
#   - PPOTrainer：基于上述 Agent 的 PPO 训练器
# ============================================================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计(GAE)

    Args:
        rewards: [T, B] 或 [T,]，每个时间步的奖励
        values: [T+1, B] 或 [T+1,]，每个状态的价值（包括最后状态）
        dones: [T, B] 或 [T,]，是否终止
        gamma: 折扣因子
        lam: GAE参数

    Returns:
        advantages: [T, B] 或 [T,]，优势函数
        returns: [T, B] 或 [T,]，折扣回报
    """
    rewards = torch.as_tensor(rewards)
    values = torch.as_tensor(values)
    dones = torch.as_tensor(dones)

    # 确保维度正确
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)
        values = values.unsqueeze(1)
        dones = dones.unsqueeze(1)

    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # 最后一步的TD残差
    last_gae = 0

    # 反向计算
    for t in reversed(range(T)):
        if t == T - 1:
            # 最后一步：用最后的状态价值
            next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t]
        else:
            next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        # TD残差
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]

        # GAE
        advantages[t] = last_gae = delta + gamma * lam * next_non_terminal * last_gae

        # 折扣回报
        returns[t] = advantages[t] + values[t]

    return advantages, returns

class Critic(nn.Module):
    """分位数 Critic，forward 只接受一个特征张量 h。

    该类在 `MolecularRLAgent` 中被用作分位数价值函数估计器：
    - 输入维度为 `hidden_dim * 2`，通常是 encoder 输出经线性层投影后的特征；
    - 输出为 `num_quantiles` 个分位数，用于风险敏感的价值估计。
    """

    def __init__(self, hidden_dim, num_quantiles=16):
        """
        参数
        ----
        hidden_dim : int
            与 Agent 的隐藏维度一致，Critic 实际输入维度为 2 * hidden_dim。
        num_quantiles : int
            输出的分位数个数。
        """
        super().__init__()

        if hidden_dim is None:
            raise ValueError(
                "Critic.hidden_dim 不能为 None，请在构造时显式传入 hidden_dim。"
            )

        # 简化注意力机制（当前只在长度为 1 的序列上做 self-attention）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True
        )

        # 简化值网络：2 * hidden_dim → num_quantiles
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(256, num_quantiles),
        )

        # 初始化
        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, h):
        """
        只接受一个参数 h。

        参数
        ----
        h : torch.Tensor
            形状为 [batch_size, hidden_dim * 2] 或
            [batch_size, seq_len, hidden_dim * 2] 的特征张量。
            在当前实现中，通常是由 `MolecularRLAgent.encoder` 输出，
            经过 `critic_proj` 投影得到的 [batch_size, 2 * hidden_dim]。
        """
        # 确保 h 是 3D 张量 [batch_size, seq_len, hidden_dim * 2]
        if h.dim() == 2:
            h = h.unsqueeze(1)  # [batch_size, 1, hidden_dim*2]

        # 分割h为query, key, value
        batch_size = h.size(0)
        hidden_dim = h.size(2) // 2

        # 简单处理：使用h作为query和key
        attn_output, _ = self.attention(h, h, h)
        attn_output = attn_output.squeeze(1)

        # 值预测
        quantiles = self.value_net(attn_output)
        quantiles = torch.clamp(quantiles, -300.0, 300.0)

        return quantiles


class MolecularRLAgent(nn.Module):
    """在 CJT-VAE 潜空间上做连续动作 PPO 的主 Agent。

    当前实现将 z 视为一个整体向量进行编码，并未实际使用 tree/graph 分拆；
    `tree_dim` / `graph_dim` 参数仅为兼容旧实验接口而保留。
    """

    def __init__(
            self,
            z_dim=48,  # 输入维度
            tree_dim=24,  # tree部分维度
            graph_dim=24,  # graph部分维度
            hidden_dim=128,
            action_dim=48,  # 输出维度
            num_quantiles=8
    ):
        super().__init__()

        self.z_dim = z_dim
        # 以下两个字段源自早期的结构化 latent 设计，当前实现未实际使用。
        self.tree_dim = tree_dim
        self.graph_dim = graph_dim

        # 简化编码器（共享特征提取）
        self.encoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Actor 的自注意力层，在共享特征 h 上做一次轻量的 self-attention
        self.actor_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )

        # Actor网络
        self.actor_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 限制输出范围[-1, 1]
        )

        # Actor的对数标准差
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic网络
        self.critic = Critic(
            hidden_dim=hidden_dim,
            num_quantiles=num_quantiles
        )

        # Critic输入投影层
        self.critic_proj = nn.Linear(hidden_dim, hidden_dim * 2)

        # 分位数位置
        self.register_buffer(
            "taus",
            torch.linspace(0.5 / num_quantiles, 1 - 0.5 / num_quantiles, num_quantiles)
        )

    # ---------- Actor ----------
    def act(self, z):
        h = self.encoder(z)                 # [B, H]

        # 自注意力：先转成 [B, 1, H]，做一次 self-attention，再还原为 [B, H]
        h_seq = h.unsqueeze(1)              # [B, 1, H]
        attn_out, _ = self.actor_attn(h_seq, h_seq, h_seq)
        h_attn = attn_out.squeeze(1)        # [B, H]

        mu = self.actor_mu(h_attn)

        # 固定标准差
        log_std = self.actor_log_std
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        action = action.clamp(-0.8, 0.8)
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob

    # ---------- Critic ----------
    def evaluate(self, z):
        h = self.encoder(z)

        # 投影为Critic输入
        critic_input = self.critic_proj(h)

        # Critic只需要一个参数
        quantiles = self.critic(critic_input)
        return quantiles

    # ---------- Value ----------
    def value(self, z, mode="mean", alpha=0.25):
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
    pred, target: [B, N]
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


# 修改agent.py中的PPOTrainer
# agent.py - 修复PPOTrainer.update方法
class MultiStepPPOTrainer:
    def __init__(self, agent, lr=1e-4, clip_eps=0.1, vf_coef=0.5,
                 ent_coef=0.01, max_grad_norm=0.3, train_iters=4,
                 target_kl=0.01, value_mode="mean", cvar_alpha=0.25,
                 device="cpu", warmup_steps=1000,
                 gamma=0.99, gae_lambda=0.95, rollout_steps=2):
        """
        新增参数：
        gamma: 折扣因子，默认0.99
        gae_lambda: GAE lambda参数，默认0.95
        rollout_steps: rollout步数，默认2步
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

        # 新增：多步参数
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
        """多步轨迹PPO更新 - 完全修正版"""
        # ========= 1. 转换输入格式 =========
        if isinstance(states, list):
            states = torch.stack(states, dim=1)  # [B, T+1, D]
        if isinstance(actions, list):
            actions = torch.stack(actions, dim=1)  # [B, T, D]
        if isinstance(old_logps, list):
            old_logps = torch.stack(old_logps, dim=1)  # [B, T]
        if isinstance(rewards, list):
            rewards = torch.stack(rewards, dim=1)  # [B, T]

        batch_size, T, action_dim = actions.shape
        state_dim = states.shape[-1]

        # ========= 2. 如果没有提供dones，假设都没终止 =========
        if dones is None:
            dones = torch.zeros(batch_size, T, device=self.device)
        elif isinstance(dones, list):
            dones = torch.stack(dones, dim=1)

        # ========= 3. 计算状态价值（不需要梯度） =========
        with torch.no_grad():
            values = []
            for t in range(T + 1):
                state_t = states[:, t, :]
                value_t = self.agent.value(state_t, mode=self.value_mode, alpha=self.cvar_alpha)
                value_t = torch.clamp(value_t, -300.0, 300.0)
                values.append(value_t)
            values = torch.stack(values, dim=1)  # [B, T+1]

        # ========= 2. 计算GAE优势 =========
        # 先转成 [T, B]
        rewards_t = rewards.transpose(0, 1)  # [T, B]
        values_t = values.transpose(0, 1)  # [T+1, B]
        dones_t = dones.transpose(0, 1)  # [T, B]

        #rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

        advantages_t, returns_t = compute_gae(
            rewards_t,
            values_t,
            dones_t,
            gamma=self.gamma,
            lam=self.gae_lambda
        )


        # 再转回 [B, T]
        advantages = advantages_t.transpose(0, 1)
        returns = returns_t.transpose(0, 1)

        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ========= 5. 展平数据 =========
        states_flat = states[:, :-1, :].reshape(-1, state_dim).detach()  # ✅ 一开始就detach
        actions_flat = actions.reshape(-1, action_dim).detach()  # ✅ 动作也detach
        old_logps_flat = old_logps.reshape(-1).detach()  # ✅ logp也detach
        advantages_flat = advantages.reshape(-1).detach()  # ✅ 优势也detach
        returns_flat = returns.reshape(-1).detach()  # ✅ 回报也detach

        # ========= 6. 初始化指标 =========
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

        # ========= 7. 多次迭代PPO更新 =========
        for iter_idx in range(self.train_iters):
            # ----- 前向传播（每次迭代重新计算）-----
            h = self.agent.encoder(states_flat)
            h_seq = h.unsqueeze(1)
            attn_out, _ = self.agent.actor_attn(h_seq, h_seq, h_seq)
            h_attn = attn_out.squeeze(1)
            mu = self.agent.actor_mu(h_attn)
            metrics["mu_std"].append(mu.std().item())

            # 标准差处理（移除clamp！）
            log_std = self.agent.actor_log_std
            std = torch.exp(log_std)
            metrics["action_std"].append(std.mean().item())

            # 创建分布
            dist = torch.distributions.Normal(mu, std)

            # 新log概率
            new_logp = dist.log_prob(actions_flat).sum(-1)
            metrics["new_logp_mean"].append(new_logp.mean().item())

            # 熵
            entropy = dist.entropy().mean()

            # Ratio计算
            log_ratio = new_logp - old_logps_flat
            ratio = torch.exp(log_ratio)
            #ratio = torch.clamp(ratio, 0.1, 10.0)  # 限制ratio范围
            ratio_mean = ratio.mean().item()
            ratio_std = ratio.std().item()

            # 策略损失
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_flat
            policy_loss = -torch.min(surr1, surr2).mean()

            # Critic损失
            quantiles = self.agent.evaluate(states_flat)
            target = returns_flat.unsqueeze(1).expand_as(quantiles)
            critic_loss = quantile_huber_loss(quantiles, target, self.agent.taus)

            # 总损失
            loss = policy_loss + self.vf_coef * critic_loss - self.ent_coef * entropy
            print(f'critic_loss: {critic_loss.item()} | policy_loss: {policy_loss.item()} | loss: {loss.item()}')


            # ----- 优化步骤 -----
            self.optimizer.zero_grad()
            loss.backward()  # ✅ 每次迭代都是新的计算图

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.max_grad_norm
            )

            # 检查梯度
            if torch.isnan(grad_norm) or grad_norm > 100:
                self.optimizer.zero_grad()
                break

            self.optimizer.step()

            # ----- KL散度检查 -----
            approx_kl = (old_logps_flat - new_logp).mean().item()

            # 记录指标
            metrics["policy_loss"].append(policy_loss.item())
            metrics["critic_loss"].append(critic_loss.item())
            metrics["total_loss"].append(loss.item())
            metrics["entropy"].append(entropy.item())
            metrics["kl_divergence"].append(approx_kl)
            metrics["gradient_norm"].append(grad_norm.item())
            metrics["ratio_mean"].append(ratio_mean)
            metrics["ratio_std"].append(ratio_std)

            # KL提前停止
            # if approx_kl > 1.5 * self.target_kl and iter_idx > 0:
            #     break

        self.update_step += 1

        # ========= 8. 汇总指标 =========
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



