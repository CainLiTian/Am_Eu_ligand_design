import os
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import umap
import pandas as pd

PPO_LR = 0.001

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import os


def create_fingerprint_visualizations(df, save_dir):
    if df.empty or "smiles" not in df.columns:
        print("⚠️ df 为空或不包含 smiles，跳过")
        return

    os.makedirs(save_dir, exist_ok=True)

    # 提取数据
    fps = []
    rewards = []
    smiles_list = []

    for _, row in df.iterrows():
        smi = row.get("smiles")
        reward = row.get("reward")

        if pd.isna(smi) or smi is None or smi == '':
            continue

        if reward is None or pd.isna(reward):
            continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue

        # 生成指纹
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)

        fps.append(arr)
        rewards.append(float(reward))
        smiles_list.append(smi)

    if len(fps) < 10:
        print(f"有效分子过少（{len(fps)}），跳过可视化")
        return

    # 转换为numpy数组
    X = np.stack(fps)
    rewards = np.array(rewards)

    print(f"分子数量: {len(fps)}")
    print(f"Reward范围: {rewards.min():.3f} - {rewards.max():.3f}")

    # 固定点大小
    point_size = 20

    # 2. t-SNE降维
    print("正在计算t-SNE...")
    perplexity = 40
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        n_iter=1000,
        metric="jaccard"
    )
    X_tsne = tsne.fit_transform(X)

    # 绘制t-SNE-Reward图
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=rewards,
        cmap="viridis",
        s=point_size,
        alpha=0.6,
        edgecolors="none"
    )
    plt.colorbar(sc, label="Reward")
    plt.title("t-SNE of Generated Molecules (Color = Reward)")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tsne_reward_jaccard.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ t-SNE-Reward图已保存")

    print(f"\n所有图片已保存到: {save_dir}")



def plot_training_metrics(metrics_history, save_path=None, window_size=50):
    if not metrics_history:
        print("No metrics history to plot")
        return

    # 准备数据
    steps = [m.get('update_step', i) for i, m in enumerate(metrics_history)]

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

    # 1. 损失曲线
    ax1 = axes[0, 0]
    policy_loss = [m.get('policy_loss', 0) for m in metrics_history]
    critic_loss = [m.get('critic_loss', 0) for m in metrics_history]
    total_loss = [m.get('total_loss', 0) for m in metrics_history]

    # 滑动平均
    def moving_average(data, window):
        return np.convolve(data, np.ones(window) / window, mode='valid')

    if len(policy_loss) >= window_size:
        policy_loss_ma = moving_average(policy_loss, window_size)
        critic_loss_ma = moving_average(critic_loss, window_size)
        total_loss_ma = moving_average(total_loss, window_size)

        ax1.plot(steps[window_size - 1:], policy_loss_ma, alpha=0.8, linewidth=2)
        ax1.plot(steps[window_size - 1:], critic_loss_ma, alpha=0.8, linewidth=2)
        ax1.plot(steps[window_size - 1:], total_loss_ma, alpha=0.8, linewidth=2)

    ax1.plot(steps, policy_loss, alpha=0.7, label='Policy Loss')
    ax1.plot(steps, critic_loss, alpha=0.7, label='Critic Loss')
    ax1.plot(steps, total_loss, alpha=0.7, label='Total Loss')

    ax1.set_xlabel('Update Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Losses')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. 策略质量指标
    ax2 = axes[0, 1]
    entropy = [m.get('entropy', 0) for m in metrics_history]
    kl_divergence = [m.get('kl_divergence', 0) for m in metrics_history]

    ax2.plot(steps, entropy, 'b-', label='Policy Entropy', linewidth=2)
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Entropy', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(True, alpha=0.3)

    ax2_kl = ax2.twinx()
    ax2_kl.plot(steps, kl_divergence, 'r-', label='KL Divergence', linewidth=2, alpha=0.7)
    ax2_kl.set_ylabel('KL Divergence', color='r')
    ax2_kl.tick_params(axis='y', labelcolor='r')
    ax2_kl.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='Target KL (0.01)')

    ax2.set_title('Policy Quality Metrics')
    ax2.legend(loc='upper left')
    ax2_kl.legend(loc='upper right')

    # 3. 比率和梯度范数
    ax3 = axes[0, 2]
    ratio_mean = [m.get('ratio_mean', 1) for m in metrics_history]
    ratio_std = [m.get('ratio_std', 0) for m in metrics_history]
    gradient_norm = [m.get('gradient_norm', 0) for m in metrics_history]

    ax3.plot(steps, ratio_mean, 'g-', label='Ratio Mean', linewidth=2)
    ax3.fill_between(steps,
                     np.array(ratio_mean) - np.array(ratio_std),
                     np.array(ratio_mean) + np.array(ratio_std),
                     alpha=0.2, color='g')

    ax3.set_xlabel('Update Step')
    ax3.set_ylabel('Importance Ratio', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Target Ratio (1.0)')
    ax3.axhspan(0.9, 1.1, alpha=0.1, color='g')
    ax3.grid(True, alpha=0.3)

    ax3_grad = ax3.twinx()
    ax3_grad.plot(steps, gradient_norm, 'm-', label='Gradient Norm', linewidth=2, alpha=0.7)
    ax3_grad.set_ylabel('Gradient Norm', color='m')
    ax3_grad.tick_params(axis='y', labelcolor='m')
    ax3_grad.axhline(y=0.5, color='m', linestyle='--', alpha=0.5, label='Max Norm (0.5)')

    ax3.set_title('Importance Ratio & Gradient Norm')
    ax3.legend(loc='upper left')
    ax3_grad.legend(loc='upper right')

    # 4. 价值函数指标
    ax4 = axes[1, 0]
    value_mean = [m.get('value_mean', 0) for m in metrics_history]
    value_std = [m.get('value_std', 0) for m in metrics_history]
    advantage_mean = [m.get('advantage_mean', 0) for m in metrics_history]

    ax4.plot(steps, value_mean, 'b-', label='Value Mean', linewidth=2)
    ax4.fill_between(steps,
                     np.array(value_mean) - np.array(value_std),
                     np.array(value_mean) + np.array(value_std),
                     alpha=0.2, color='b')
    ax4.set_xlabel('Update Step')
    ax4.set_ylabel('Value', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4.grid(True, alpha=0.3)

    ax4_adv = ax4.twinx()
    ax4_adv.plot(steps, advantage_mean, 'r-', label='Advantage Mean', linewidth=2, alpha=0.7)
    ax4_adv.set_ylabel('Advantage', color='r')
    ax4_adv.tick_params(axis='y', labelcolor='r')
    ax4_adv.axhline(y=0.0, color='r', linestyle='--', alpha=0.5)

    ax4.set_title('Value Function Metrics')
    ax4.legend(loc='upper left')
    ax4_adv.legend(loc='upper right')

    # 5. 最终奖励分析（综合版）
    ax5 = axes[1, 1]

    final_rewards_list = [m.get('final_rewards', []) for m in metrics_history]

    if final_rewards_list and any(len(r) > 0 for r in final_rewards_list):
        # 计算分位数
        percentiles = [50, 75, 90, 95]
        quantiles = {p: [] for p in percentiles}
        valid_steps = []

        # 计算Top-K
        top1 = []
        top5_avg = []
        all_avg = []

        for i, rewards in enumerate(final_rewards_list):
            if len(rewards) >= 5:  # 至少5个样本
                rewards_array = np.array(rewards)
                rewards_sorted = sorted(rewards_array, reverse=True)

                for p in percentiles:
                    quantiles[p].append(np.percentile(rewards_array, p))

                top1.append(rewards_sorted[0])
                top5_avg.append(np.mean(rewards_sorted[:5]))
                all_avg.append(np.mean(rewards_array))
                valid_steps.append(steps[i])

        if valid_steps:
            # 主要区域：50-90百分位
            ax5.fill_between(valid_steps,
                             quantiles[50],
                             quantiles[90],
                             alpha=0.2, color='blue', label='50th-90th percentile')

            # 高分位数线
            ax5.plot(valid_steps, quantiles[90], 'b-', linewidth=1.5, label='90th percentile')
            ax5.plot(valid_steps, quantiles[95], 'g-', linewidth=2, label='95th percentile')

            # Top-K表现
            ax5.plot(valid_steps, top1, 'r-', linewidth=2.5, label='Best molecule')
            ax5.plot(valid_steps, top5_avg, 'orange', linewidth=2, label='Top-5 avg')

            # 平均水平
            ax5.plot(valid_steps, all_avg, 'gray', linestyle='--',
                     linewidth=1.5, alpha=0.7, label='Overall avg')

            ax5.set_xlabel('Update Step')
            ax5.set_ylabel('Final Reward')
            ax5.set_title('Molecular Quality Tracking')
            ax5.legend(loc='upper left', fontsize=8)
            ax5.grid(True, alpha=0.3)

            # 最新统计
            recent = min(10, len(valid_steps))
            stats_text = (f'Latest:\n'
                          f'Best: {top1[-1]:.1f}\n'
                          f'Top-5 avg: {top5_avg[-1]:.1f}\n'
                          f'95th %ile: {quantiles[95][-1]:.1f}')
            ax5.text(0.02, 0.98, stats_text,
                     transform=ax5.transAxes, fontsize=8,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 6. 训练统计热力图
    ax6 = axes[1, 2]

    # 准备热力图数据
    metrics_to_show = ['policy_loss', 'critic_loss', 'entropy', 'kl_divergence',
                       'ratio_mean', 'gradient_norm', 'value_mean', 'reward_mean']

    # 计算最近N步的指标相关性
    recent_steps = min(200, len(metrics_history))
    if recent_steps > 10:
        # 获取最近N步的指标
        recent_metrics = []
        for metric_name in metrics_to_show:
            metric_values = [m.get(metric_name, 0) for m in metrics_history[-recent_steps:]]
            recent_metrics.append(metric_values)

        # 计算相关性矩阵
        corr_matrix = np.corrcoef(recent_metrics)

        # 绘制热力图
        im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

        # 设置刻度标签
        ax6.set_xticks(range(len(metrics_to_show)))
        ax6.set_yticks(range(len(metrics_to_show)))
        ax6.set_xticklabels([name.replace('_', '\n') for name in metrics_to_show],
                            rotation=45, ha='right', fontsize=8)
        ax6.set_yticklabels([name.replace('_', '\n') for name in metrics_to_show],
                            fontsize=8)

        # 添加文本标注
        for i in range(len(metrics_to_show)):
            for j in range(len(metrics_to_show)):
                text = ax6.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                ha="center", va="center", color="w", fontsize=7)

        ax6.set_title(f'Recent {recent_steps} Steps Metrics Correlation')
    else:
        ax6.text(0.5, 0.5, 'Not enough data\nfor correlation analysis',
                 ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Metrics Correlation')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to: {save_path}")
    plt.close()



def plot_training_dashboard(metrics_history, save_path=None):
    """
    PPO diagnostics dashboard
    Compatible with your current metrics_history structure
    """

    if not metrics_history:
        print("No metrics history to plot")
        return

    steps = [m["update_step"] for m in metrics_history]

    def get(metric, default=0.0):
        return np.array([m.get(metric, default) for m in metrics_history])

    # ====== Extract metrics ======
    reward_mean = get("reward_mean")
    reward_std = get("reward_std")

    policy_loss = get("policy_loss")
    critic_loss = get("critic_loss")
    total_loss = get("total_loss")

    entropy = get("entropy")
    kl = get("kl_divergence")

    ratio_mean = get("ratio_mean")
    ratio_std = get("ratio_std")

    gradient_norm = get("gradient_norm")

    value_mean = get("value_mean")
    value_std = get("value_std")

    advantage_mean = get("advantage_mean")
    advantage_std = get("advantage_std")

    action_std = get("action_std")
    mu_std = get("mu_std")

    learning_rate = get("learning_rate")

    # ====== Best-so-far (用 mean 近似) ======
    best_reward_so_far = np.maximum.accumulate(reward_mean)

    # ===============================
    # Create figure
    # ===============================
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("PPO Training Diagnostics Dashboard", fontsize=16)

    # ======================================================
    # 1️⃣ Reward Trend
    # ======================================================
    ax = axes[0, 0]
    ax.plot(steps, reward_mean, label="Reward Mean")
    ax.fill_between(
        steps,
        reward_mean - reward_std,
        reward_mean + reward_std,
        alpha=0.2
    )
    ax.plot(steps, best_reward_so_far, linestyle="--", label="Best Reward So Far")

    ax.set_title("Reward Trend")
    ax.set_xlabel("Update Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ======================================================
    # 2️⃣ Policy Stability
    # ======================================================
    ax = axes[0, 1]
    ax.plot(steps, kl, label="KL Divergence")
    ax.plot(steps, entropy, label="Entropy")
    ax.axhline(0.01, linestyle="--", alpha=0.5)
    ax.set_title("Policy Stability")
    ax.set_xlabel("Update Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ======================================================
    # 3️⃣ Importance Ratio
    # ======================================================
    ax = axes[1, 0]
    ax.plot(steps, ratio_mean, label="Ratio Mean")
    ax.fill_between(
        steps,
        ratio_mean - ratio_std,
        ratio_mean + ratio_std,
        alpha=0.2
    )
    ax.axhline(1.0, linestyle="--", alpha=0.5)
    ax.axhspan(0.9, 1.1, alpha=0.1)
    ax.set_title("Importance Sampling Ratio")
    ax.set_xlabel("Update Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ======================================================
    # 4️⃣ Critic vs Reward
    # ======================================================
    ax = axes[1, 1]
    ax.plot(steps, value_mean, label="Value Mean")
    ax.fill_between(
        steps,
        value_mean - value_std,
        value_mean + value_std,
        alpha=0.2
    )
    ax.plot(steps, reward_mean, label="Reward Mean")

    ax.set_title("Critic Tracking Reward")
    ax.set_xlabel("Update Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ======================================================
    # 5️⃣ Optimization Health
    # ======================================================
    ax = axes[2, 0]
    ax.plot(steps, policy_loss, label="Policy Loss")
    ax.plot(steps, critic_loss, label="Critic Loss")
    ax.plot(steps, gradient_norm, label="Gradient Norm")
    ax.set_title("Optimization Signals")
    ax.set_xlabel("Update Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ======================================================
    # 6️⃣ Exploration Diagnostics
    # ======================================================
    ax = axes[2, 1]
    ax.plot(steps, action_std, label="Action Std")
    ax.plot(steps, mu_std, label="Mu Std")
    ax.plot(steps, advantage_std, label="Adv Std")
    ax.plot(steps, learning_rate, label="Learning Rate")

    ax.set_title("Exploration & Advantage Diagnostics")
    ax.set_xlabel("Update Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Dashboard saved to {save_path}")

    plt.close()


def plot_reward_comparison(
        rl_df,
        random_df,
        greedy_df,
        cem_df,
        save_path,
        window_size=3,
        total_step=None):
    """
    比较不同算法的reward曲线

    Args:
        rl_df: RL算法的DataFrame
        random_df: Random Search的DataFrame
        greedy_df: Greedy Search的DataFrame
        cem_df: CEM的DataFrame
        save_path: 保存路径
        window_size: 滑动平均的窗口大小
        total_step: 规范化的总step数，如果提供则只取前total_step个step
    """

    # 定义计算滑动平均的函数
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # 从每个DataFrame计算step平均reward，并规范step数量
    def get_step_means(df, total_step=None):
        step_means = df.groupby('step')['reward'].mean().reset_index()

        if total_step is not None:
            # 获取唯一的steps并排序
            unique_steps = sorted(step_means['step'].unique())

            # 只取前total_step个step
            if len(unique_steps) > total_step:
                valid_steps = unique_steps[:total_step]
                step_means = step_means[step_means['step'].isin(valid_steps)]

        return step_means['step'].values, step_means['reward'].values

    # 获取每个算法的step和平均reward
    rl_steps, rl_rewards = get_step_means(rl_df, total_step)
    random_steps, random_rewards = get_step_means(random_df, total_step)
    greedy_steps, greedy_rewards = get_step_means(greedy_df, total_step)
    cem_steps, cem_rewards = get_step_means(cem_df, total_step)

    plt.figure(figsize=(7, 5))

    # 绘制原始曲线（半透明）
    plt.plot(rl_steps, rl_rewards, alpha=0.3, linewidth=1, color='blue')
    plt.plot(random_steps, random_rewards, alpha=0.3, linewidth=1, color='orange')
    plt.plot(greedy_steps, greedy_rewards, alpha=0.3, linewidth=1, color='green')
    plt.plot(cem_steps, cem_rewards, alpha=0.3, linewidth=1, color='red')

    # 绘制滑动平均曲线
    if len(rl_rewards) >= window_size:
        plt.plot(rl_steps[window_size - 1:], moving_average(rl_rewards, window_size),
                 label="RL", linewidth=2, color='blue')
    else:
        plt.plot(rl_steps, rl_rewards, label="RL", linewidth=2, color='blue')

    if len(random_rewards) >= window_size:
        plt.plot(random_steps[window_size - 1:], moving_average(random_rewards, window_size),
                 label="Random", linestyle="--", linewidth=2, color='orange')
    else:
        plt.plot(random_steps, random_rewards, label="Random", linestyle="--", linewidth=2, color='orange')

    if len(greedy_rewards) >= window_size:
        plt.plot(greedy_steps[window_size - 1:], moving_average(greedy_rewards, window_size),
                 label="Greedy", linestyle="--", linewidth=2, color='green')
    else:
        plt.plot(greedy_steps, greedy_rewards, label="Greedy", linestyle="--", linewidth=2, color='green')

    if len(cem_rewards) >= window_size:
        plt.plot(cem_steps[window_size - 1:], moving_average(cem_rewards, window_size),
                 label="CEM", linestyle="--", linewidth=2, color='red')
    else:
        plt.plot(cem_steps, cem_rewards, label="CEM", linestyle="--", linewidth=2, color='red')

    plt.xlabel("Step")
    plt.ylabel("Mean Reward")

    plt.title("Reward Mean vs Step (Algorithm Comparison)")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_baseline_results(df, reward_means, save_path):

    rewards = df["reward"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(reward_means)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward Mean vs Step")

    axes[1].hist(rewards, bins=50)
    axes[1].set_xlabel("Reward")
    axes[1].set_ylabel("Molecule Count")
    axes[1].set_title("Reward Distribution")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_baseline(df, save_path, window_size=3):
    # 按step计算平均reward
    step_means = df.groupby('step')['reward'].mean().reset_index()
    steps = step_means['step'].values
    rewards = step_means['reward'].values

    # 计算滑动平均
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # 获取所有reward值用于统计分析
    all_rewards = df["reward"].values

    # 计算reward分布的统计量
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    median_reward = np.median(all_rewards)
    q25 = np.percentile(all_rewards, 25)
    q75 = np.percentile(all_rewards, 75)
    min_reward = np.min(all_rewards)
    max_reward = np.max(all_rewards)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 第一个子图：平均reward vs step
    axes[0].plot(steps, rewards, alpha=0.5, label='Raw Mean', linewidth=1)

    # 添加滑动平均曲线
    if len(rewards) >= window_size:
        smoothed_rewards = moving_average(rewards, window_size)
        smoothed_steps = steps[window_size - 1:]  # 对齐滑动平均后的步数
        axes[0].plot(smoothed_steps, smoothed_rewards, 'r-',
                     label=f'{window_size}-step Moving Avg', linewidth=2)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Reward Mean vs Step")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 第二个子图：reward分布直方图
    n, bins, patches = axes[1].hist(all_rewards, bins=50, edgecolor='black', alpha=0.7)

    # 在图上添加统计信息文本框
    stats_text = (
        f'Statistics:\n'
        f'Mean: {mean_reward:.3f}\n'
        f'Std: {std_reward:.3f}\n'
        f'Median: {median_reward:.3f}\n'
        f'Q1-Q3: [{q25:.3f}, {q75:.3f}]\n'
        f'Range: [{min_reward:.3f}, {max_reward:.3f}]'
    )

    # 添加文本框
    axes[1].text(0.95, 0.95, stats_text,
                 transform=axes[1].transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    axes[1].set_xlabel("Reward")
    axes[1].set_ylabel("Molecule Count")
    axes[1].set_title("Reward Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 可选：打印统计信息到控制台
    print(f"\nReward Distribution Statistics:")
    print(f"  Mean: {mean_reward:.4f}")
    print(f"  Std: {std_reward:.4f}")
    print(f"  Median: {median_reward:.4f}")
    print(f"  Q1: {q25:.4f}")
    print(f"  Q3: {q75:.4f}")
    print(f"  Min: {min_reward:.4f}")
    print(f"  Max: {max_reward:.4f}")
    print(f"  Total molecules: {len(all_rewards)}")

