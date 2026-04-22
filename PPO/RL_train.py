import pickle
import torch
import pandas as pd
from tqdm import trange
from rdkit import RDLogger
from collections import defaultdict
from env import CJTVAE_XGB_Env
from agent import MolecularRLAgent, MultiStepPPOTrainer
from jtnn_vae import JTNNVAE
from vocab import Vocab
import os
import numpy as np
import signal
import time
from plot import plot_training_metrics, plot_training_dashboard, plot_reward_comparison, plot_baseline_results
from control_group import (
    RandomSearchBaseline,
    GreedySearchBaseline,
    CEMBaseline,
    BaselineConfig,
)
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from collections import defaultdict
import numpy as np


RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 48
BATCH_SIZE = 16

TOTAL_UPDATES = 100
LOG_INTERVAL = 50
SAVE_INTERVAL = 10

PPO_LR = 1e-4
CLIP_EPS = 0.2
ENT_COEF = 0.02
VF_COEF = 0.5

ROLLOUT_STEPS = 3
TRAIN_ITERS = 3
GAMMA = 0.9             # 折扣因子
GAE_LAMBDA = 0.95

REWARD_THRESHOLD_ = 5.5
TOPK_REPLAY = 2

SIMILARITY_THRESHOLD = 0.8

CJTVAE_CKPT = "/home/dc/data_new/finetune/best_cjtvae_cond_finetuned_final.pth"
VOCAB_PATH = "/home/dc/data_new/new_vocab.txt"
XGB_MODEL_PATH = "/home/dc/data_new/XGB/XGB.pkl"

SAVE_DIR = "/home/dc/data_new/ppo_results/new_ppo/5"
os.makedirs(SAVE_DIR, exist_ok=True)

baseline_config = BaselineConfig()

baseline_config.LATENT_DIM = LATENT_DIM
baseline_config.BATCH_SIZE = BATCH_SIZE
baseline_config.TOTAL_STEPS = TOTAL_UPDATES
baseline_config.ROLLOUT_STEPS = ROLLOUT_STEPS
baseline_config.SAVE_STEPS = SAVE_INTERVAL

class StepTimeout:
    def __init__(self, step, timeout_seconds):
        self.step = step
        self.timeout_seconds = timeout_seconds
        self.timed_out = False
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

        def handler(signum, frame):
            self.timed_out = True
            raise TimeoutError(f"Step {self.step} timed out after {self.timeout_seconds}s")

        try:
            self.original_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.timeout_seconds)
        except (AttributeError, ValueError):
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            signal.alarm(0)
            if hasattr(self, 'original_handler'):
                signal.signal(signal.SIGALRM, self.original_handler)
        except:
            pass

        if not self.timed_out and self.start_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                self.timed_out = True
                print(f"\n⏰ Step {self.step} 实际耗时{elapsed:.1f}s超时")

        if exc_type is not None and exc_type == TimeoutError:
            return True

        return False


class EliteReplayBuffer:
    def __init__(self, max_size=200, noise_std=0.05, similarity_threshold=0.8):
        self.max_size = max_size
        self.noise_std = noise_std
        self.similarity_threshold = similarity_threshold
        self.buffer = []
        self.scores = []
        self.scaffolds = []  # 新增：存储每个轨迹的骨架

    def _get_scaffold(self, smiles):
        """提取Murcko骨架"""
        if smiles is None or smiles == '':
            return None
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except:
            return None

    def _is_duplicate_scaffold(self, new_scaffold):
        """检查新骨架是否与已有骨架重复"""
        if new_scaffold is None:
            return False

        for existing_scaffold in self.scaffolds:
            if existing_scaffold is None:
                continue
            # 计算骨架相似度
            try:
                mol1 = Chem.MolFromSmiles(new_scaffold)
                mol2 = Chem.MolFromSmiles(existing_scaffold)
                if mol1 is None or mol2 is None:
                    continue
                fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=512)
                fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=512)
                sim = TanimotoSimilarity(fp1, fp2)
                if sim >= self.similarity_threshold:
                    return True
            except:
                continue
        return False

    def add_trajectory(self, states, actions, rewards, final_reward, final_smiles):
        # 骨架去重检查
        new_scaffold = self._get_scaffold(final_smiles)
        if new_scaffold is not None and self._is_duplicate_scaffold(new_scaffold):
            # 检查是否比已有同类骨架的奖励更高
            for i, existing_scaffold in enumerate(self.scaffolds):
                if existing_scaffold is None:
                    continue
                try:
                    mol1 = Chem.MolFromSmiles(new_scaffold)
                    mol2 = Chem.MolFromSmiles(existing_scaffold)
                    if mol1 is None or mol2 is None:
                        continue
                    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=512)
                    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=512)
                    sim = TanimotoSimilarity(fp1, fp2)
                    if sim >= self.similarity_threshold:
                        if final_reward > self.scores[i]:
                            # 替换旧的
                            self.buffer.pop(i)
                            self.scores.pop(i)
                            self.scaffolds.pop(i)
                            break
                        else:
                            # 奖励不如旧的，直接返回
                            return
                except:
                    continue

        trajectory = {
            'states': [z.detach().cpu().numpy() for z in states],
            'actions': [a.detach().cpu().numpy() for a in actions],
            'rewards': np.array(rewards, dtype=np.float32),
            'final_reward': float(final_reward),
            'final_smiles': final_smiles,
            'length': len(actions)
        }

        self.buffer.append(trajectory)
        self.scores.append(float(final_reward))
        self.scaffolds.append(new_scaffold)

        # 按final_reward排序并保留top-k
        if len(self.buffer) > self.max_size:
            sorted_indices = np.argsort(self.scores)[::-1]
            self.buffer = [self.buffer[i] for i in sorted_indices[:self.max_size]]
            self.scores = [self.scores[i] for i in sorted_indices[:self.max_size]]
            self.scaffolds = [self.scaffolds[i] for i in sorted_indices[:self.max_size]]

    def sample_initial_z(self, batch_size, latent_dim, device, sample_type='last_state'):
        if len(self.buffer) == 0:
            return torch.randn(batch_size, latent_dim, device=device)

        scores = np.array(self.scores)
        scores = scores - scores.min() + 1e-8  # 确保正数
        probs = scores / scores.sum()

        traj_indices = np.random.choice(
            len(self.buffer),
            size=batch_size,
            replace=True,
            p=probs
        )

        z_samples = []
        for traj_idx in traj_indices:
            trajectory = self.buffer[traj_idx]

            if sample_type == 'last_state':
                z_np = trajectory['states'][-1]
            elif sample_type == 'first_state':
                z_np = trajectory['states'][0]
            else:  # random_state
                state_idx = np.random.randint(0, len(trajectory['states']))
                z_np = trajectory['states'][state_idx]

            noise = np.random.normal(0, self.noise_std, z_np.shape)
            z_noisy = z_np + noise
            z_samples.append(z_noisy)

        return torch.tensor(np.array(z_samples), dtype=torch.float32, device=device)

    def __len__(self):
        return len(self.buffer)


def collect_rollout(agent, env, z0, rollout_steps):
    """
    收集轨迹，返回原始SF值而不是差分
    """
    states = [z0]  # 所有状态
    actions = []  # 所有动作
    logps = []  # 动作概率
    reward_values = []
    smiles_list = []
    valid_list = []
    infos_list = []

    current_z = z0

    # 获取初始状态的SF
    smiles0, valid0 = env.decode(z0)
    reward0, info0 = env.compute_reward(z0, smiles0, valid0)

    for t in range(rollout_steps):
        # 1. Actor决定动作
        action, logp = agent.act(current_z)

        # 2. 状态更新
        next_z = current_z + action

        print(f'第{t + 1}步解码中......')
        smiles, valid_mask = env.decode(next_z)
        print(f'smiles: {smiles}解码成功')
        reward, infos = env.compute_reward(next_z, smiles, valid_mask)
        print(f'奖励计算成功,奖励{reward}')
        # 4. 存储数据
        states.append(next_z)
        actions.append(action)
        logps.append(logp)
        reward_values.append(reward)
        smiles_list.append(smiles)
        valid_list.append(valid_mask)
        infos_list.append(infos)

        # 5. 更新当前状态
        current_z = next_z

    return states, actions, logps, reward_values, smiles_list, valid_list, infos_list


def record_batch_multi_step(step, trajectories, smiles_counter, records,
                            record_intermediate=False, record_final_only=True):
    batch_size = len(trajectories['all_smiles'][0])
    num_steps = len(trajectories['all_smiles'])

    if record_final_only:
        final_smiles = trajectories['all_smiles'][-1]
        final_rewards = trajectories['all_rewards'][-1]
        final_valid = trajectories['all_valid'][-1]
        final_infos = trajectories['all_infos'][-1] if 'all_infos' in trajectories else [{}] * batch_size

        return record_batch_single_step(
            step, final_smiles, final_rewards, final_valid,
            final_infos, smiles_counter, records
        )

    else:
        for t in range(num_steps):
            smiles_t = trajectories['all_smiles'][t]
            rewards_t = trajectories['all_rewards'][t]
            valid_t = trajectories['all_valid'][t]
            infos_t = trajectories['all_infos'][t] if 'all_infos' in trajectories else [{}] * batch_size

            for i in range(batch_size):
                smi = smiles_t[i]
                smiles_counter[smi] += 1

                rec = {
                    "step": step,
                    "traj_step": t,  # 新增：轨迹中的步数
                    "index_in_batch": i,
                    "reward": float(rewards_t[i]),
                    "valid": bool(valid_t[i]),
                    "smiles": smi,
                    "visit_count": smiles_counter[smi],
                    "is_final_step": (t == num_steps - 1)
                }

                if infos_t and i < len(infos_t) and "soft_sf" in infos_t[i]:
                    rec.update({
                        "soft_sf": infos_t[i]["soft_sf"],
                        "sim_mean": infos_t[i].get("sim_mean", np.nan),
                        "explore_bonus": infos_t[i].get("explore_bonus", np.nan),
                    })
                else:
                    rec.update({
                        "soft_sf": np.nan,
                        "sim_mean": np.nan,
                        "explore_bonus": np.nan,
                    })

                if t < num_steps - 1:
                    rec["is_intermediate"] = True
                    if "soft_sf" not in rec or np.isnan(rec["soft_sf"]):
                        rec["soft_sf"] = 0.0  # 默认值

                records.append(rec)

        return records


def record_batch_single_step(step, smiles, rewards, valid_mask, infos, smiles_counter, records):
    batch_size = len(smiles)

    for i in range(batch_size):
        smi = smiles[i]
        smiles_counter[smi] += 1

        rec = {
            "step": step,
            "index_in_batch": i,
            "reward": float(rewards[i]),
            "valid": bool(valid_mask[i]),
            "smiles": smi,
            "visit_count": smiles_counter[smi],
        }

        if infos and i < len(infos) and "soft_sf" in infos[i]:
            rec.update({
                "soft_sf": infos[i]["soft_sf"],
                "sim_mean": infos[i].get("sim_mean", np.nan),
                "explore_bonus": infos[i].get("explore_bonus", np.nan),
            })
        else:
            rec.update({
                "soft_sf": np.nan,
                "sim_mean": np.nan,
                "explore_bonus": np.nan,
            })

        records.append(rec)

    return records

def save_top_molecules(df, save_dir):
    top_df = (
        df.sort_values("reward", ascending=False)
        .drop_duplicates("smiles")
        .head(50)
    )
    top_df.to_excel(os.path.join(save_dir, "top_molecules.xlsx"), index=False)

def load_models():
    vocab = Vocab([x.strip() for x in open(VOCAB_PATH)])

    cjtvae = JTNNVAE(
        vocab=vocab,
        hidden_size=256,
        latent_size=LATENT_DIM,
        depthT=15,
        depthG=3,
        cond_dim=0,
    )

    ckpt = torch.load(CJTVAE_CKPT, map_location=DEVICE)
    cjtvae.load_state_dict(ckpt, strict=False)
    cjtvae = cjtvae.to(DEVICE).eval()

    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)

    return cjtvae, xgb_model, vocab


def sample_initial_z_mixed(batch_size, latent_dim, device, replay_buffer, replay_ratio=0.2):
    z_list = []

    # 1️⃣ replay部分
    n_replay = int(batch_size * replay_ratio)
    if n_replay > 0 and len(replay_buffer) > 0:
        z_replay = replay_buffer.sample_initial_z(
            n_replay, latent_dim, device, sample_type='last_state'
        )
        z_list.append(z_replay)

    current_total = sum(z.shape[0] for z in z_list) if z_list else 0
    n_remaining = batch_size - current_total

    if n_remaining > 0:
        z_random = torch.randn(n_remaining, latent_dim, device=device)
        z_list.append(z_random)

    z0 = torch.cat(z_list, dim=0)

    if z0.shape[0] != batch_size:
        print("⚠ 修正 batch 大小")
        z0 = torch.randn(batch_size, latent_dim, device=device)

    return z0


def get_replay_ratio(step, total_steps=100):
    # 中心点设在总步数的30%处
    center = total_steps * 0.3  # 30%位置

    # 宽度设为总步数的15%
    width = total_steps * 0.15

    x = (step - center) / width
    ratio = 0.9 / (1 + np.exp(-x))

    return max(0.1, min(0.9, ratio))


def apply_repeat_penalty_final_only(rewards_list, smiles_list, smiles_counter, coef=0.1, similarity_threshold=SIMILARITY_THRESHOLD):
    """
    重复惩罚池子只包含最终步的分子，但中间步骤如果与最终步分子重复也会受到惩罚

    Args:
        similarity_threshold: 相似度阈值，默认为1.0（完全匹配）。设为0.9则相似度>0.9即视为重复
    """
    num_steps = len(rewards_list)
    batch_size = len(rewards_list[0])
    adjusted_list = []

    final_step_idx = num_steps - 1

    # 辅助函数：计算相似度
    def get_similarity(smi1, smi2):
        """计算两个SMILES的Tanimoto相似度"""
        if smi1 == smi2:  # 完全相同直接返回1，跳过计算
            return 1.0
        try:
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)
            if mol1 is None or mol2 is None:
                return 0.0
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            return TanimotoSimilarity(fp1, fp2)
        except:
            return 0.0

    # 1. 统计最后一步的分子（构建惩罚池）
    final_smiles_pool = defaultdict(int)  # 记录每个最终分子在当前batch中出现的次数
    for i in range(batch_size):
        smi = smiles_list[final_step_idx][i]
        if smi:  # 只统计有效分子
            final_smiles_pool[smi] += 1

    # 2. 对所有步应用惩罚（基于最终步分子池）
    for t in range(num_steps):
        step_adjusted = []
        step_rewards = rewards_list[t]
        step_smiles = smiles_list[t]

        for i in range(batch_size):
            r = step_rewards[i]
            smi = step_smiles[i]

            if not smi:  # 无效分子，不调整
                step_adjusted.append(r)
                continue

            # 检查这个分子是否与最终步分子池中的任何分子相似
            is_similar = False
            max_similarity = 0.0

            # 只有当相似度阈值小于1时才需要计算相似度
            if similarity_threshold < 1.0:
                for final_smi in final_smiles_pool.keys():
                    sim = get_similarity(smi, final_smi)
                    max_similarity = max(max_similarity, sim)
                    if max_similarity >= similarity_threshold:
                        is_similar = True
                        break
            else:
                # 阈值=1.0时保持原有逻辑，只检查完全匹配
                is_similar = (smi in final_smiles_pool)
                max_similarity = 1.0 if is_similar else 0.0

            if is_similar:
                global_count = smiles_counter.get(smi, 0)  # 历史出现次数
                # 计算当前batch中相似的最终分子数量
                if similarity_threshold < 1.0:
                    batch_count = 0
                    for final_smi, count in final_smiles_pool.items():
                        if get_similarity(smi, final_smi) >= similarity_threshold:
                            batch_count += count
                else:
                    batch_count = final_smiles_pool[smi]  # 当前batch中作为最终步出现的次数

                total_visits = global_count + batch_count
                # 根据相似度调整惩罚强度（越相似惩罚越重）
                penalty_multiplier = max_similarity if similarity_threshold < 1.0 else 1.0
                penalty = coef * penalty_multiplier * np.log(total_visits + 1)
                adjusted = r - penalty
            else:
                adjusted = r

            step_adjusted.append(adjusted)

        adjusted_list.append(np.array(step_adjusted, dtype=np.float32))

    # 3. 更新全局计数器（只累加最终步的分子）
    for smi, count in final_smiles_pool.items():
        smiles_counter[smi] += count

    return adjusted_list, smiles_counter

def main():
    print("Using device:", DEVICE)

    cjtvae, xgb_model, vocab = load_models()

    env = CJTVAE_XGB_Env(
        cjtvae_model=cjtvae,
        xgb_model=xgb_model,
        device=DEVICE,
    )

    print("Env ready.")

    agent = MolecularRLAgent(
        z_dim=LATENT_DIM
    ).to(DEVICE)

    ppo = MultiStepPPOTrainer(
        agent=agent,
        lr=PPO_LR,
        ent_coef=ENT_COEF,
        clip_eps=CLIP_EPS,
        vf_coef=VF_COEF,
        device=DEVICE,
        gamma=0.9,  # 折扣因子（对短轨迹用稍小的值）
        gae_lambda=0.95,  # GAE lambda
        rollout_steps=ROLLOUT_STEPS,  # rollout步数（从2步开始）
        train_iters=TRAIN_ITERS
    )

    print("Agent & PPO ready.")

    replay_buffer = EliteReplayBuffer(max_size=100)

    records = []
    smiles_counter = defaultdict(int)
    metrics_history = []
    ppo_reward_means = []

    best_reward = -1e9
    best_record = None

    for step in trange(1, TOTAL_UPDATES + 1, desc="PPO Training"):
        with StepTimeout(step, 120) as timeout:
            # ========= 采样初始点 =========
            replay_ratio = get_replay_ratio(step, TOTAL_UPDATES)

            z0 = sample_initial_z_mixed(
                batch_size=BATCH_SIZE,
                latent_dim=LATENT_DIM,
                device=DEVICE,
                replay_buffer=replay_buffer,
                replay_ratio=replay_ratio
            )
            print('初始点采样完成')
            # ========= 多步rollout =========
            states, actions, logps, raw_rewards_list, smiles_list, valid_list, infos_list = \
                collect_rollout(agent, env, z0, ppo.rollout_steps)

            # ========= 重复惩罚 =========
            adjusted_rewards_list, smiles_counter = apply_repeat_penalty_final_only(
                raw_rewards_list, smiles_list, smiles_counter, coef=0.2,similarity_threshold=SIMILARITY_THRESHOLD
            )



            dones = []
            for t in range(ppo.rollout_steps):
                if t == ppo.rollout_steps - 1:
                    dones.append(torch.ones(BATCH_SIZE, device=DEVICE))
                else:
                    dones.append(torch.zeros(BATCH_SIZE, device=DEVICE))
            # ========= PPO更新 =========
            rewards_tensors = [torch.tensor(r, device=DEVICE) for r in adjusted_rewards_list]
            update_metrics = ppo.update_trajectory(
                states=states,
                actions=actions,
                old_logps=logps,
                rewards=rewards_tensors,
                dones=dones,
            )
            print('PPO更新完成')

            # ========= 提取最终步数据 =========
            final_z = states[-1]
            final_smiles = smiles_list[-1]
            final_valid_mask = valid_list[-1]
            final_rewards = adjusted_rewards_list[-1]
            final_infos = infos_list[-1]

            update_metrics['final_rewards'] = final_rewards
            ppo_reward_means.append(np.mean(final_rewards))

            if update_metrics is not None:
                metrics_history.append(update_metrics)

            # ========= 记录数据（只记录一次！）=========
            trajectories_data = {
                'all_smiles': smiles_list,
                'all_rewards': adjusted_rewards_list,
                'all_valid': valid_list,
                'all_infos': infos_list,
                'states': states,
            }

            records = record_batch_multi_step(
                step=step,
                trajectories=trajectories_data,
                smiles_counter=smiles_counter,
                records=records,
                record_final_only=False,
            )

            # ========= 回放缓冲区 =========
            for t in range(ppo.rollout_steps):
                infos_t = infos_list[t]
                smiles_t = smiles_list[t]
                rewards_t = adjusted_rewards_list[t]

                candidates = []
                for i in range(BATCH_SIZE):
                    if not valid_list[t][i]:
                        continue

                    total_reward = rewards_t[i]  # 改用Total Reward
                    if total_reward >= REWARD_THRESHOLD_:  # 改用Total Reward阈值
                        candidates.append((total_reward, i))

                candidates.sort(reverse=True)

                for _, i in candidates[:TOPK_REPLAY]:
                    replay_buffer.add_trajectory(
                        states=[s[i] for s in states[:t + 1]],
                        actions=[a[i] for a in actions[:t + 1]],
                        rewards=[r[i] for r in adjusted_rewards_list[:t + 1]],
                        final_reward=rewards_t[i],
                        final_smiles=smiles_t[i]
                    )


        # ========= 日志输出 =========
        if step % LOG_INTERVAL == 0:
            valid_count = sum(final_valid_mask)
            valid_ratio = valid_count / BATCH_SIZE
            avg_reward = np.mean(final_rewards)
            max_reward = np.max(final_rewards) if len(final_rewards) > 0 else 0

            sf_values = [info.get("soft_sf", 0) for info in final_infos if info.get("soft_sf") is not None]
            avg_sf = np.mean(sf_values) if sf_values else 0
            max_sf = np.max(sf_values) if sf_values else 0

            print(f"\n[Step {step}]")
            print(f"  有效分子: {valid_count}/{BATCH_SIZE} ({valid_ratio:.0%})")
            print(f"  最终奖励: 平均={avg_reward:.3f}, 最高={max_reward:.3f}")
            print(f"  最终SF值: 平均={avg_sf:.3f}, 最高={max_sf:.3f}")

        # ========= 更新最佳记录（从records中获取）=========
        # 不需要再遍历records，直接比较final_rewards
        for i in range(BATCH_SIZE):
            if final_valid_mask[i] and final_rewards[i] > best_reward:
                best_reward = final_rewards[i]
                best_record = {
                    "step": step,
                    "reward": final_rewards[i],
                    "smiles": final_smiles[i],
                    "valid": True,
                    "soft_sf": final_infos[i].get("soft_sf", np.nan),
                }

        if step % SAVE_INTERVAL == 0:

            torch.save(
                {
                    "agent": agent.state_dict(),
                    "ppo": ppo.optimizer.state_dict(),
                    "step": step,
                    "best_record": best_record,
                    "metrics_history": metrics_history,
                },
                os.path.join(SAVE_DIR, f"checkpoint_step{step}.pth")
            )
            df = pd.DataFrame(records)
            df.to_excel(os.path.join(SAVE_DIR, "training_log.xlsx"), index=False)

            metrics_df = pd.DataFrame(metrics_history)
            metrics_df.to_csv(os.path.join(SAVE_DIR, "training_metrics.csv"), index=False)



            final_plot_path = os.path.join(SAVE_DIR, "final_training_metrics.png")
            plot_training_metrics(metrics_history, save_path=final_plot_path)

            save_top_molecules(df, SAVE_DIR)


    print("\nTraining finished.")
    print("Best reward:", best_reward)
    print("Best molecule:", best_record["smiles"])

    # print("\n========== Running Control Groups ==========")
    # PPO dataframe
    ppo_df = pd.DataFrame(records)

    plot_baseline_results(
        ppo_df,
        ppo_reward_means,
        os.path.join(SAVE_DIR, "ppo_plot.png")
    )


if __name__ == "__main__":
    main()