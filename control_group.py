import torch
import numpy as np
import pandas as pd
from tqdm import trange
from plot import plot_baseline_results
from collections import defaultdict


class BaselineConfig:

    LATENT_DIM = 48
    BATCH_SIZE = 32
    TOTAL_STEPS = 100
    ROLLOUT_STEPS = 3
    SAVE_STEPS = 10


class RandomSearchBaseline:

    def __init__(self, env, device, config):

        self.env = env
        self.device = device

        self.latent_dim = config.LATENT_DIM
        self.batch_size = config.BATCH_SIZE
        self.total_steps = config.TOTAL_STEPS
        self.save_steps = config.SAVE_STEPS

        # 添加计数器用于重复惩罚
        self.smiles_counter = defaultdict(int)  # 全局SMILES计数器
        self.penalty_coef = 0.1  # 惩罚系数

    def run(self):

        records = []
        reward_means = []

        for step in trange(1, self.total_steps + 1, desc="Random Search"):

            z = torch.randn(self.batch_size,
                            self.latent_dim,
                            device=self.device)

            smiles, valid = self.env.decode(z)
            rewards, infos = self.env.compute_reward(z, smiles, valid)

            # 统计当前批次中每个SMILES的出现次数
            temp_counter = defaultdict(int)
            for i, smi in enumerate(smiles):
                if smi and valid[i]:  # 只统计有效分子
                    temp_counter[smi] += 1

            # 应用重复惩罚
            adjusted_rewards = []
            for i in range(self.batch_size):
                r = rewards[i]
                smi = smiles[i]
                v = valid[i]

                if not smi or not v:  # 无效分子，不调整
                    adjusted_rewards.append(r)
                    continue

                # 计算总出现次数：历史次数 + 当前批次次数
                global_count = self.smiles_counter.get(smi, 0)
                batch_count = temp_counter[smi]
                total_visits = global_count + batch_count

                # 应用惩罚 (coef=0.1)
                penalty = self.penalty_coef * np.log(total_visits + 1)
                adjusted = r - penalty
                adjusted_rewards.append(adjusted)

            # 更新全局计数器
            for smi, count in temp_counter.items():
                self.smiles_counter[smi] += count

            reward_means.append(np.mean(adjusted_rewards))

            for i in range(self.batch_size):
                records.append({
                    "step": step,
                    "reward": float(adjusted_rewards[i]),  # 使用调整后的奖励
                    "smiles": smiles[i],
                    "valid": bool(valid[i])
                })

            if step % self.save_steps == 0:
                df = pd.DataFrame(records)
                df.to_excel('/home/dc/data_new/ppo_results/random.xlsx', index=False)
                plot_baseline_results(df, reward_means, '/home/dc/data_new/ppo_results/random.png')

        return df, reward_means


class GreedySearchBaseline:

    def __init__(self, env, device, config,
                 neighbors=4,
                 noise_std=0.2):

        self.env = env
        self.device = device

        self.latent_dim = config.LATENT_DIM
        self.batch_size = config.BATCH_SIZE
        self.total_steps = config.TOTAL_STEPS
        self.save_steps = config.SAVE_STEPS

        self.neighbors = neighbors
        self.noise_std = noise_std

        # 添加计数器用于重复惩罚
        self.smiles_counter = defaultdict(int)  # 全局SMILES计数器
        self.penalty_coef = 0.1  # 惩罚系数

    def run(self):

        records = []
        reward_means = []

        for step in trange(1, self.total_steps + 1, desc="Greedy Search"):

            z = torch.randn(self.batch_size,
                            self.latent_dim,
                            device=self.device)

            candidates = []

            for k in range(self.neighbors):
                noise = torch.randn_like(z) * self.noise_std
                candidates.append(z + noise)

            candidates = torch.stack(candidates)

            rewards_all = []
            smiles_all = []  # 存储所有候选的SMILES
            valid_all = []  # 存储所有候选的valid标志

            for k in range(self.neighbors):
                smiles, valid = self.env.decode(candidates[k])
                rewards, infos = self.env.compute_reward(
                    candidates[k], smiles, valid)

                rewards_all.append(rewards)
                smiles_all.append(smiles)
                valid_all.append(valid)

            rewards_all = np.stack(rewards_all)
            smiles_all = np.stack(smiles_all)
            valid_all = np.stack(valid_all)

            best_idx = np.argmax(rewards_all, axis=0)

            new_z = []
            new_smiles = []  # 存储选中的SMILES
            new_valid = []  # 存储选中的valid标志
            new_rewards = []  # 存储选中的原始奖励

            for i in range(self.batch_size):
                idx = best_idx[i]
                new_z.append(candidates[idx, i])
                new_smiles.append(smiles_all[idx, i])
                new_valid.append(valid_all[idx, i])
                new_rewards.append(rewards_all[idx, i])

            z = torch.stack(new_z)

            # 统计当前批次中每个SMILES的出现次数
            temp_counter = defaultdict(int)
            for i, smi in enumerate(new_smiles):
                if smi and new_valid[i]:  # 只统计有效分子
                    temp_counter[smi] += 1

            # 应用重复惩罚
            adjusted_rewards = []
            for i in range(self.batch_size):
                r = new_rewards[i]
                smi = new_smiles[i]
                v = new_valid[i]

                if not smi or not v:  # 无效分子，不调整
                    adjusted_rewards.append(r)
                    continue

                # 计算总出现次数：历史次数 + 当前批次次数
                global_count = self.smiles_counter.get(smi, 0)
                batch_count = temp_counter[smi]
                total_visits = global_count + batch_count

                # 应用惩罚 (coef=0.1)
                penalty = self.penalty_coef * np.log(total_visits + 1)
                adjusted = r - penalty
                adjusted_rewards.append(adjusted)

            # 更新全局计数器
            for smi, count in temp_counter.items():
                self.smiles_counter[smi] += count

            reward_means.append(np.mean(adjusted_rewards))

            for i in range(self.batch_size):
                records.append({
                    "step": step,
                    "reward": float(adjusted_rewards[i]),  # 使用调整后的奖励
                    "smiles": new_smiles[i],
                    "valid": bool(new_valid[i]),
                })

            if step % self.save_steps == 0:
                df = pd.DataFrame(records)
                df.to_excel('/home/dc/data_new/ppo_results/Greedy.xlsx', index=False)
                plot_baseline_results(df, reward_means, '/home/dc/data_new/ppo_results/Greedy.png')

        return df, reward_means


class CEMBaseline:

    def __init__(self, env, device, config,
                 elite_ratio=0.2):

        self.env = env
        self.device = device

        self.latent_dim = config.LATENT_DIM
        self.batch_size = config.BATCH_SIZE
        self.total_steps = config.TOTAL_STEPS
        self.save_steps = config.SAVE_STEPS

        self.elite_ratio = elite_ratio

        # 添加计数器用于重复惩罚
        self.smiles_counter = defaultdict(int)  # 全局SMILES计数器
        self.penalty_coef = 0.1  # 惩罚系数

    def run(self):

        records = []
        reward_means = []

        mu = torch.zeros(self.latent_dim, device=self.device)
        sigma = torch.ones(self.latent_dim, device=self.device)

        for step in trange(1, self.total_steps + 1, desc="CEM Search"):

            z = torch.randn(self.batch_size,
                            self.latent_dim,
                            device=self.device)

            z = z * sigma + mu

            smiles, valid = self.env.decode(z)
            rewards, infos = self.env.compute_reward(z, smiles, valid)

            # 统计当前批次中每个SMILES的出现次数
            temp_counter = defaultdict(int)
            for i, smi in enumerate(smiles):
                if smi and valid[i]:  # 只统计有效分子
                    temp_counter[smi] += 1

            # 应用重复惩罚
            adjusted_rewards = []
            for i in range(self.batch_size):
                r = rewards[i]
                smi = smiles[i]
                v = valid[i]

                if not smi or not v:  # 无效分子，不调整
                    adjusted_rewards.append(r)
                    continue

                # 计算总出现次数：历史次数 + 当前批次次数
                global_count = self.smiles_counter.get(smi, 0)
                batch_count = temp_counter[smi]
                total_visits = global_count + batch_count

                # 应用惩罚 (coef=0.1)
                penalty = self.penalty_coef * np.log(total_visits + 1)
                adjusted = r - penalty
                adjusted_rewards.append(adjusted)

            # 更新全局计数器
            for smi, count in temp_counter.items():
                self.smiles_counter[smi] += count

            reward_means.append(np.mean(adjusted_rewards))

            # 使用调整后的奖励进行精英选择
            adjusted_rewards_np = np.array(adjusted_rewards)
            elite_num = int(self.batch_size * self.elite_ratio)
            elite_idx = adjusted_rewards_np.argsort()[-elite_num:]

            elite_z = z[elite_idx]

            mu = elite_z.mean(dim=0)
            sigma = elite_z.std(dim=0) + 1e-6

            for i in range(self.batch_size):
                records.append({
                    "step": step,
                    "reward": float(adjusted_rewards[i]),  # 使用调整后的奖励
                    "original_reward": float(rewards[i]),  # 可选：保留原始奖励用于分析
                    "smiles": smiles[i],
                    "valid": bool(valid[i])
                })

            if step % self.save_steps == 0:
                df = pd.DataFrame(records)
                df.to_excel('/home/dc/data_new/ppo_results/cem.xlsx', index=False)
                plot_baseline_results(df, reward_means, '/home/dc/data_new/ppo_results/cem.png')

        return df, reward_means





