import torch
import numpy as np
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


class CJTVAE_XGB_Env:
    def __init__(
            self,
            cjtvae_model,
            xgb_model,
            device,
            data_file_path='/home/dc/data_new/S.xlsx',
            latent_dim=48,
            z_prior=None,
            # ---------- reward control ----------
            similarity_threshold=0.3,
            similarity_gate_k=0.05,  # ⭐ sigmoid 平滑度
            topk_similarity=2,  # ⭐ Top-K 相似分子
            topk_similarity_sol=5,
            reward_scale=1.0,  # ⭐ PPO 稳定性
    ):
        self.model = cjtvae_model.eval().to(device)
        self.xgb = xgb_model
        self.device = device
        self.latent_dim = latent_dim
        self.z_prior = z_prior
        print(f"Env initialized with latent_dim={latent_dim}")

        self.similarity_threshold = similarity_threshold
        self.similarity_gate_k = similarity_gate_k
        self.topk_similarity = topk_similarity
        self.topk_similarity_sol = topk_similarity_sol
        self.reward_scale = reward_scale

        # 加载数据文件
        self.data_file_path = data_file_path
        self._load_and_prepare_data()

    def _similarity_gate(self, sim):
        """
        Continuous gate in (0, 1)
        """
        return 1.0 / (1.0 + np.exp(-(sim - self.similarity_threshold) / self.similarity_gate_k))

    def _find_topk_similar(self, query_smiles, k=5, deduplicate=True):
        """
        快速在数据库中找到与查询分子最相似的k个分子

        Args:
            query_smiles: 查询分子的SMILES
            k: 返回的相似分子数量
            deduplicate: 是否对SMILES去重（True用于相似度计算，False用于溶剂匹配）

        Returns:
            - rows: DataFrame行的列表
            - sims: 相似度分数的列表
            - unique_smiles: 去重后的SMILES列表（当deduplicate=True时）
        """
        mol = Chem.MolFromSmiles(query_smiles)
        if mol is None:
            if deduplicate:
                return [], [], []
            else:
                return [], []

        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)

        # 计算与所有分子的相似度
        similarities = DataStructs.BulkTanimotoSimilarity(
            query_fp, self.database_explicit_bvs
        )

        if len(similarities) == 0:
            if deduplicate:
                return [], [], []
            else:
                return [], []

        if deduplicate:
            # ========== 去重模式：用于相似度计算 ==========
            # 创建 (smiles, max_similarity) 的映射
            smiles_to_max_sim = {}
            smiles_to_best_idx = {}

            for idx, sim in enumerate(similarities):
                smi = self.database_df.iloc[idx]['SMILES']
                if smi not in smiles_to_max_sim or sim > smiles_to_max_sim[smi]:
                    smiles_to_max_sim[smi] = sim
                    smiles_to_best_idx[smi] = idx

            # 按相似度排序，取top-k个唯一SMILES
            unique_smiles = sorted(smiles_to_max_sim.keys(),
                                   key=lambda x: smiles_to_max_sim[x],
                                   reverse=True)[:k]

            # 获取对应的相似度
            unique_sims = [smiles_to_max_sim[smi] for smi in unique_smiles]

            # 获取对应的行（每个唯一SMILES取相似度最高的那条记录）
            unique_rows = [self.database_df.iloc[smiles_to_best_idx[smi]]
                           for smi in unique_smiles]

            return unique_rows, unique_sims, unique_smiles

        else:
            # ========== 不去重模式：用于溶剂匹配 ==========
            idxs = np.argsort(similarities)[-k:][::-1]
            rows = [self.database_df.iloc[i] for i in idxs]
            sims = [similarities[i] for i in idxs]

            return rows, sims

    def _load_and_prepare_data(self):
        """加载数据文件并准备用于相似性搜索"""
        self.data_df = pd.read_excel(self.data_file_path, sheet_name='main')

        required_columns = [
            'SMILES',
            'Solvent1', 'Solvent2',
            'Medium1', 'Medium2',
            'me1_c', 'me2_c', 'ligand_c', 'T'
        ]
        missing_cols = [c for c in required_columns if c not in self.data_df.columns]
        if missing_cols:
            raise ValueError(f"数据文件缺少必要的列: {missing_cols}")

        print("正在计算数据库中配体的指纹...")

        database_explicit_bvs = []
        valid_masks = []

        for _, row in self.data_df.iterrows():
            mol = Chem.MolFromSmiles(str(row['SMILES']))
            if mol is None:
                valid_masks.append(False)
                continue

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            database_explicit_bvs.append(fp)
            valid_masks.append(True)

        self.database_df = self.data_df[valid_masks].reset_index(drop=True)
        self.database_explicit_bvs = database_explicit_bvs

        print(f"数据库加载完成，有效分子数：{len(self.database_df)}")

    @torch.no_grad()
    def decode(self, z):
        """批量解码，分批处理避免卡死"""
        if torch.isnan(z).any() or torch.isinf(z).any():
            return [None] * z.size(0), np.zeros(z.size(0), dtype=np.bool_)

        z = z.to(self.device)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        batch_size = z.size(0)
        smiles = []
        valid = []

        # 分批处理，每批最多4个
        chunk_size = 4
        for i in range(0, batch_size, chunk_size):
            chunk_z = z[i:i + chunk_size]

            try:
                # 设置一个总的超时时间
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("Batch decode timeout")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)  # 每批最多10秒

                try:
                    chunk_smiles = self.model.decode_from_latent(chunk_z)

                    for j, smi in enumerate(chunk_smiles):
                        if smi is None:
                            smiles.append(None)
                            valid.append(False)
                        else:
                            mol = Chem.MolFromSmiles(smi)
                            if mol is not None:
                                smiles.append(smi)
                                valid.append(True)
                            else:
                                smiles.append(None)
                                valid.append(False)

                finally:
                    signal.alarm(0)

            except TimeoutError:
                print(f"⚠️ 解码批次超时: samples {i}-{i + chunk_size - 1}")
                # 这一批全部标记为无效
                for _ in range(chunk_size):
                    smiles.append(None)
                    valid.append(False)
            except Exception as e:
                print(f"⚠️ 解码批次错误: {e}")
                for _ in range(chunk_size):
                    smiles.append(None)
                    valid.append(False)

        return smiles, np.array(valid, dtype=np.bool_)

    def _manual_decode(self, z):
        """手动解码潜变量"""
        batch_size = z.size(0)
        smiles = []

        # 分割潜变量
        tree_dim = self.latent_dim // 2
        mol_dim = self.latent_dim // 2

        for i in range(batch_size):
            try:
                tree_vec = z[i, :tree_dim].unsqueeze(0)
                mol_vec = z[i, tree_dim:].unsqueeze(0)

                # 调用模型的decode方法
                smi = self.model.decode(
                    x_tree_vecs=tree_vec,
                    x_mol_vecs=mol_vec,
                    prob_decode=False
                )
                smiles.append(smi)
            except Exception as e:
                print(f"Error decoding sample {i}: {e}")
                smiles.append(None)

        return smiles

    def smiles_to_fingerprint_array(self, smiles):
        """SMILES转指纹数组"""
        mol = Chem.MolFromSmiles(smiles) if smiles and isinstance(smiles, str) else None
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            arr = np.zeros(512, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        return np.zeros(512, dtype=np.float32)

    def prepare_xgb_features(self, ligand_fp_array, row):
        """
        特征结构（严格对齐 XGB 训练）：
        512 ligand
        4×512 (Solvent1, Solvent2, Medium1, Medium2)
        4 numeric (me1_c, me2_c, ligand_c, T)
        """

        env_fps = []
        for col in ['Solvent1', 'Solvent2', 'Medium1', 'Medium2']:
            smi = row[col]
            if pd.isna(smi) or str(smi).strip() == '':
                env_fps.append(np.zeros(512, dtype=np.float32))
            else:
                fp = self.smiles_to_fingerprint_array(str(smi))
                env_fps.append(fp)

        numeric = np.array([
            row['me1_c'] if pd.notna(row['me1_c']) else 0.0,
            row['me2_c'] if pd.notna(row['me2_c']) else 0.0,
            row['ligand_c'] if pd.notna(row['ligand_c']) else 0.0,
            row['T'] if pd.notna(row['T']) else 0.0,
        ], dtype=np.float32)

        features = np.concatenate([
            ligand_fp_array,
            *env_fps,
            numeric
        ])

        return features.reshape(1, -1)

    def compute_reward(self, z, smiles, valid_mask):
        """
        Reward = signal (SF) + exploration (similarity bonus)
        """

        rewards = []
        infos = []

        # ---------- hyper-parameters ----------
        TEMP_SF = 10.0
        W_SIGNAL = 1.0
        W_EXPLORE = 0.3
        SIM_LOW = 0.1
        SIM_PEAK = 0.50
        SIM_SIGMA = 0.25
        SF_REWARD_THRESHOLD = 200.0

        for i, smi in enumerate(smiles):

            # ---------- invalid molecule ----------
            if not valid_mask[i] or (smi and len(smi) <= 3):
                rewards.append(-0.2)
                infos.append({"valid": False})
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rewards.append(-0.2)
                infos.append({"valid": False})
                continue

            # ---------- ligand fingerprint ----------
            lig_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            lig_fp_arr = np.zeros(512, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(lig_fp, lig_fp_arr)

            # ======================================================
            # 关键修改：两套topk
            # ======================================================

            # 1. 用于溶剂匹配的topk（不去重，允许多种实验条件）
            rows_for_solvent, _ = self._find_topk_similar(
                smi,
                k=self.topk_similarity_sol,
                deduplicate=False
            )

            # 2. 用于相似度计算的topk（去重，只看结构多样性）
            _, sims_unique, unique_smiles = self._find_topk_similar(
                smi,
                k=self.topk_similarity,
                deduplicate=True
            )

            if len(rows_for_solvent) == 0:
                rewards.append(-0.1)
                infos.append({"valid": True, "has_neighbor": False})
                continue

            # ======================================================
            # 1 Signal: SF under realistic conditions
            # 使用不去重的rows_for_solvent（保留多种溶剂条件）
            # ======================================================
            sf_list = []
            for row in rows_for_solvent:
                x = self.prepare_xgb_features(lig_fp_arr, row)
                sf = float(self.xgb.predict(x)[0])
                sf = np.exp(sf) - 1
                sf_list.append(sf)

            sf_arr = np.array(sf_list, dtype=np.float32)

            # soft aggregation
            weights = np.exp(sf_arr / TEMP_SF)
            weights /= (weights.sum() + 1e-8)
            soft_sf = float((weights * sf_arr).sum())

            # SF奖励衰减
            if soft_sf > SF_REWARD_THRESHOLD:
                reward_sf = SF_REWARD_THRESHOLD + np.log1p(soft_sf - SF_REWARD_THRESHOLD)
            else:
                reward_sf = soft_sf

            signal = W_SIGNAL * np.log1p(reward_sf)

            # ======================================================
            # 2 Exploration: similarity-shaped bonus
            # 使用去重的sims_unique（基于唯一结构）
            # ======================================================
            sim_arr = np.array(sims_unique, dtype=np.float32)
            sim_mean = float(sim_arr.mean())

            # 记录用于调试的信息
            n_unique_structures = len(sims_unique)
            n_total_neighbors = len(rows_for_solvent)

            # Gaussian-shaped exploration bonus
            explore_bonus = np.exp(
                -0.5 * ((sim_mean - SIM_PEAK) / SIM_SIGMA) ** 2
            )

            # hard reliability gate
            if sim_mean < SIM_LOW:
                explore_bonus *= 0.0

            exploration = W_EXPLORE * explore_bonus

            # ======================================================
            # Final reward
            # ======================================================
            reward = signal + exploration

            rewards.append(reward)
            infos.append({
                "valid": True,
                "soft_sf": soft_sf,
                "reward_sf": reward_sf,
                "signal": signal,
                "explore_bonus": exploration,
                "sim_mean": sim_mean,  # 现在基于去重后的结构
                "n_unique_structures": n_unique_structures,  # 新增：实际有多少独特结构
                "n_total_neighbors": n_total_neighbors,  # 新增：总共有多少邻居（含重复）
                "unique_smiles_sample": unique_smiles[:3] if unique_smiles else [],  # 采样显示
            })

        rewards = np.array(rewards, dtype=np.float32)

        if hasattr(self, "reward_scale"):
            rewards *= self.reward_scale

        return rewards, infos

    def step(self, z):
        smiles, valid_mask = self.decode(z)
        reward, info = self.compute_reward(z, smiles, valid_mask)

        return {
            "reward": reward,
            "done": np.ones(len(smiles), dtype=bool),
            "info": info,
        }
