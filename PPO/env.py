import torch
import numpy as np
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


class CJTVAE_XGB_Env:
    """
    Reinforcement learning environment for ligand optimization.

    This environment wraps a pre-trained JT-VAE generator and an XGBoost
    surrogate model to provide reward signals for PPO-based molecular
    optimization in latent space. Rewards are computed based on predicted
    Am/Eu separation factors and structural similarity to known ligands.
    """

    def __init__(
            self,
            cjtvae_model,
            xgb_model,
            device,
            data_file_path='/home/dc/data_new/S.xlsx',
            latent_dim=48,
            z_prior=None,
            # Reward control parameters
            similarity_threshold=0.3,
            similarity_gate_k=0.05,
            topk_similarity=2,
            topk_similarity_sol=5,
            reward_scale=1.0,
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

        # Load reference database for similarity computation
        self.data_file_path = data_file_path
        self._load_and_prepare_data()

    def _similarity_gate(self, sim):
        """
        Smooth sigmoid gate mapping similarity to (0, 1).

        Provides a continuous transition between low and high similarity
        regimes for the exploration reward.
        """
        return 1.0 / (1.0 + np.exp(-(sim - self.similarity_threshold) / self.similarity_gate_k))

    def _find_topk_similar(self, query_smiles, k=5, deduplicate=True):
        """
        Find the top-k most similar molecules in the reference database.

        Args:
            query_smiles: SMILES string of the query molecule.
            k: Number of similar molecules to return.
            deduplicate: If True, return unique SMILES (for similarity scoring);
                         if False, allow duplicates (for solvent condition matching).

        Returns:
            rows: List of DataFrame rows for matched molecules.
            sims: List of Tanimoto similarity scores.
            unique_smiles: List of unique SMILES (only when deduplicate=True).
        """
        mol = Chem.MolFromSmiles(query_smiles)
        if mol is None:
            if deduplicate:
                return [], [], []
            else:
                return [], []

        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)

        # Compute similarities to all database molecules
        similarities = DataStructs.BulkTanimotoSimilarity(
            query_fp, self.database_explicit_bvs
        )

        if len(similarities) == 0:
            if deduplicate:
                return [], [], []
            else:
                return [], []

        if deduplicate:
            # Deduplication mode: keep only the best match per unique SMILES
            smiles_to_max_sim = {}
            smiles_to_best_idx = {}

            for idx, sim in enumerate(similarities):
                smi = self.database_df.iloc[idx]['SMILES']
                if smi not in smiles_to_max_sim or sim > smiles_to_max_sim[smi]:
                    smiles_to_max_sim[smi] = sim
                    smiles_to_best_idx[smi] = idx

            # Sort by similarity and take top-k unique SMILES
            unique_smiles = sorted(smiles_to_max_sim.keys(),
                                   key=lambda x: smiles_to_max_sim[x],
                                   reverse=True)[:k]

            unique_sims = [smiles_to_max_sim[smi] for smi in unique_smiles]

            unique_rows = [self.database_df.iloc[smiles_to_best_idx[smi]]
                           for smi in unique_smiles]

            return unique_rows, unique_sims, unique_smiles

        else:
            # Non-deduplication mode: allow multiple entries with same SMILES
            idxs = np.argsort(similarities)[-k:][::-1]
            rows = [self.database_df.iloc[i] for i in idxs]
            sims = [similarities[i] for i in idxs]

            return rows, sims

    def _load_and_prepare_data(self):
        """Load the reference ligand database and precompute fingerprints for similarity search."""
        self.data_df = pd.read_excel(self.data_file_path, sheet_name='main')

        required_columns = [
            'SMILES',
            'Solvent1', 'Solvent2',
            'Medium1', 'Medium2',
            'me1_c', 'me2_c', 'ligand_c', 'T'
        ]
        missing_cols = [c for c in required_columns if c not in self.data_df.columns]
        if missing_cols:
            raise ValueError(f"Data file missing required columns: {missing_cols}")

        print("Computing fingerprints for database ligands...")

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

        print(f"Database loaded: {len(self.database_df)} valid molecules")

    @torch.no_grad()
    def decode(self, z):
        """
        Batch decode latent vectors to SMILES strings.

        Processing is done in small chunks to prevent hanging on
        difficult-to-decode latent points.

        Args:
            z: Latent vectors of shape [batch_size, latent_dim].

        Returns:
            smiles: List of SMILES strings (None for failed decodings).
            valid: Boolean array indicating which decodings succeeded.
        """
        if torch.isnan(z).any() or torch.isinf(z).any():
            return [None] * z.size(0), np.zeros(z.size(0), dtype=np.bool_)

        z = z.to(self.device)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        batch_size = z.size(0)
        smiles = []
        valid = []

        # Process in chunks of 4 to limit decoding time per batch
        chunk_size = 4
        for i in range(0, batch_size, chunk_size):
            chunk_z = z[i:i + chunk_size]

            try:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("Batch decode timeout")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)  # Maximum 10 seconds per chunk

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
                print(f"Warning: decode timeout for samples {i}-{i + chunk_size - 1}")
                for _ in range(chunk_size):
                    smiles.append(None)
                    valid.append(False)
            except Exception as e:
                print(f"Warning: decode error: {e}")
                for _ in range(chunk_size):
                    smiles.append(None)
                    valid.append(False)

        return smiles, np.array(valid, dtype=np.bool_)

    def _manual_decode(self, z):
        """
        Manual decoding by splitting the latent vector into tree and molecular components.

        This is a fallback method that directly calls the model's decode function
        with separated tree and molecular latent vectors.
        """
        batch_size = z.size(0)
        smiles = []

        tree_dim = self.latent_dim // 2
        mol_dim = self.latent_dim // 2

        for i in range(batch_size):
            try:
                tree_vec = z[i, :tree_dim].unsqueeze(0)
                mol_vec = z[i, tree_dim:].unsqueeze(0)

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
        """Convert a SMILES string to a 512-bit Morgan fingerprint array."""
        mol = Chem.MolFromSmiles(smiles) if smiles and isinstance(smiles, str) else None
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            arr = np.zeros(512, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        return np.zeros(512, dtype=np.float32)

    def prepare_xgb_features(self, ligand_fp_array, row):
        """
        Build the feature vector for XGBoost prediction.

        Feature structure (aligned with XGBoost training):
          512-d ligand fingerprint
          4 × 512-d solvent fingerprints (Solvent1, Solvent2, Medium1, Medium2)
          4 numeric features (me1_c, me2_c, ligand_c, T)
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
        Compute the composite reward for each generated molecule.

        Reward = signal (predicted SF) + exploration (similarity-based bonus).

        The signal reward uses a soft aggregation over the top-k most similar
        reference molecules with their experimental conditions. The exploration
        bonus encourages generation of molecules with moderate structural novelty.
        """
        rewards = []
        infos = []

        # Reward hyperparameters
        TEMP_SF = 10.0
        W_SIGNAL = 1.0
        W_EXPLORE = 0.3
        SIM_LOW = 0.1
        SIM_PEAK = 0.50
        SIM_SIGMA = 0.25
        SF_REWARD_THRESHOLD = 200.0

        for i, smi in enumerate(smiles):

            # Invalid molecule: assign negative penalty
            if not valid_mask[i] or (smi and len(smi) <= 3):
                rewards.append(-0.2)
                infos.append({"valid": False})
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rewards.append(-0.2)
                infos.append({"valid": False})
                continue

            # Compute ligand fingerprint
            lig_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            lig_fp_arr = np.zeros(512, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(lig_fp, lig_fp_arr)

            # Two separate top-k queries:
            # 1. For solvent condition matching (no dedup, preserves diverse conditions)
            rows_for_solvent, _ = self._find_topk_similar(
                smi,
                k=self.topk_similarity_sol,
                deduplicate=False
            )

            # 2. For structural similarity scoring (dedup, focuses on unique structures)
            _, sims_unique, unique_smiles = self._find_topk_similar(
                smi,
                k=self.topk_similarity,
                deduplicate=True
            )

            if len(rows_for_solvent) == 0:
                rewards.append(-0.1)
                infos.append({"valid": True, "has_neighbor": False})
                continue

            # ---- Signal reward: predicted SF under realistic conditions ----
            sf_list = []
            for row in rows_for_solvent:
                x = self.prepare_xgb_features(lig_fp_arr, row)
                sf = float(self.xgb.predict(x)[0])
                sf = np.exp(sf) - 1  # Inverse log-transform
                sf_list.append(sf)

            sf_arr = np.array(sf_list, dtype=np.float32)

            # Soft aggregation: weighted average with temperature scaling
            weights = np.exp(sf_arr / TEMP_SF)
            weights /= (weights.sum() + 1e-8)
            soft_sf = float((weights * sf_arr).sum())

            # Apply logarithmic decay for very high SF values
            if soft_sf > SF_REWARD_THRESHOLD:
                reward_sf = SF_REWARD_THRESHOLD + np.log1p(soft_sf - SF_REWARD_THRESHOLD)
            else:
                reward_sf = soft_sf

            signal = W_SIGNAL * np.log1p(reward_sf)

            # ---- Exploration bonus: similarity-shaped reward ----
            sim_arr = np.array(sims_unique, dtype=np.float32)
            sim_mean = float(sim_arr.mean())

            n_unique_structures = len(sims_unique)
            n_total_neighbors = len(rows_for_solvent)

            # Gaussian-shaped bonus centered at SIM_PEAK
            explore_bonus = np.exp(
                -0.5 * ((sim_mean - SIM_PEAK) / SIM_SIGMA) ** 2
            )

            # Hard reliability gate: zero bonus below SIM_LOW
            if sim_mean < SIM_LOW:
                explore_bonus *= 0.0

            exploration = W_EXPLORE * explore_bonus

            # ---- Final composite reward ----
            reward = signal + exploration

            rewards.append(reward)
            infos.append({
                "valid": True,
                "soft_sf": soft_sf,
                "reward_sf": reward_sf,
                "signal": signal,
                "explore_bonus": exploration,
                "sim_mean": sim_mean,
                "n_unique_structures": n_unique_structures,
                "n_total_neighbors": n_total_neighbors,
                "unique_smiles_sample": unique_smiles[:3] if unique_smiles else [],
            })

        rewards = np.array(rewards, dtype=np.float32)

        if hasattr(self, "reward_scale"):
            rewards *= self.reward_scale

        return rewards, infos

    def step(self, z):
        """
        Execute one environment step: decode latent vectors and compute rewards.

        Args:
            z: Latent vectors of shape [batch_size, latent_dim].

        Returns:
            A dictionary with 'reward', 'done' (always True for single-step),
            and 'info' containing per-molecule diagnostics.
        """
        smiles, valid_mask = self.decode(z)
        reward, info = self.compute_reward(z, smiles, valid_mask)

        return {
            "reward": reward,
            "done": np.ones(len(smiles), dtype=bool),
            "info": info,
        }
