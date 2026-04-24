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


# Suppress RDKit logging
RDLogger.DisableLog("rdApp.*")
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ==================== Hyperparameters ====================
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
GAMMA = 0.9
GAE_LAMBDA = 0.95

REWARD_THRESHOLD_ = 5.5
TOPK_REPLAY = 2

SIMILARITY_THRESHOLD = 0.8

# ==================== Paths ====================
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
    """
    Context manager for per-step timeout protection.

    Raises a TimeoutError if the enclosed code block exceeds the specified
    time limit. Also tracks the actual elapsed time for diagnostics.
    """

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
                print(f"\nTimeout: Step {self.step} exceeded limit (elapsed: {elapsed:.1f}s)")

        if exc_type is not None and exc_type == TimeoutError:
            return True

        return False


class EliteReplayBuffer:
    """
    Experience replay buffer that retains high-reward trajectories.

    Maintains a fixed-size buffer of the best trajectories, with scaffold-based
    deduplication to encourage chemical diversity. Supports sampling of initial
    latent states from historically successful trajectories with added noise.
    """

    def __init__(self, max_size=200, noise_std=0.05, similarity_threshold=0.8):
        self.max_size = max_size
        self.noise_std = noise_std
        self.similarity_threshold = similarity_threshold
        self.buffer = []
        self.scores = []
        self.scaffolds = []

    def _get_scaffold(self, smiles):
        """Extract the Murcko scaffold from a SMILES string."""
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
        """
        Check whether a scaffold is already present in the buffer.

        Two scaffolds are considered duplicates if their Tanimoto similarity
        exceeds the configured threshold.
        """
        if new_scaffold is None:
            return False

        for existing_scaffold in self.scaffolds:
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
                    return True
            except:
                continue
        return False

    def add_trajectory(self, states, actions, rewards, final_reward, final_smiles):
        """
        Add a trajectory to the buffer with scaffold-based deduplication.

        If a similar scaffold already exists, the new trajectory only replaces
        the old one if it achieves a higher reward.
        """
        new_scaffold = self._get_scaffold(final_smiles)
        if new_scaffold is not None and self._is_duplicate_scaffold(new_scaffold):
            # Check if the new trajectory outperforms the existing one with the same scaffold
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
                            # Replace the old entry
                            self.buffer.pop(i)
                            self.scores.pop(i)
                            self.scaffolds.pop(i)
                            break
                        else:
                            # Reward is not better; discard
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

        # Sort by final_reward and keep only top-k
        if len(self.buffer) > self.max_size:
            sorted_indices = np.argsort(self.scores)[::-1]
            self.buffer = [self.buffer[i] for i in sorted_indices[:self.max_size]]
            self.scores = [self.scores[i] for i in sorted_indices[:self.max_size]]
            self.scaffolds = [self.scaffolds[i] for i in sorted_indices[:self.max_size]]

    def sample_initial_z(self, batch_size, latent_dim, device, sample_type='last_state'):
        """
        Sample initial latent vectors from the buffer.

        Sampling is weighted by trajectory reward to favor high-performing
        regions of the latent space.

        Args:
            sample_type: 'last_state', 'first_state', or 'random_state'.
        """
        if len(self.buffer) == 0:
            return torch.randn(batch_size, latent_dim, device=device)

        scores = np.array(self.scores)
        scores = scores - scores.min() + 1e-8
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
    Collect a multi-step trajectory by rolling out the current policy.

    At each step, the agent predicts a latent displacement, the environment
    decodes the new latent vector and computes rewards.

    Returns:
        states: List of latent states [z0, z1, ..., zT].
        actions: List of actions taken.
        logps: List of action log-probabilities.
        reward_values: List of raw reward arrays.
        smiles_list: List of decoded SMILES per step.
        valid_list: List of validity masks per step.
        infos_list: List of info dictionaries per step.
    """
    states = [z0]
    actions = []
    logps = []
    reward_values = []
    smiles_list = []
    valid_list = []
    infos_list = []

    current_z = z0

    # Evaluate the initial state
    smiles0, valid0 = env.decode(z0)
    reward0, info0 = env.compute_reward(z0, smiles0, valid0)

    for t in range(rollout_steps):
        # Step 1: Actor predicts action
        action, logp = agent.act(current_z)

        # Step 2: Update latent state
        next_z = current_z + action

        print(f'Decoding step {t + 1}...')
        smiles, valid_mask = env.decode(next_z)
        print(f'SMILES: {smiles} — decoding successful')
        reward, infos = env.compute_reward(next_z, smiles, valid_mask)
        print(f'Reward computed: {reward}')

        # Step 3: Store trajectory data
        states.append(next_z)
        actions.append(action)
        logps.append(logp)
        reward_values.append(reward)
        smiles_list.append(smiles)
        valid_list.append(valid_mask)
        infos_list.append(infos)

        # Step 4: Update current state
        current_z = next_z

    return states, actions, logps, reward_values, smiles_list, valid_list, infos_list


def record_batch_multi_step(step, trajectories, smiles_counter, records,
                            record_intermediate=False, record_final_only=True):
    """
    Record molecule-level results from a multi-step trajectory.

    Can record only the final step or all intermediate steps.
    """
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
                    "traj_step": t,
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
                        rec["soft_sf"] = 0.0

                records.append(rec)

        return records


def record_batch_single_step(step, smiles, rewards, valid_mask, infos, smiles_counter, records):
    """Record results for a single-step batch."""
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
    """Save the top-50 unique molecules by reward to an Excel file."""
    top_df = (
        df.sort_values("reward", ascending=False)
        .drop_duplicates("smiles")
        .head(50)
    )
    top_df.to_excel(os.path.join(save_dir, "top_molecules.xlsx"), index=False)


def load_models():
    """Load the pre-trained JT-VAE generator and XGBoost surrogate model."""
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
    """
    Sample initial latent vectors using a mixed strategy.

    A fraction (replay_ratio) is drawn from the elite replay buffer,
    and the remainder is sampled from the standard normal prior.
    """
    z_list = []

    # Replay samples from elite buffer
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
        print("Warning: correcting batch size")
        z0 = torch.randn(batch_size, latent_dim, device=device)

    return z0


def get_replay_ratio(step, total_steps=100):
    """
    Compute the replay ratio using a sigmoid schedule.

    The ratio peaks around 30% of training and decays smoothly
    towards the boundaries, encouraging more random exploration
    early and late in training.
    """
    center = total_steps * 0.3
    width = total_steps * 0.15

    x = (step - center) / width
    ratio = 0.9 / (1 + np.exp(-x))

    return max(0.1, min(0.9, ratio))


def apply_repeat_penalty_final_only(rewards_list, smiles_list, smiles_counter, coef=0.1, similarity_threshold=0.8):
    """
    Apply a repetition penalty to discourage generating the same molecules repeatedly.

    The penalty pool is built from the final-step molecules, but intermediate
    steps that are similar to any final-step molecule also receive a penalty.
    The penalty strength scales with the Tanimoto similarity.

    Args:
        similarity_threshold: Molecules with similarity above this threshold
                              are considered duplicates (default 0.8).
    """
    num_steps = len(rewards_list)
    batch_size = len(rewards_list[0])
    adjusted_list = []

    final_step_idx = num_steps - 1

    def get_similarity(smi1, smi2):
        """Compute Tanimoto similarity between two SMILES strings."""
        if smi1 == smi2:
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

    # Step 1: Build the penalty pool from final-step molecules
    final_smiles_pool = defaultdict(int)
    for i in range(batch_size):
        smi = smiles_list[final_step_idx][i]
        if smi:
            final_smiles_pool[smi] += 1

    # Step 2: Apply penalty to all steps based on the final-step pool
    for t in range(num_steps):
        step_adjusted = []
        step_rewards = rewards_list[t]
        step_smiles = smiles_list[t]

        for i in range(batch_size):
            r = step_rewards[i]
            smi = step_smiles[i]

            if not smi:
                step_adjusted.append(r)
                continue

            # Check similarity to any molecule in the final-step pool
            is_similar = False
            max_similarity = 0.0

            if similarity_threshold < 1.0:
                for final_smi in final_smiles_pool.keys():
                    sim = get_similarity(smi, final_smi)
                    max_similarity = max(max_similarity, sim)
                    if max_similarity >= similarity_threshold:
                        is_similar = True
                        break
            else:
                is_similar = (smi in final_smiles_pool)
                max_similarity = 1.0 if is_similar else 0.0

            if is_similar:
                global_count = smiles_counter.get(smi, 0)

                # Count similar molecules in the current batch
                if similarity_threshold < 1.0:
                    batch_count = 0
                    for final_smi, count in final_smiles_pool.items():
                        if get_similarity(smi, final_smi) >= similarity_threshold:
                            batch_count += count
                else:
                    batch_count = final_smiles_pool[smi]

                total_visits = global_count + batch_count
                # Scale penalty by similarity
                penalty_multiplier = max_similarity if similarity_threshold < 1.0 else 1.0
                penalty = coef * penalty_multiplier * np.log(total_visits + 1)
                adjusted = r - penalty
            else:
                adjusted = r

            step_adjusted.append(adjusted)

        adjusted_list.append(np.array(step_adjusted, dtype=np.float32))

    # Step 3: Update global counter (only for final-step molecules)
    for smi, count in final_smiles_pool.items():
        smiles_counter[smi] += count

    return adjusted_list, smiles_counter


def main():
    """Main PPO training loop for latent-space ligand optimization."""
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
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        rollout_steps=ROLLOUT_STEPS,
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
            # Sample initial latent points (mixed random + replay)
            replay_ratio = get_replay_ratio(step, TOTAL_UPDATES)

            z0 = sample_initial_z_mixed(
                batch_size=BATCH_SIZE,
                latent_dim=LATENT_DIM,
                device=DEVICE,
                replay_buffer=replay_buffer,
                replay_ratio=replay_ratio
            )
            print('Initial states sampled')

            # Multi-step rollout
            states, actions, logps, raw_rewards_list, smiles_list, valid_list, infos_list = \
                collect_rollout(agent, env, z0, ppo.rollout_steps)

            # Apply repetition penalty
            adjusted_rewards_list, smiles_counter = apply_repeat_penalty_final_only(
                raw_rewards_list, smiles_list, smiles_counter, coef=0.2, similarity_threshold=SIMILARITY_THRESHOLD
            )

            # Build terminal flags (only the last step is terminal)
            dones = []
            for t in range(ppo.rollout_steps):
                if t == ppo.rollout_steps - 1:
                    dones.append(torch.ones(BATCH_SIZE, device=DEVICE))
                else:
                    dones.append(torch.zeros(BATCH_SIZE, device=DEVICE))

            # PPO update
            rewards_tensors = [torch.tensor(r, device=DEVICE) for r in adjusted_rewards_list]
            update_metrics = ppo.update_trajectory(
                states=states,
                actions=actions,
                old_logps=logps,
                rewards=rewards_tensors,
                dones=dones,
            )
            print('PPO update completed')

            # Extract final-step data
            final_z = states[-1]
            final_smiles = smiles_list[-1]
            final_valid_mask = valid_list[-1]
            final_rewards = adjusted_rewards_list[-1]
            final_infos = infos_list[-1]

            update_metrics['final_rewards'] = final_rewards
            ppo_reward_means.append(np.mean(final_rewards))

            if update_metrics is not None:
                metrics_history.append(update_metrics)

            # Record trajectory data
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

            # Update elite replay buffer
            for t in range(ppo.rollout_steps):
                infos_t = infos_list[t]
                smiles_t = smiles_list[t]
                rewards_t = adjusted_rewards_list[t]

                candidates = []
                for i in range(BATCH_SIZE):
                    if not valid_list[t][i]:
                        continue

                    total_reward = rewards_t[i]
                    if total_reward >= REWARD_THRESHOLD_:
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

        # Logging
        if step % LOG_INTERVAL == 0:
            valid_count = sum(final_valid_mask)
            valid_ratio = valid_count / BATCH_SIZE
            avg_reward = np.mean(final_rewards)
            max_reward = np.max(final_rewards) if len(final_rewards) > 0 else 0

            sf_values = [info.get("soft_sf", 0) for info in final_infos if info.get("soft_sf") is not None]
            avg_sf = np.mean(sf_values) if sf_values else 0
            max_sf = np.max(sf_values) if sf_values else 0

            print(f"\n[Step {step}]")
            print(f"  Valid molecules: {valid_count}/{BATCH_SIZE} ({valid_ratio:.0%})")
            print(f"  Final reward: mean={avg_reward:.3f}, max={max_reward:.3f}")
            print(f"  Final SF: mean={avg_sf:.3f}, max={max_sf:.3f}")

        # Update best record
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

        # Periodic saving
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

    # Generate baseline comparison plots
    ppo_df = pd.DataFrame(records)

    plot_baseline_results(
        ppo_df,
        ppo_reward_means,
        os.path.join(SAVE_DIR, "ppo_plot.png")
    )


if __name__ == "__main__":
    main()
