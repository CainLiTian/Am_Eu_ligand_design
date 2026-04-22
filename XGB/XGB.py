import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import StratifiedShuffleSplit
import shap
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib全局参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False


class ModelRebuilder:
    def __init__(self, file_path: str, params_dir: str = '/home/dc/data_new/XGB/',
                 output_fig_dir: str = '/home/dc/data_new/XGB/figures/'):
        self.file_path = file_path
        self.params_dir = params_dir
        self.output_fig_dir = output_fig_dir
        self.solvents = ['Solvent1', 'Solvent2', 'Solvent3', 'Solvent4']
        self.best_params = None
        self.y_min_train = None
        self.n_bins = 5
        self.random_state = 42

        os.makedirs(self.params_dir, exist_ok=True)
        os.makedirs(self.output_fig_dir, exist_ok=True)

    @staticmethod
    def safe_smiles_to_mol(smiles: str):
        return Chem.MolFromSmiles(smiles) if pd.notna(smiles) and isinstance(smiles, str) else None

    def generate_molecular_fingerprints(self, smiles_list):
        fingerprints = []
        for smiles in smiles_list:
            mol = self.safe_smiles_to_mol(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
                fingerprint_array = np.zeros(512)
                DataStructs.ConvertToNumpyArray(fp, fingerprint_array)
                fingerprints.append(fingerprint_array)
            else:
                fingerprints.append(np.zeros(512))
        return fingerprints

    def prepare_features(self, sheet, concentration_weight=1):
        ligand_fps = self.generate_molecular_fingerprints(sheet['SMILES'])
        solvent_fps = {solvent: self.generate_molecular_fingerprints(sheet[solvent]) for solvent in self.solvents}

        features = []
        for i in range(len(sheet)):
            row_features = list(ligand_fps[i])
            for solvent in self.solvents:
                row_features.extend(list(solvent_fps[solvent][i]))

            solvent3_conc = sheet.loc[i, 'Solvent3_Conc'] if pd.notna(sheet.loc[i, 'Solvent3_Conc']) else 0
            solvent4_conc = sheet.loc[i, 'Solvent4_Conc'] if pd.notna(sheet.loc[i, 'Solvent4_Conc']) else 0

            row_features.extend([
                solvent3_conc * concentration_weight,
                solvent4_conc * concentration_weight,
                sheet.loc[i, 'ligand_c'] if pd.notna(sheet.loc[i, 'ligand_c']) else 0,
                sheet.loc[i, 'T'] if pd.notna(sheet.loc[i, 'T']) else 0
            ])

            features.append(row_features)

        X = np.array(features)
        y = sheet['SF'].values
        return X, y

    def create_stratified_split(self, X, y, test_size=0.2):
        y_binned = pd.qcut(y, q=self.n_bins, labels=False, duplicates='drop')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_state)

        for train_index, test_index in sss.split(X, y_binned):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test

    def log_transform(self, y):
        y_shifted = y - self.y_min_train + 1
        y_transformed = np.log1p(y_shifted)
        return y_transformed

    def split_data_with_stratification(self, X, y, test_size=0.2):
        X_train, X_test, y_train_original, y_test_original = self.create_stratified_split(
            X, y, test_size=test_size
        )
        self.y_min_train = y_train_original.min()
        y_train_log = self.log_transform(y_train_original)
        y_test_log = self.log_transform(y_test_original)
        return X_train, X_test, y_train_log, y_test_log

    def load_best_params(self, run_id='best_experiment'):
        params_path = os.path.join(self.params_dir, f'best_params_{run_id}.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.best_params = json.load(f)
            print(f"✅ Loaded best params: {params_path}")
            return self.best_params
        else:
            print(f"❌ Params file not found: {params_path}")
            return None

    def create_ml_pipeline(self, **model_params):
        xgb_params = model_params.copy()
        for param in ['random_state', 'tree_method', 'device', 'gpu_id', 'predictor']:
            xgb_params.pop(param, None)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', XGBRegressor(random_state=self.random_state, **xgb_params))
        ])
        return pipeline

    def plot_r2_scatter(self, y_train, y_train_pred, y_test, y_test_pred, run_id):
        """绘制R²散点图（log空间）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 训练集
        ax = axes[0]
        ax.scatter(y_train, y_train_pred, alpha=0.6, edgecolors='w', linewidth=0.5)

        min_val = min(y_train.min(), y_train_pred.min())
        max_val = max(y_train.max(), y_train_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')

        r2_train = r2_score(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

        ax.set_xlabel('True Values (log)', fontsize=12)
        ax.set_ylabel('Predicted Values (log)', fontsize=12)
        ax.set_title(f'Training Set\nR² = {r2_train:.4f}, MAE = {mae_train:.4f}, RMSE = {rmse_train:.4f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 测试集
        ax = axes[1]
        ax.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='w', linewidth=0.5, color='orange')

        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')

        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        ax.set_xlabel('True Values (log)', fontsize=12)
        ax.set_ylabel('Predicted Values (log)', fontsize=12)
        ax.set_title(f'Test Set\nR² = {r2_test:.4f}, MAE = {mae_test:.4f}, RMSE = {rmse_test:.4f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Predicted vs True Values (Log Space)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        fig_path = os.path.join(self.output_fig_dir, f'{run_id}_r2_scatter_log.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ R² scatter plot saved to: {fig_path}")

    def plot_combined_shap_figures(self, pipeline, X_test, feature_names, run_id='rebuilt_experiment'):
        """绘制组合SHAP图：热力图 + 分组条形图（拼在一张图中）"""
        try:
            print("\n" + "=" * 60)
            print("Generating Combined SHAP Figures")
            print("=" * 60)

            # 获取XGBoost模型并计算SHAP值
            xgb_model = pipeline.named_steps['regressor']
            explainer = shap.TreeExplainer(xgb_model)

            n_samples = min(200, X_test.shape[0])
            X_test_sample = X_test[:n_samples]

            print(f"Computing SHAP values (using {n_samples} test samples)...")
            shap_values = explainer.shap_values(X_test_sample)

            if feature_names is None:
                feature_names = [f'F{i}' for i in range(X_test.shape[1])]

            # 简化特征名称（保持原来的简写）
            simplified_names = []
            for name in feature_names:
                if name.startswith('Ligand_FP_'):
                    simplified_names.append(f'Lig_{name.split("_")[-1]}')
                elif name.startswith('Solvent1_FP_'):
                    simplified_names.append(f'S1_{name.split("_")[-1]}')
                elif name.startswith('Solvent2_FP_'):
                    simplified_names.append(f'S2_{name.split("_")[-1]}')
                elif name.startswith('Solvent3_FP_'):
                    simplified_names.append(f'S3_{name.split("_")[-1]}')
                elif name.startswith('Solvent4_FP_'):
                    simplified_names.append(f'S4_{name.split("_")[-1]}')
                elif name == 'Solvent3_Conc':
                    simplified_names.append('S3_Conc')
                elif name == 'Solvent4_Conc':
                    simplified_names.append('S4_Conc')
                elif name == 'Ligand_C':
                    simplified_names.append('Lig_Conc')
                elif name == 'Temperature':
                    simplified_names.append('Temp')
                else:
                    simplified_names.append(name)

            # 先单独生成热力图（使用原来的方式），保存到临时文件
            temp_heatmap_path = os.path.join(self.params_dir, f'temp_shap_heatmap_{run_id}.png')
            shap.summary_plot(shap_values, X_test_sample, feature_names=simplified_names,
                              plot_type="dot", show=False, max_display=20,
                              plot_size=(7, 6))
            plt.title('SHAP Value Heatmap (Top 20 Features)', fontsize=12, fontweight='bold')

            # 保存热力图
            plt.savefig(temp_heatmap_path, dpi=300, bbox_inches='tight')

            # 获取当前matplotlib的字体属性（从热力图中提取）
            import matplotlib.font_manager as fm
            current_font = plt.rcParams['font.family'][0]
            current_font_size = plt.rcParams['font.size']

            plt.close()

            # 生成分组条形图
            feature_groups = {
                'Ligand': {
                    'indices': [i for i, name in enumerate(feature_names) if name.startswith('Ligand_FP_')],
                },
                'Solvent1': {
                    'indices': [i for i, name in enumerate(feature_names) if name.startswith('Solvent1_FP_')],
                },
                'Solvent2': {
                    'indices': [i for i, name in enumerate(feature_names) if name.startswith('Solvent2_FP_')],
                },
                'Solvent3': {
                    'indices': [i for i, name in enumerate(feature_names) if name.startswith('Solvent3_FP_')],
                },
                'Solvent4': {
                    'indices': [i for i, name in enumerate(feature_names) if name.startswith('Solvent4_FP_')],
                },
                'S3_Conc': {
                    'indices': [i for i, name in enumerate(feature_names) if name == 'Solvent3_Conc'],
                },
                'S4_Conc': {
                    'indices': [i for i, name in enumerate(feature_names) if name == 'Solvent4_Conc'],
                },
                'Lig_Conc': {
                    'indices': [i for i, name in enumerate(feature_names) if name == 'Ligand_C'],
                },
                'Temp': {
                    'indices': [i for i, name in enumerate(feature_names) if name == 'Temperature'],
                }
            }

            group_names = []
            group_importance = []

            for group_name, group_info in feature_groups.items():
                indices = group_info['indices']
                if indices:
                    group_shap = shap_values[:, indices]
                    mean_abs_per_feature = np.mean(np.abs(group_shap), axis=0)
                    total_importance = np.sum(mean_abs_per_feature)
                    group_names.append(group_name)
                    group_importance.append(total_importance)

            sorted_indices = np.argsort(group_importance)[::-1]
            group_names_sorted = [group_names[i] for i in sorted_indices]
            group_importance_sorted = [group_importance[i] for i in sorted_indices]

            # 创建分组条形图，使用与热力图相同的字体
            plt.rcParams['font.family'] = current_font
            plt.rcParams['font.size'] = current_font_size

            fig2, ax2 = plt.subplots(figsize=(7, 6))
            uniform_color = '#2E86AB'
            y_pos = np.arange(len(group_names_sorted))
            bars = ax2.barh(y_pos, group_importance_sorted, color=uniform_color,
                            edgecolor='black', alpha=0.8, linewidth=1)

            for i, (bar, val) in enumerate(zip(bars, group_importance_sorted)):
                ax2.text(val + 0.01 * max(group_importance_sorted), bar.get_y() + bar.get_height() / 2,
                         f'{val:.4f}', va='center')

            # y轴标签放在左边，使用与热力图相同的字体
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(group_names_sorted)
            ax2.invert_yaxis()
            ax2.set_xlabel('Total Mean |SHAP Value|')
            ax2.set_ylabel('')
            ax2.set_title('SHAP Grouped Feature Importance', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)

            # 调整左边距，给纵坐标标签留出空间
            plt.subplots_adjust(left=0.25)
            plt.tight_layout()

            temp_bar_path = os.path.join(self.params_dir, f'temp_shap_bar_{run_id}.png')
            plt.savefig(temp_bar_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 拼接两张图
            from PIL import Image

            heatmap_img = Image.open(temp_heatmap_path)
            bar_img = Image.open(temp_bar_path)

            # 创建新图，尺寸与R²散点图一致 (14, 6)
            total_width = heatmap_img.width + bar_img.width
            max_height = max(heatmap_img.height, bar_img.height)
            combined_img = Image.new('RGB', (total_width, max_height), color='white')

            combined_img.paste(heatmap_img, (0, 0))
            combined_img.paste(bar_img, (heatmap_img.width, 0))

            # 保存组合图
            fig_path = os.path.join(self.output_fig_dir, f'{run_id}_combined_shap.png')
            combined_img.save(fig_path, dpi=(300, 300))
            print(f"✅ Combined SHAP figure saved to: {fig_path}")

            # 删除临时文件
            os.remove(temp_heatmap_path)
            os.remove(temp_bar_path)

            # 保存数据
            shap_data = {
                'shap_values': shap_values.tolist(),
                'feature_names': feature_names,
                'X_test_sample': X_test_sample.tolist()
            }
            shap_path = os.path.join(self.params_dir, f'shap_values_{run_id}.pkl')
            with open(shap_path, 'wb') as f:
                pickle.dump(shap_data, f)

            group_importance_df = pd.DataFrame({
                'Feature_Group': group_names,
                'Total_Mean_Abs_SHAP': group_importance
            }).sort_values('Total_Mean_Abs_SHAP', ascending=False)
            csv_path = os.path.join(self.params_dir, f'shap_grouped_importance_{run_id}.csv')
            group_importance_df.to_csv(csv_path, index=False)

            print("\n" + "-" * 40)
            print("Group Importance Ranking:")
            print("-" * 40)
            for i, (name, imp) in enumerate(zip(group_names_sorted, group_importance_sorted), 1):
                print(f"{i}. {name}: {imp:.4f}")

            return shap_values, group_importance_df

        except Exception as e:
            print(f"⚠️ Combined SHAP figures failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def rebuild_and_train(self, run_id='best_experiment', save_pipeline=True, do_shap=True):
        """主函数：重建并训练模型，只生成R²图和SHAP热力图"""
        print("=" * 60)
        print("Starting Model Rebuilding and Training")
        print("=" * 60)

        # 1. 加载最佳参数
        best_params = self.load_best_params(run_id)
        if best_params is None:
            print("❌ Failed to load parameters, exiting")
            return None

        # 2. 读取数据
        print("\n" + "=" * 60)
        print("Loading Data")
        print("=" * 60)

        try:
            sheet1 = pd.read_excel(self.file_path, sheet_name='Sheet3')
            print(f"✅ Data loaded: {self.file_path}")
            print(f"Rows: {len(sheet1)}")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return None

        # 3. 准备特征
        X, y_original = self.prepare_features(sheet1)
        print(f"✅ Features prepared: {X.shape}")

        # 4. 划分数据（得到log变换后的y）
        X_train, X_test, y_train, y_test = self.split_data_with_stratification(X, y_original, test_size=0.2)
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # 5. 创建并训练模型
        print("\n" + "=" * 60)
        print("Training Model")
        print("=" * 60)

        pipeline = self.create_ml_pipeline(**best_params)
        pipeline.fit(X_train, y_train)
        print("✅ Model training complete")

        # 6. 预测并计算指标
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        print("\n" + "=" * 60)
        print("Model Performance (Log Space)")
        print("=" * 60)
        print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
        print(f"Test R²: {r2_score(y_test, y_test_pred):.4f}")
        print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
        print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
        print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
        print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")

        # 7. 生成R²散点图
        self.plot_r2_scatter(y_train, y_train_pred, y_test, y_test_pred, f'{run_id}_rebuilt')

        # 8. 生成SHAP热力图
        if do_shap:
            feature_names = [f'Ligand_FP_{i}' for i in range(512)] + \
                            [f'Solvent1_FP_{i}' for i in range(512)] + \
                            [f'Solvent2_FP_{i}' for i in range(512)] + \
                            [f'Solvent3_FP_{i}' for i in range(512)] + \
                            [f'Solvent4_FP_{i}' for i in range(512)] + \
                            ['Solvent3_Conc', 'Solvent4_Conc', 'Ligand_C', 'Temperature']

            shap_values, group_importance = self.plot_combined_shap_figures(
                pipeline, X_test, feature_names, f'{run_id}_rebuilt'
            )

        # 9. 保存结果
        if save_pipeline:
            pipeline_path = os.path.join(self.params_dir, f'trained_pipeline_{run_id}_rebuilt.pkl')
            with open(pipeline_path, 'wb') as f:
                pickle.dump(pipeline, f)
            print(f"\n✅ Pipeline saved to: {pipeline_path}")

        print("\n" + "=" * 60)
        print("Complete! Generated:")
        print("  1. R² Scatter Plot (log space)")
        print("  2. SHAP Heatmap")
        print("=" * 60)

        return {
            'pipeline': pipeline,
            'metrics': {
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': r2_score(y_test, y_test_pred),
                'train_mae': mean_absolute_error(y_train, y_train_pred),
                'test_mae': mean_absolute_error(y_test, y_test_pred),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
            }
        }


def main():
    file_path = '/home/dc/data_new/XGB/sf.xlsx'
    params_dir = '/home/dc/data_new/XGB/'
    output_fig_dir = '/home/dc/data_new/XGB/figures/'

    rebuilder = ModelRebuilder(file_path, params_dir, output_fig_dir)

    result = rebuilder.rebuild_and_train(
        run_id='best_experiment',
        save_pipeline=True,
        do_shap=True
    )

    if result:
        print("\n✅ Model rebuilt successfully!")
        print(f"Test R² (log): {result['metrics']['test_r2']:.4f}")
    else:
        print("❌ Model rebuilding failed")


if __name__ == '__main__':
    main()