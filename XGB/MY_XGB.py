import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from xgboost import XGBRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import shap
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib全局参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False


class XGBFromScratch:
    """从零开始训练的XGBoost模型，包含完整的超参数优化"""

    def __init__(self, file_path: str, params_dir: str = '/home/dc/data_new/MY_XGB/',
                 output_fig_dir: str = '/home/dc/data_new/MY_XGB/figures/'):
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
        """安全转换SMILES到分子对象"""
        return Chem.MolFromSmiles(smiles) if pd.notna(smiles) and isinstance(smiles, str) else None

    def generate_molecular_fingerprints(self, smiles_list):
        """生成Morgan指纹"""
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
        """准备特征矩阵"""
        ligand_fps = self.generate_molecular_fingerprints(sheet['SMILES'])
        solvent_fps = {solvent: self.generate_molecular_fingerprints(sheet[solvent])
                       for solvent in self.solvents}

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
        """创建分层划分"""
        y_binned = pd.qcut(y, q=self.n_bins, labels=False, duplicates='drop')
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=self.random_state)

        for train_index, test_index in sss.split(X, y_binned):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        return X_train, X_test, y_train, y_test

    def log_transform(self, y):
        """对数变换"""
        y_shifted = y - self.y_min_train + 1
        y_transformed = np.log1p(y_shifted)
        return y_transformed

    def split_data_with_stratification(self, X, y, test_size=0.2):
        """划分数据并应用对数变换"""
        X_train, X_test, y_train_original, y_test_original = self.create_stratified_split(
            X, y, test_size=test_size
        )
        self.y_min_train = y_train_original.min()
        y_train_log = self.log_transform(y_train_original)
        y_test_log = self.log_transform(y_test_original)
        return X_train, X_test, y_train_log, y_test_log

    def define_param_space(self):
        """定义超参数搜索空间"""
        param_distributions = {
            'regressor__n_estimators': [200, 250, 300],  # 介于100-200和400之间
            'regressor__max_depth': [4, 5, 6],  # 介于3-4和6之间
            'regressor__learning_rate': [0.03, 0.05, 0.07],  # 保持中等
            'regressor__subsample': [0.7, 0.8],  # 从0.6-0.7放宽
            'regressor__colsample_bytree': [0.5, 0.6, 0.7],  # 从0.4-0.6放宽
            'regressor__min_child_weight': [5, 7, 10],  # 从7-15降低
            'regressor__gamma': [0.1, 0.3, 0.5],  # 从0.2-1.0降低
            'regressor__reg_alpha': [0.5, 1, 3],  # 从1-10降低
            'regressor__reg_lambda': [2, 5, 8]  # 从5-20降低
        }
        return param_distributions

    def create_base_pipeline(self):
        """创建基础pipeline"""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', XGBRegressor(
                random_state=self.random_state,
                tree_method='hist',  # 使用hist方法加速
                verbosity=0,
                n_jobs=-1
            ))
        ])
        return pipeline

    def hyperparameter_optimization(self, X_train, y_train, n_iter=100, cv=5):
        """随机搜索超参数优化"""
        print("\n" + "=" * 60)
        print("Starting Hyperparameter Optimization")
        print("=" * 60)
        print(f"Search iterations: {n_iter}")
        print(f"Cross-validation folds: {cv}")
        print(f"Training data shape: {X_train.shape}")

        # 创建pipeline和参数空间
        pipeline = self.create_base_pipeline()
        param_distributions = self.define_param_space()

        # 创建随机搜索对象
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        # 执行搜索
        print("\nFitting RandomizedSearchCV...")
        random_search.fit(X_train, y_train)

        # 保存最佳参数
        self.best_params = random_search.best_params_

        # 清理参数名（移除'regressor__'前缀）
        clean_params = {k.replace('regressor__', ''): v
                        for k, v in self.best_params.items()}

        print("\n" + "=" * 60)
        print("Optimization Results")
        print("=" * 60)
        print(f"Best CV Score (MSE): {-random_search.best_score_:.4f}")
        print(f"Best Parameters:")
        for param, value in clean_params.items():
            print(f"  {param}: {value}")

        # 保存搜索结果
        cv_results_df = pd.DataFrame(random_search.cv_results_)
        cv_results_path = os.path.join(self.params_dir, 'cv_results_from_scratch.csv')
        cv_results_df.to_csv(cv_results_path, index=False)
        print(f"\n✅ CV results saved to: {cv_results_path}")

        # 保存最佳参数
        params_path = os.path.join(self.params_dir, 'best_params_from_scratch.json')
        with open(params_path, 'w') as f:
            json.dump(clean_params, f, indent=4)
        print(f"✅ Best parameters saved to: {params_path}")

        return random_search.best_estimator_

    def plot_optimization_history(self, cv_results_df):
        """绘制超参数优化历史"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 学习曲线
        ax = axes[0, 0]
        train_scores = -cv_results_df['mean_train_score']
        test_scores = -cv_results_df['mean_test_score']
        ax.plot(train_scores, label='Train MSE', alpha=0.7)
        ax.plot(test_scores, label='CV MSE', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. n_estimators vs score
        ax = axes[0, 1]
        param_col = 'param_regressor__n_estimators'
        if param_col in cv_results_df.columns:
            ax.scatter(cv_results_df[param_col], -cv_results_df['mean_test_score'],
                       alpha=0.6)
            ax.set_xlabel('n_estimators')
            ax.set_ylabel('CV MSE')
            ax.set_title('n_estimators Impact')
            ax.grid(True, alpha=0.3)

        # 3. learning_rate vs score
        ax = axes[1, 0]
        param_col = 'param_regressor__learning_rate'
        if param_col in cv_results_df.columns:
            ax.scatter(cv_results_df[param_col], -cv_results_df['mean_test_score'],
                       alpha=0.6)
            ax.set_xlabel('learning_rate')
            ax.set_ylabel('CV MSE')
            ax.set_title('Learning Rate Impact')
            ax.grid(True, alpha=0.3)

        # 4. max_depth vs score
        ax = axes[1, 1]
        param_col = 'param_regressor__max_depth'
        if param_col in cv_results_df.columns:
            ax.scatter(cv_results_df[param_col], -cv_results_df['mean_test_score'],
                       alpha=0.6)
            ax.set_xlabel('max_depth')
            ax.set_ylabel('CV MSE')
            ax.set_title('Max Depth Impact')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Hyperparameter Optimization Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        fig_path = os.path.join(self.output_fig_dir, 'optimization_history_from_scratch.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Optimization history plot saved to: {fig_path}")

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
        ax.set_title(f'Training Set\nR² = {r2_train:.4f}, MAE = {mae_train:.4f}, RMSE = {rmse_train:.4f}',
                     fontsize=12)
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
        ax.set_title(f'Test Set\nR² = {r2_test:.4f}, MAE = {mae_test:.4f}, RMSE = {rmse_test:.4f}',
                     fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Predicted vs True Values (Log Space)', fontsize=14, fontweight='bold')
        plt.tight_layout()

        fig_path = os.path.join(self.output_fig_dir, f'{run_id}_r2_scatter_log.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ R² scatter plot saved to: {fig_path}")

    def plot_feature_importance(self, pipeline, feature_names, run_id):
        """绘制XGBoost内置的特征重要性"""
        xgb_model = pipeline.named_steps['regressor']

        # 获取特征重要性
        importance_types = ['weight', 'gain', 'cover']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, imp_type in enumerate(importance_types):
            importance = xgb_model.get_booster().get_score(importance_type=imp_type)

            # 转换为完整特征维度
            full_importance = np.zeros(len(feature_names))
            for k, v in importance.items():
                if k.startswith('f'):
                    full_importance[int(k[1:])] = v

            # 取前20个特征
            top_indices = np.argsort(full_importance)[-20:]
            top_importance = full_importance[top_indices]
            top_names = [feature_names[i] for i in top_indices]

            # 简化名称
            simplified_names = []
            for name in top_names:
                if len(name) > 30:
                    simplified_names.append(name[:15] + '...' + name[-10:])
                else:
                    simplified_names.append(name)

            ax = axes[idx]
            y_pos = np.arange(len(top_indices))
            ax.barh(y_pos, top_importance)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(simplified_names, fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title(f'Top 20 Features ({imp_type})')
            ax.grid(True, alpha=0.3)

        plt.suptitle('XGBoost Feature Importance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        fig_path = os.path.join(self.output_fig_dir, f'{run_id}_feature_importance.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature importance plot saved to: {fig_path}")

    def plot_combined_shap_figures(self, pipeline, X_test, feature_names, run_id='from_scratch'):
        """绘制组合SHAP图：热力图 + 分组条形图"""
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

            # 简化特征名称
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

            # 生成热力图
            temp_heatmap_path = os.path.join(self.params_dir, f'temp_shap_heatmap_{run_id}.png')
            shap.summary_plot(shap_values, X_test_sample, feature_names=simplified_names,
                              plot_type="dot", show=False, max_display=20,
                              plot_size=(7, 6))
            plt.title('SHAP Value Heatmap (Top 20 Features)', fontsize=12, fontweight='bold')
            plt.savefig(temp_heatmap_path, dpi=300, bbox_inches='tight')

            import matplotlib.font_manager as fm
            current_font = plt.rcParams['font.family'][0]
            current_font_size = plt.rcParams['font.size']
            plt.close()

            # 生成分组条形图
            feature_groups = {
                'Ligand': {'indices': [i for i, name in enumerate(feature_names)
                                       if name.startswith('Ligand_FP_')]},
                'Solvent1': {'indices': [i for i, name in enumerate(feature_names)
                                         if name.startswith('Solvent1_FP_')]},
                'Solvent2': {'indices': [i for i, name in enumerate(feature_names)
                                         if name.startswith('Solvent2_FP_')]},
                'Solvent3': {'indices': [i for i, name in enumerate(feature_names)
                                         if name.startswith('Solvent3_FP_')]},
                'Solvent4': {'indices': [i for i, name in enumerate(feature_names)
                                         if name.startswith('Solvent4_FP_')]},
                'S3_Conc': {'indices': [i for i, name in enumerate(feature_names)
                                        if name == 'Solvent3_Conc']},
                'S4_Conc': {'indices': [i for i, name in enumerate(feature_names)
                                        if name == 'Solvent4_Conc']},
                'Lig_Conc': {'indices': [i for i, name in enumerate(feature_names)
                                         if name == 'Ligand_C']},
                'Temp': {'indices': [i for i, name in enumerate(feature_names)
                                     if name == 'Temperature']}
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

            plt.rcParams['font.family'] = current_font
            plt.rcParams['font.size'] = current_font_size

            fig2, ax2 = plt.subplots(figsize=(7, 6))
            uniform_color = '#2E86AB'
            y_pos = np.arange(len(group_names_sorted))
            bars = ax2.barh(y_pos, group_importance_sorted, color=uniform_color,
                            edgecolor='black', alpha=0.8, linewidth=1)

            for i, (bar, val) in enumerate(zip(bars, group_importance_sorted)):
                ax2.text(val + 0.01 * max(group_importance_sorted),
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:.4f}', va='center')

            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(group_names_sorted)
            ax2.invert_yaxis()
            ax2.set_xlabel('Total Mean |SHAP Value|')
            ax2.set_ylabel('')
            ax2.set_title('SHAP Grouped Feature Importance', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)

            plt.subplots_adjust(left=0.25)
            plt.tight_layout()

            temp_bar_path = os.path.join(self.params_dir, f'temp_shap_bar_{run_id}.png')
            plt.savefig(temp_bar_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 拼接两张图
            from PIL import Image

            heatmap_img = Image.open(temp_heatmap_path)
            bar_img = Image.open(temp_bar_path)

            total_width = heatmap_img.width + bar_img.width
            max_height = max(heatmap_img.height, bar_img.height)
            combined_img = Image.new('RGB', (total_width, max_height), color='white')

            combined_img.paste(heatmap_img, (0, 0))
            combined_img.paste(bar_img, (heatmap_img.width, 0))

            fig_path = os.path.join(self.output_fig_dir, f'{run_id}_combined_shap.png')
            combined_img.save(fig_path, dpi=(300, 300))
            print(f"✅ Combined SHAP figure saved to: {fig_path}")

            # 删除临时文件
            os.remove(temp_heatmap_path)
            os.remove(temp_bar_path)

            # 保存SHAP数据
            shap_data = {
                'shap_values': shap_values.tolist(),
                'feature_names': feature_names,
                'X_test_sample': X_test_sample.tolist()
            }
            shap_path = os.path.join(self.params_dir, f'shap_values_{run_id}.pkl')
            with open(shap_path, 'wb') as f:
                pickle.dump(shap_data, f)

            # 保存分组重要性
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

    def train_from_scratch(self, run_id='from_scratch', save_pipeline=True,
                           do_shap=True, n_iter=100, cv=5):
        """主函数：从零开始训练XGBoost模型"""
        print("=" * 60)
        print("Starting XGBoost Training FROM SCRATCH")
        print("=" * 60)

        # 1. 读取数据
        print("\n" + "=" * 60)
        print("Loading Data")
        print("=" * 60)

        try:
            sheet1 = pd.read_excel(self.file_path, sheet_name='Sheet4')
            print(f"✅ Data loaded: {self.file_path}")
            print(f"Rows: {len(sheet1)}")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            return None

        # 2. 准备特征
        X, y_original = self.prepare_features(sheet1)
        print(f"✅ Features prepared: {X.shape}")

        # 3. 划分数据
        X_train, X_test, y_train, y_test = self.split_data_with_stratification(
            X, y_original, test_size=0.2
        )
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")

        # 4. 超参数优化
        best_pipeline = self.hyperparameter_optimization(X_train, y_train, n_iter=n_iter, cv=cv)

        # 5. 绘制优化历史
        cv_results_df = pd.read_csv(os.path.join(self.params_dir, 'cv_results_from_scratch.csv'))
        self.plot_optimization_history(cv_results_df)

        # 6. 用最佳参数训练最终模型
        print("\n" + "=" * 60)
        print("Training Final Model with Best Parameters")
        print("=" * 60)

        final_pipeline = best_pipeline
        final_pipeline.fit(X_train, y_train)
        print("✅ Final model training complete")

        # 7. 预测并计算指标
        y_train_pred = final_pipeline.predict(X_train)
        y_test_pred = final_pipeline.predict(X_test)

        print("\n" + "=" * 60)
        print("Final Model Performance (Log Space)")
        print("=" * 60)
        print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
        print(f"Test R²: {r2_score(y_test, y_test_pred):.4f}")
        print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
        print(f"Test MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")
        print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
        print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")

        # 8. 生成可视化
        self.plot_r2_scatter(y_train, y_train_pred, y_test, y_test_pred, run_id)

        # 9. 特征重要性
        feature_names = [f'Ligand_FP_{i}' for i in range(512)] + \
                        [f'Solvent1_FP_{i}' for i in range(512)] + \
                        [f'Solvent2_FP_{i}' for i in range(512)] + \
                        [f'Solvent3_FP_{i}' for i in range(512)] + \
                        [f'Solvent4_FP_{i}' for i in range(512)] + \
                        ['Solvent3_Conc', 'Solvent4_Conc', 'Ligand_C', 'Temperature']

        self.plot_feature_importance(final_pipeline, feature_names, run_id)

        # 10. SHAP分析
        if do_shap:
            shap_values, group_importance = self.plot_combined_shap_figures(
                final_pipeline, X_test, feature_names, run_id
            )

        # 11. 保存模型
        if save_pipeline:
            pipeline_path = os.path.join(self.params_dir, f'trained_pipeline_{run_id}.pkl')
            with open(pipeline_path, 'wb') as f:
                pickle.dump(final_pipeline, f)
            print(f"\n✅ Pipeline saved to: {pipeline_path}")

        print("\n" + "=" * 60)
        print("Training Complete! Generated:")
        print("  1. Hyperparameter optimization history")
        print("  2. R² Scatter Plot (log space)")
        print("  3. Feature importance plots")
        print("  4. SHAP analysis (heatmap + grouped importance)")
        print("=" * 60)

        return {
            'pipeline': final_pipeline,
            'best_params': self.best_params,
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
    """主函数"""
    file_path = '/home/dc/data_new/MY_XGB/sf.xlsx'
    params_dir = '/home/dc/data_new/MY_XGB/'
    output_fig_dir = '/home/dc/data_new/MY_XGB/figures/'

    # 创建从零开始训练的实例
    trainer = XGBFromScratch(file_path, params_dir, output_fig_dir)

    # 开始训练
    result = trainer.train_from_scratch(
        run_id='from_scratch',
        save_pipeline=True,
        do_shap=True,
        n_iter=50,  # 可以根据需要调整搜索迭代次数
        cv=5  # 交叉验证折数
    )

    if result:
        print("\n✅ Model trained successfully from scratch!")
        print(f"Best Test R² (log): {result['metrics']['test_r2']:.4f}")
        print("\nBest Parameters Found:")
        for param, value in result['best_params'].items():
            print(f"  {param}: {value}")
    else:
        print("❌ Model training failed")


if __name__ == '__main__':
    main()