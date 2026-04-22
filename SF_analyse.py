import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd


def plot_sf_distribution(df, save_path=None, run_id='sf_distribution'):
    try:
        # 提取SF数据
        sf_values = df['SF'].dropna().values

        print(f"SF值统计:")
        print(f"  样本数: {len(sf_values)}")
        print(f"  最小值: {np.min(sf_values):.2f}")
        print(f"  最大值: {np.max(sf_values):.2f}")
        print(f"  均值: {np.mean(sf_values):.2f}")
        print(f"  中位数: {np.median(sf_values):.2f}")
        print(f"  标准差: {np.std(sf_values):.2f}")

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 6))

        # 方法1：使用直方图的密度曲线（推荐）
        n_bins = 50
        counts, bins, patches = ax.hist(sf_values, bins=n_bins, density=True,
                                        alpha=0.3, color='steelblue', edgecolor='black', linewidth=0.5)

        # 计算bin中心点
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # 绘制折线图
        ax.plot(bin_centers, counts, 'o-', color='darkred', linewidth=2, markersize=4, label='Density')

        # 添加核密度估计曲线（更平滑）
        kde = stats.gaussian_kde(sf_values)
        x_range = np.linspace(np.min(sf_values), np.max(sf_values), 200)
        ax.plot(x_range, kde(x_range), '--', color='green', linewidth=2, alpha=0.7, label='KDE (smoothed)')

        # 设置标签和标题
        ax.set_xlabel('Soft SF', fontsize=14, fontweight='semibold')
        ax.set_ylabel('Density', fontsize=14, fontweight='semibold')
        ax.set_title('SF Distribution', fontsize=16, fontweight='bold')

        # 添加统计信息文本框
        stats_text = f'Mean: {np.mean(sf_values):.2f}\nMedian: {np.median(sf_values):.2f}\nStd: {np.std(sf_values):.2f}\nN: {len(sf_values)}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10, family='monospace')

        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        # 保存图像
        if save_path is None:
            save_path = f'/home/dc/data_new/ppo_results/ppo/{run_id}.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ SF分布图已保存到: {save_path}")
        plt.show()

        return fig, ax

    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")
        return None, None


# 使用示例（如果只需要简单的折线图）
def plot_sf_simple_line(df, save_path=None, run_id='sf_distribution_simple'):
    """
    绘制简单的SF分布折线图（基于直方图计数）
    """
    try:
        sf_values = df['soft_sf'].dropna().values

        # 创建直方图统计
        n_bins = 50
        counts, bins = np.histogram(sf_values, bins=n_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # 绘制折线图
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(bin_centers, counts, 'o-', color='steelblue', linewidth=2, markersize=4)

        # 设置标签和标题
        ax.set_xlabel('Soft SF', fontsize=14, fontweight='semibold')
        ax.set_ylabel('Frequency (Number of Molecules)', fontsize=14, fontweight='semibold')
        ax.set_title('SF Distribution - Frequency Plot', fontsize=16, fontweight='bold')

        # 填充线下面积
        ax.fill_between(bin_centers, counts, alpha=0.3, color='steelblue')

        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        # 保存图像
        if save_path is None:
            save_path = f'/home/dc/data_new/ppo_results/ppo/{run_id}.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 简单SF分布图已保存到: {save_path}")
        plt.show()

        return fig, ax

    except Exception as e:
        print(f"⚠️ 绘图失败: {e}")
        return None, None


# 使用示例
if __name__ == "__main__":
    # 读取数据
    file_path = '/home/dc/data_new/S.xlsx'
    df = pd.read_excel(file_path)

    # 过滤SF>200的数据
    #df = df[df['soft_sf'] > 200]

    # 绘制详细分布图（包含密度曲线和KDE）
    plot_sf_distribution(df,
                         save_path='/home/dc/data_new/ppo_results/ppo/sf_distribution.png',
                         run_id='sf_distribution')

    # 或者绘制简单折线图
    # plot_sf_simple_line(df_filtered,
    #                    save_path='/home/dc/data_new/ppo_results/ppo/sf_distribution_simple.png',
    #                    run_id='sf_distribution_simple')