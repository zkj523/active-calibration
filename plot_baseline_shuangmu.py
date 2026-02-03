import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ================= 配置路径 =================
# 确保路径指向你存放 result CSV 的目录
ROOT_DIR = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output"
GREEDY_CSV = os.path.join(ROOT_DIR, "stereo_greedy/greedy_convergence_analysis.csv")
RANDOM_CSV = os.path.join(ROOT_DIR, "stereo_random/random_convergence_analysis.csv")
SAVE_DIR = os.path.join(ROOT_DIR, "paper_plots")

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# 设置论文风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'

def load_final_accuracy_data():
    print(f"正在读取数据以分析标定精度...")
    
    if not os.path.exists(GREEDY_CSV) or not os.path.exists(RANDOM_CSV):
        print("[错误] 找不到 CSV 文件，请检查路径。")
        return None

    try:
        df_g = pd.read_csv(GREEDY_CSV)
        df_r = pd.read_csv(RANDOM_CSV)
    except Exception as e:
        print(f"[错误] 读取失败: {e}")
        return None

    # --- 统一列名 (兼容不同版本的代码) ---
    # 你的 greedy_shuangmu.py 现在的输出列名应该是 'base_err' 和 'n_views'
    # 这里做一个映射，防止报错
    name_map = {
        "n_views": "views", 
        "base_err": "Baseline Error",
        "baseline_error": "Baseline Error", # 兼容旧版
        "noise": "Noise Level"
    }
    df_g.rename(columns=name_map, inplace=True)
    df_r.rename(columns=name_map, inplace=True)

    # --- 关键步骤：只提取“最终”结果 ---
    # 我们不关心第1次还是第5次采样的误差，只关心采集完所有图片后(例如N=20)的最终精度
    max_views_g = df_g["views"].max()
    max_views_r = df_r["views"].max()

    print(f"提取最终收敛数据: Greedy(N={max_views_g}), Random(N={max_views_r})")

    df_g_final = df_g[df_g["views"] == max_views_g].copy()
    df_r_final = df_r[df_r["views"] == max_views_r].copy()

    # 标记方法
    df_g_final["Method"] = "Active Strategy (Ours)"
    df_r_final["Method"] = "Random Baseline"

    return pd.concat([df_g_final, df_r_final], ignore_index=True)

def plot_baseline_error(df):
    plt.figure(figsize=(8, 6), dpi=300)

    # 颜色配置：Ours用醒目的红色，Random用冷静的蓝色/灰色
    palette = {"Active Strategy (Ours)": "#D62728", "Random Baseline": "#1F77B4"}
    markers = {"Active Strategy (Ours)": "o", "Random Baseline": "X"}
    
    # 绘制折线图 (带误差棒)
    # err_style='bars' 表示显示标准差/置信区间的竖线
    sns.lineplot(
        data=df,
        x="Noise Level", 
        y="Baseline Error",
        hue="Method", 
        style="Method",
        palette=palette,
        markers=markers,
        linewidth=3,
        markersize=10,
        err_style="bars",  # 显示误差棒
        errorbar=("se", 1) # 显示标准误 (Standard Error)
    )

    # === 美化图表 ===
    plt.yscale('log') 
    plt.title("Calibration Robustness vs. Image Noise", fontweight='bold', fontsize=20, pad=15)
    plt.xlabel("Image Noise Level ($\sigma$)", fontweight='bold', fontsize=16)
    plt.ylabel(r"Stereo Baseline Error $\Delta b$ (m)", fontweight='bold', fontsize=16)
    
    # 如果误差很小，可以添加一条基准线 (例如 1mm 误差限)
    # plt.axhline(y=0.001, color='green', linestyle='--', alpha=0.5, label='Target 1mm')

    # 刻度网格
    plt.grid(True, linestyle='-', alpha=0.6)
    
    # 坐标轴范围微调 (让下面不要顶到0，留点空隙)
    ymin = df["Baseline Error"].min()
    ymax = df["Baseline Error"].max()
    plt.ylim(0, ymax * 1.1) 

    # 图例放在左上角，因为通常误差是随着噪声(X轴)增大而增大(Y轴)，左上角一般是空的
    plt.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, framealpha=0.95, edgecolor='black')

    save_path = os.path.join(SAVE_DIR, "Baseline_Accuracy.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n[Done] 精度对比图已保存: {save_path}")

if __name__ == "__main__":
    df = load_final_accuracy_data()
    if df is not None:
        plot_baseline_error(df)