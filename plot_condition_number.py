import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 配置路径 =================
file_greedy = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/stereo_greedy/greedy_convergence_analysis.csv"
file_random = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/stereo_random/random_convergence_analysis.csv"
output_dir = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/paper_plots"

if not os.path.exists(output_dir): os.makedirs(output_dir)

# 设置风格：使用白色背景带网格，适合学术论文
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

# ================= 数据加载与处理 =================
def load_and_prep_data():
    if not os.path.exists(file_greedy) or not os.path.exists(file_random):
        print("错误：找不到CSV文件！")
        return None

    df_g = pd.read_csv(file_greedy)
    df_r = pd.read_csv(file_random)

    # 统一列名
    rename_dict = {"n_views": "views", "repeat_id": "repeat"}
    df_g.rename(columns=rename_dict, inplace=True)
    df_r.rename(columns=rename_dict, inplace=True)

    # 标记方法
    df_g["Method"] = "Active Strategy (Ours)"
    df_r["Method"] = "Random Baseline"

    # 合并
    df_all = pd.concat([df_g, df_r], ignore_index=True)

    # 选择噪声等级 (Noise = 1.0)
    target_noise = 1.0
    df_plot = df_all[df_all["noise"] == target_noise].copy()
    
    # 数据清洗：去除极端无穷大的值，避免绘图报错
    df_plot.loc[df_plot['cond_num'] > 1e9, 'cond_num'] = 1e9
    
    print(f"正在绘制 Noise Level = {target_noise} 的条件数对比图...")
    return df_plot

# ================= 绘图函数 =================
def plot_paper_figure(df):
    # 调整画布比例 (宽:高 = 4:3 比较适合论文插图)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 自定义颜色 (红 vs 蓝)
    custom_palette = {"Active Strategy (Ours)": "#D62728", "Random Baseline": "#1F77B4"}
    custom_markers = {"Active Strategy (Ours)": "o", "Random Baseline": "X"}
    
    # 绘制 Random (先画，作为背景对比)
    sns.lineplot(
        data=df[df["Method"]=="Random Baseline"], 
        x="views", y="cond_num", hue="Method", style="Method",
        palette=custom_palette, markers=custom_markers,
        linewidth=3, markersize=9, alpha=0.8,
        err_style="band", errorbar=("ci", 95), # 95% 置信区间
        ax=ax
    )
    
    # 绘制 Active (后画，确保由图层上方，不被遮挡)
    sns.lineplot(
        data=df[df["Method"]=="Active Strategy (Ours)"], 
        x="views", y="cond_num", hue="Method", style="Method",
        palette=custom_palette, markers=custom_markers,
        linewidth=4, markersize=10, alpha=1.0,
        err_style="band",
        ax=ax
    )

    # === 关键修正区域 ===
    ax.set_yscale('log')
    
    # 1. 标题和轴标签优化
    ax.set_title(r"Geometric Stability Analysis ($\kappa(J)$)", fontsize=22, fontweight='bold', pad=15)
    ax.set_xlabel("Number of Views ($N$)", fontsize=18, fontweight='bold')
    ax.set_ylabel(r"Condition Number (Log Scale)", fontsize=18, fontweight='bold')
    
    # 2. 坐标轴范围限定（消除左右多余空白）
    ax.set_xlim(0, 20.5)
    
    # 3. 网格线设置 (主网格明显，次网格隐约)
    ax.grid(True, which="major", ls="-", alpha=0.5, color='gray')
    ax.grid(True, which="minor", ls=":", alpha=0.2, color='gray')

    # 4. 【核心修复】图例位置
    # 因为 Random 在最上面，Active 在最下面，中间是空的。
    # loc='center right' 将图例放在右侧中间，完美避开数据。
    handles, labels = ax.get_legend_handles_labels()
    # 重新排序图例，把 Ours 放在第一位
    order = [1, 0] if "Active" in labels[1] else [0, 1]
    
    ax.legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        fontsize=15, 
        loc='center right',          # <--- 移到右侧中间
        bbox_to_anchor=(0.98, 0.65), # <--- 微调垂直高度，0.55 表示在 Y 轴 55% 的高度
        frameon=True, 
        fancybox=True, 
        framealpha=0.95,             # 背景不透明，防止文字混淆
        edgecolor='black'
    )

    # 5. 注释优化
    # 将箭头和文字也移到中间空白区，箭头指向红线
    try:
        # 找个第 6 步的数据点作为箭头指向
        target_x = 6
        target_y = df[(df["Method"]=="Active Strategy (Ours)") & (df["views"]==target_x)]["cond_num"].mean()
        
        ax.annotate(
            'Better Stability', 
            xy=(target_x, target_y * 1.5),     # 箭头尖端 (稍微在红线上方一点)
            xytext=(target_x + 1, target_y * 80), # 文字位置 (向上拉开距离)
            arrowprops=dict(facecolor='#D62728', shrink=0.05, width=3, headwidth=10),
            fontsize=16, color='#D62728', fontweight='bold'
        )
    except:
        pass

    # 边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "Final_Condition_Number.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Done] 优化后的图表已保存: {save_path}")

if __name__ == "__main__":
    df = load_and_prep_data()
    if df is not None:
        plot_paper_figure(df)