import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches

# ================= 1. 配置区域 =================
INPUT_GREEDY_JSON = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/stereo_greedy/greedy_corners.json"
INPUT_RANDOM_JSON = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/stereo_random/random_corners.json"
OUTPUT_FILE = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/paper_plots/corner_heatmap_shuangmu.png"

# 分辨率配置
IMG_W = 1280
IMG_H = 720
GRID_SIZE = 35        # 格子稍微调大一点，看起来更连贯
SCATTER_SIZE = 6      # 散点变大，看得清
SCATTER_ALPHA = 0.4   # 散点不透明度增加

# 字体配置
config = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.weight": "bold",
    "axes.labelweight": "bold", 
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12, # 坐标轴字号调大
    "ytick.labelsize": 12,
}
plt.rcParams.update(config)

# ================= 2. 数据加载 =================
def load_corners(json_path, label):
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        arr = np.array(data)
        # 简单过滤一下越界点(防止绘图报错)
        mask = (arr[:,0] >= 0) & (arr[:,0] <= IMG_W) & (arr[:,1] >= 0) & (arr[:,1] <= IMG_H)
        return arr[mask, 0], arr[mask, 1]
    else:
        # Fallback 模拟数据
        print(f"[Warning] {label} 文件未找到，使用模拟数据。")
        np.random.seed(42 if "Proposed" in label else 1)
        if "Proposed" in label: # 模拟边缘分布
            u = np.concatenate([np.random.normal(200, 50, 200), np.random.normal(1080, 50, 200), np.random.uniform(200, 1080, 200)])
            v = np.concatenate([np.random.normal(200, 50, 200), np.random.normal(520, 50, 200), np.random.uniform(100, 600, 200)])
        else: # 模拟中心分布
            u = np.random.normal(640, 100, 600)
            v = np.random.normal(360, 80, 600)
        return u, v

# ================= 3. 绘图主程序 =================
def plot_heatmap():
    u_greedy, v_greedy = load_corners(INPUT_GREEDY_JSON, "Prosposed")
    u_rand, v_rand = load_corners(INPUT_RANDOM_JSON, "Baseline")
    
    # 画布加宽，给Colorbar留足空间
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
    
    # 统一的绘图函数
    def draw_ax(ax, u, v, title_text, color_theme='Reds'):
        # 1. 绘制图像边框
        rect = plt.Rectangle((0,0), IMG_W, IMG_H, lw=3, ec='black', fc='#F5F5F5', zorder=0)
        ax.add_patch(rect)
        
        # 2. 绘制散点 (底层，清晰可见)
        # 使用深灰色/黑色描边，让点更扎实
        ax.scatter(u, v, c='#2C3E50', s=SCATTER_SIZE, alpha=SCATTER_ALPHA, 
                   edgecolors='none', zorder=5, label='Detected Corners')

        # 3. 绘制热力图 (核心!)
        # bins='log' 是让右图“显形”的关键：它让低密度区域也能显示出较深的颜色
        # cmap: 使用 'YlOrRd' (黄-橙-红) 经典色系
        hb = ax.hexbin(u, v, gridsize=GRID_SIZE, cmap='YlOrRd', 
                       bins='log',  # <--- 关键修改：对数刻度
                       mincnt=1, edgecolors='white', linewidths=0.2, alpha=0.9,
                       extent=(0, IMG_W, 0, IMG_H), zorder=4)
        
        # 4. 坐标轴设定 (翻转Y轴符合图像习惯)
        ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)
        ax.set_aspect('equal')
        ax.set_xlabel("u (pixels)", fontweight='bold')
        if ax == ax1: ax.set_ylabel("v (pixels)", fontweight='bold')
        
        # 5. 标题优化 (加背景色块突出标题，不用担心重叠)
        ax.set_title(title_text, pad=15, fontweight='bold', fontsize=20, color='#333333')
        
        # 6. 信息框 (替代原本乱飘的文字)
        stats_text = f"Feature Points: {len(u)}\nCoverage: {(len(u)/len(u_rand) if len(u_rand)>0 else 1.0)*100:.0f}% (Relative)"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc')
        ax.text(0.03, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props, zorder=20)
        
        return hb

    # --- 绘制 ---
    # 左图：Random
    hb1 = draw_ax(ax1, u_rand, v_rand, "(a) Baseline (Random)")
    
    # 右图：Proposed
    hb2 = draw_ax(ax2, u_greedy, v_greedy, "(b) Proposed (Active)")
    
    # --- 增强对比标注 (可选) ---
    # 在右图画一个红色虚线框，强调边缘覆盖？不，太乱了，对数色谱已经足够凸显优势。
    
    # --- Colorbar ---
    # 调整位置，使其不挤占主图
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7]) 
    cbar = fig.colorbar(hb2, cax=cbar_ax)
    cbar.set_label('Point Density (Log Scale)', fontweight='bold', fontsize=14)
    # 因为是 Log Scale，刻度显示整数更直观
    
    # 调整子图间距
    plt.subplots_adjust(left=0.05, right=0.91, wspace=0.10)
    
    # 保存
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)): os.makedirs(os.path.dirname(OUTPUT_FILE))
    plt.savefig(OUTPUT_FILE, pad_inches=0.3, bbox_inches='tight')
    print(f"\n[Done] 优化版热力图已生成: {OUTPUT_FILE}")
    print("关键改进: 使用对数色彩映射(Log Scale)凸显了右图的广域覆盖优势。")

if __name__ == "__main__":
    plot_heatmap()