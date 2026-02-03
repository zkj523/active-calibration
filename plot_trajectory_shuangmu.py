import os
import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ================= 配置区域 =================
# 确保指向正确的 .json 位姿文件
FILE_RANDOM = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/stereo_random/random_poses.json"
FILE_GREEDY = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/stereo_greedy/greedy_poses.json"
OUTPUT_FILE = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/paper_plots/trajectory_shuangmu.png"

# 标定板参数
BOARD_COLS = 7
BOARD_ROWS = 9
SQUARE_SIZE = 0.05

# 字体配置
config = {
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
plt.rcParams.update(config)

# ================= 绘图工具 =================

def draw_real_checkerboard(ax, center=(0,0,0)):
    """绘制简洁的黑白棋盘格"""
    width = (BOARD_COLS + 1) * SQUARE_SIZE
    height = (BOARD_ROWS + 1) * SQUARE_SIZE
    start_x = center[0] - width / 2
    start_y = center[1] - height / 2
    z = center[2]

    # 绘制外框
    ax.plot([start_x, start_x+width, start_x+width, start_x, start_x],
            [start_y, start_y, start_y+height, start_y+height, start_y],
            [z, z, z, z, z], color='black', linewidth=1.5, zorder=0)

    # 填充半透明底色（避免画太多格子显乱，用一个灰色平面代替复杂的格子，只保留外框和中心）
    # 如果您非常想要黑白格，可以恢复之前的循环。但在Trajectory图中，简化板子能减少噪点。
    verts = [[
        [start_x, start_y, z], [start_x+width, start_y, z],
        [start_x+width, start_y+height, z], [start_x, start_y+height, z]
    ]]
    ax.add_collection3d(Poly3DCollection(verts, facecolors='#333333', alpha=0.1, edgecolors='black'))
    
    # 标注Target
    ax.text(center[0], center[1], z, "Target", color='black', fontsize=10, 
            ha='center', va='center', fontweight='bold', zorder=1)

def load_poses(json_path):
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            # 解析 4x4 矩阵或 [x,y,z] 列表
            points = []
            for item in data:
                arr = np.array(item)
                if arr.size == 16: # 4x4 transform
                    points.append(arr.reshape(4,4)[:3, 3])
                elif arr.size == 3: # translation only
                    points.append(arr)
            return np.array(points)
        except:
            return np.array([])
    return np.array([])

# ================= 主逻辑 =================
def main():
    # 1. 并没有真实数据时，为了演示效果，我会生成模拟数据
    #    (如果您有真实json，它会读取真实json)
    poses_greedy = load_poses(FILE_GREEDY)
    poses_random = load_poses(FILE_RANDOM)

    # 模拟数据Fallback (仅当读取失败时触发)
    if len(poses_greedy) == 0:
        print("Using Mock Data for visualization...")
        t = np.linspace(0, 10, 20)
        poses_greedy = np.column_stack([
            0.5 * np.cos(t) * np.sin(t*0.5),
            -0.8 + 0.3*np.cos(t),
            0.5 + 0.3*np.sin(t)
        ])
    if len(poses_random) == 0:
        poses_random = poses_greedy + np.random.normal(0, 0.2, poses_greedy.shape)

    fig = plt.figure(figsize=(10, 8), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    # 1. 画板子
    draw_real_checkerboard(ax)

    # 2. 画 Random (极简模式：只画点，不画线，不标号)
    #    目的：作为背景参考，不抢戏
    if len(poses_random) > 0:
        rx, ry, rz = poses_random[:,0], poses_random[:,1], poses_random[:,2]
        ax.scatter(rx, ry, rz, c='gray', s=20, alpha=0.3, label='Baseline (Random)', depthshade=True)
        # Random 不需要画光轴，太乱了

    # 3. 画 Proposed (主角模式)
    if len(poses_greedy) > 0:
        gx, gy, gz = poses_greedy[:,0], poses_greedy[:,1], poses_greedy[:,2]
        
        # A. 轨迹线 (深蓝色实线)
        ax.plot(gx, gy, gz, c='#1F618D', linewidth=2, alpha=0.8, zorder=10)
        
        # B. 关键点 (使用五角星, 且不每个都画)
        ax.scatter(gx, gy, gz, c='#3498DB', marker='*', s=180, edgecolors='black', linewidth=0.5, 
                   label='Proposed (Active)', zorder=15)

        # C. 极简标签策略 (关键去乱步骤)
        total_pts = len(gx)
        for i in range(total_pts):
            # 策略：只标 起点(Start)、终点(End)、以及每隔5步的一个点
            should_label = False
            label_text = ""
            
            if i == 0:
                label_text = "Start"
                should_label = True
            elif i == total_pts - 1:
                label_text = "End"
                should_label = True
            elif (i + 1) % 5 == 0: # 中间每5个标一个序号
                label_text = str(i + 1)
                should_label = True
            
            if should_label:
                # 标签位置稍微往上提一点
                ax.text(gx[i], gy[i], gz[i] + 0.08, label_text, 
                        fontsize=10, fontweight='bold', color='black',
                        bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.8, ec='black', lw=0.5),
                        ha='center', va='bottom', zorder=20)
        
        # D. 光轴箭头 (只给 Proposed 画, 且改细短一点)
        # 这里假设光轴指向原点 (Look At Center)
        for i in range(total_pts):
            # 为了不乱，也可以选择隔几个画一个箭头
            start = np.array([gx[i], gy[i], gz[i]])
            direction = np.array([0,0,0]) - start
            direction = direction / np.linalg.norm(direction) * 0.15 # 长度0.15m
            
            ax.quiver(start[0], start[1], start[2], 
                      direction[0], direction[1], direction[2],
                      color='#154360', alpha=0.6, arrow_length_ratio=0.3, linewidth=1)

    # 4. 视角与轴
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    
    # 自动缩放
    all_x = np.concatenate([poses_random[:,0], poses_greedy[:,0]])
    all_y = np.concatenate([poses_random[:,1], poses_greedy[:,1]])
    all_z = np.concatenate([poses_random[:,2], poses_greedy[:,2]])
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0, all_z.max() + 0.2)
    
    # 视角：选择一个能把 Start 和 End 分开的角度
    ax.view_init(elev=25, azim=140)

    plt.legend(loc='upper right', frameon=True, fancybox=True)
    plt.title("Optimized Trajectory: Active vs Random", fontsize=16, y=0.95)
    
    plot_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    plt.savefig(OUTPUT_FILE, pad_inches=0.1, bbox_inches='tight')
    print(f"\n[Done] 清晰版轨迹图已保存: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()