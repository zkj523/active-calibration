import os
import sys
import json  # [新增] 用于保存 JSON 数据

# ================= 1. 渲染后端设置 =================
if 'MUJOCO_GL' in os.environ:
    del os.environ['MUJOCO_GL']
os.environ['MUJOCO_GL'] = 'egl' 

import mujoco
import numpy as np
import cv2
import pandas as pd
import random
import traceback
from scipy.spatial.transform import Rotation as R

# ================= 2. 配置参数 =================
# 请确保这里指向的一致的新复杂场景文件
XML_PATH = "/home/zkj/camera_calibration_sim/zhudongbiaoding/assets/scene2.xml"
OUTPUT_ROOT = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/stereo_random"
IMG_SAVE_DIR = os.path.join(OUTPUT_ROOT, "random_images")

# 关键参数需与 Greedy 保持完全一致
CHECKERBOARD = (7, 9)  
SQUARE_SIZE = 0.05

# 适配 720P 高分辨率
CAM_WIDTH = 1280
CAM_HEIGHT = 720

TARGET_VALID_PAIRS = 20

# 适配新模型的物理参数
GT_BASELINE = 0.10   # 基线 0.10m
GT_FX = 869.3        # 720p 焦距

# 噪声等级列表
NOISE_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]
REPEATS_PER_LEVEL = 10 

if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
if not os.path.exists(IMG_SAVE_DIR): os.makedirs(IMG_SAVE_DIR)

# ================= 3. 核心分析工具：计算条件数 =================
class GeometricAnalyzer:
    def __init__(self):
        # 累积的信息矩阵 H = J.T @ J (针对内参 fx, fy, cx, cy)
        self.H_accumulated = np.zeros((4, 4))
        
    def update_and_compute_cond(self, obj_pts, rvec, tvec):
        """
        计算当前视图的 Jacobian，累加到信息矩阵，并返回条件数
        """
        R_mat, _ = cv2.Rodrigues(rvec)
        pts_cam = (R_mat @ obj_pts.T).T + tvec.T  # (N, 3)
        
        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        z = pts_cam[:, 2]
        
        z[np.abs(z) < 1e-6] = 1e-6
        
        N = len(obj_pts)
        J = np.zeros((2 * N, 4))
        
        # du / dfx = x / z; du / dcx = 1
        J[0::2, 0] = x / z
        J[0::2, 2] = 1.0
        # dv / dfy = y / z; dv / dcy = 1
        J[1::2, 1] = y / z
        J[1::2, 3] = 1.0
        
        self.H_accumulated += J.T @ J
        
        try:
            eig_vals = np.linalg.eigvalsh(self.H_accumulated)
            min_eig = np.min(eig_vals)
            max_eig = np.max(eig_vals)
            if min_eig < 1e-9: return 1e8 
            else: return max_eig / min_eig
        except:
            return 1e8

# ================= 4. 工具函数 =================
def look_at(camera_pos, target_pos, up=np.array([0, 0, 1])): 
    z_axis = camera_pos - target_pos 
    norm = np.linalg.norm(z_axis)
    if norm < 1e-6: return np.eye(4)
    z_axis = z_axis / norm
    
    if np.abs(np.dot(up, z_axis)) > 0.99:
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
    y_axis = np.cross(z_axis, x_axis)
    
    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = camera_pos
    return T

def move_rig_mocap(model, data, mocap_id, pose_matrix):
    pos = pose_matrix[:3, 3]
    quat = R.from_matrix(pose_matrix[:3, :3]).as_quat() 
    data.mocap_pos[mocap_id] = pos
    data.mocap_quat[mocap_id] = [quat[3], quat[0], quat[1], quat[2]]
    mujoco.mj_step(model, data)

def add_noise(img, sigma):
    if sigma <= 0: return img
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def find_corners_sim(gray_img, pattern_size):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray_img, pattern_size, flags)
    if ret: return True, corners
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    ret, corners = cv2.findChessboardCorners(blurred, pattern_size, flags)
    if ret: return True, corners
    return False, None

# ================= 5. 随机规划器 =================
class RandomPlanner:
    def __init__(self):
        self.candidates = self._generate_candidates()
        
    def _generate_candidates(self):
        cands = []
        radii = [0.90, 1.10, 1.30]
        elevations = np.deg2rad([40, 55, 70, 85]) 
        azimuths = np.linspace(-np.deg2rad(45), np.deg2rad(45), 7)
        target = np.array([0.0, 0.0, 0.02])
        
        for r in radii:
            for el in elevations:
                for az in azimuths:
                    z = r * np.sin(el) + target[2]
                    h = r * np.cos(el)
                    x = h * np.sin(az) + target[0]
                    y = -h * np.cos(az) + target[1]
                    pos = np.array([x, y, z])
                    pose = look_at(pos, target, up=np.array([0, 0, 1]))
                    cands.append(pose)
        return cands

    def get_next_pose(self):
        idx = random.randint(0, len(self.candidates) - 1)
        return self.candidates[idx], idx

# ================= 6. 主程序 =================
def main():
    print(f"\n[Stereo Random] Running Baseline...")
    
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)
    try: mocap_id = model.body("stereo_rig").mocapid[0]
    except: print("Error: Mocap ID not found"); return

    objp_base = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp_base[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp_base *= SQUARE_SIZE
    
    K_FOR_PNP = np.array([
        [GT_FX, 0, CAM_WIDTH/2], 
        [0, GT_FX, CAM_HEIGHT/2], 
        [0, 0, 1]
    ], dtype=np.float32)

    convergence_data = []
    
    # [新增] 存储绘图所需数据
    pose_history_for_plot = []  # 轨迹
    corner_history_for_plot = [] # 热力图角点

    for sigma in NOISE_LEVELS:
        for repeat in range(REPEATS_PER_LEVEL):
            mujoco.mj_resetData(model, data)
            planner = RandomPlanner()
            analyzer = GeometricAnalyzer() 
            
            objpoints, imgpL, imgpR = [], [], []
            valid_cnt, attempts = 0, 0
            
            # Random可能需要多试几次
            while valid_cnt < TARGET_VALID_PAIRS and attempts < 150: 
                attempts += 1
                curr_fx_err, curr_rms, curr_base_err = np.nan, np.nan, np.nan
                
                try:
                    pose, p_idx = planner.get_next_pose()
                    move_rig_mocap(model, data, mocap_id, pose)
                except: continue
                
                renderer.update_scene(data, camera="camera1_view")
                grayL = add_noise(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2GRAY), sigma)
                renderer.update_scene(data, camera="camera2_view")
                grayR = add_noise(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2GRAY), sigma)

                retL, corL = find_corners_sim(grayL, CHECKERBOARD)
                retR, corR = find_corners_sim(grayR, CHECKERBOARD)
                
                curr_cond = 1e8
                status = "Mis"
                
                if retL and retR:
                    try:
                        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        subL = cv2.cornerSubPix(grayL, corL, (11,11), (-1,-1), term)
                        subR = cv2.cornerSubPix(grayR, corR, (11,11), (-1,-1), term)
                        
                        objpoints.append(objp_base.astype(np.float32))
                        imgpL.append(subL.astype(np.float32))
                        imgpR.append(subR.astype(np.float32))
                        
                        valid_cnt += 1
                        status = "OK"
                        
                        # [新增] 数据记录: 仅针对 Noise=0, Repeat=0 的成功帧
                        if sigma == 0.0 and repeat == 0:
                            # 记录 3D 轨迹
                            pose_history_for_plot.append(pose[:3, 3].tolist())
                            # 记录 2D 角点 (用于热力图)
                            flat_corners = subL.reshape(-1, 2).tolist()
                            corner_history_for_plot.extend(flat_corners)
                        
                        # --- 几何分析 ---
                        ret_pnp, rvec_curr, tvec_curr = cv2.solvePnP(objp_base.astype(np.float32), subL, K_FOR_PNP, None)
                        if ret_pnp:
                            curr_cond = analyzer.update_and_compute_cond(objp_base.astype(np.float32), rvec_curr, tvec_curr)
                            
                        # --- 实时评估 ---
                        if valid_cnt >= 5:
                            # 单目
                            ret_temp, K_temp, _, _, _ = cv2.calibrateCamera(
                                objpoints, imgpL, (CAM_WIDTH, CAM_HEIGHT), None, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
                            curr_rms = ret_temp
                            curr_fx_err = abs(K_temp[0,0] - GT_FX) 
                            # 双目
                            _, _, _, _, _, _, T, _, _ = cv2.stereoCalibrate(
                                objpoints, imgpL, imgpR, K_temp, None, K_temp, None, 
                                (CAM_WIDTH, CAM_HEIGHT), flags=cv2.CALIB_FIX_INTRINSIC)
                            curr_base_err = abs(np.linalg.norm(T) - GT_BASELINE)
                        
                        if repeat == 0:
                            fname = f"noise_{sigma}_step_{valid_cnt:02d}.jpg"
                            cv2.imwrite(os.path.join(IMG_SAVE_DIR, fname), np.hstack((cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR), cv2.cvtColor(grayR, cv2.COLOR_GRAY2BGR))))

                    except Exception as e:
                        traceback.print_exc()
                
                # 记录每一次尝试 (无论成功失败)
                convergence_data.append({
                    "method": "random", "noise": sigma, "repeat_id": repeat,
                    "n_views": valid_cnt if status == "OK" else len(objpoints),
                    "cond_num": curr_cond, "focal_err": curr_fx_err, "rms": curr_rms, "base_err": curr_base_err
                })
                
                sys.stdout.write(f"\rNoise:{sigma} Rep:{repeat} | Got:{valid_cnt:02d} | {status} | Cond:{curr_cond:.1e} | Err(Base):{curr_base_err:.4f}")
                sys.stdout.flush()

    # 保存 CSV
    df = pd.DataFrame(convergence_data)
    df.fillna(value={"focal_err": 100.0, "rms": 100.0, "base_err": 0.10}, inplace=True)
    df.to_csv(os.path.join(OUTPUT_ROOT, "random_convergence_analysis.csv"), index=False)
    print(f"\n[Done] CSV saved.")

    # [新增] 保存 Trajectory & Corner JSONs
    if pose_history_for_plot:
        with open(os.path.join(OUTPUT_ROOT, "random_poses.json"), "w") as f:
            json.dump(pose_history_for_plot, f)
        with open(os.path.join(OUTPUT_ROOT, "random_corners.json"), "w") as f:
            json.dump(corner_history_for_plot, f)
        print(f"[Info] 绘图数据 (Poses & Corners) 已保存.")

if __name__ == "__main__":
    main()