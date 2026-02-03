import os
import sys
import json

# ================= 1. 渲染后端设置 =================
if 'MUJOCO_GL' in os.environ:
    del os.environ['MUJOCO_GL']
os.environ['MUJOCO_GL'] = 'egl' 

import mujoco
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R

# ================= 2. 配置参数 =================
XML_PATH = "/home/zkj/camera_calibration_sim/zhudongbiaoding/assets/scene2.xml"
OUTPUT_ROOT = "/home/zkj/camera_calibration_sim/zhudongbiaoding/output/stereo_greedy"
VIS_SAVE_DIR = os.path.join(OUTPUT_ROOT, "greedy_images")

CHECKERBOARD = (7, 9)
SQUARE_SIZE = 0.05

# 保持 720P 分辨率
CAM_WIDTH = 1280
CAM_HEIGHT = 720

TARGET_VALID_PAIRS = 20
GT_BASELINE = 0.10   
GT_FX = 869.3        

NOISE_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]
REPEATS_PER_LEVEL = 10 

if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
if not os.path.exists(VIS_SAVE_DIR): os.makedirs(VIS_SAVE_DIR)

# ================= 3. 核心功能函数 =================
def get_checkerboard_center(model, data):
    target_pos = np.array([0.0, 0.0, 0.02])
    found = False
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and ('check' in name.lower() or 'target' in name.lower()):
            target_pos = data.xpos[i]
            if target_pos[2] < 0.005: target_pos[2] = 0.021
            found = True
            break
    if not found: print("[Info] Target default at (0,0,0.02)")
    return target_pos

def look_at_matrix(camera_pos, target_pos, up=np.array([0, 0, 1]), flip_z=False):
    if not flip_z: z_axis = camera_pos - target_pos 
    else: z_axis = target_pos - camera_pos 
    norm = np.linalg.norm(z_axis)
    if norm < 1e-6: return np.eye(4)
    z_axis = z_axis / norm
    if np.abs(np.dot(up, z_axis)) > 0.99: x_axis = np.array([1, 0, 0])
    else: x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
    T = np.eye(4); T[:3, 0] = x_axis; T[:3, 1] = y_axis; T[:3, 2] = z_axis; T[:3, 3] = camera_pos
    return T

def move_rig(model, data, mocap_id, pose_mat):
    pos = pose_mat[:3, 3]
    quat = R.from_matrix(pose_mat[:3, :3]).as_quat() 
    data.mocap_pos[mocap_id] = pos
    data.mocap_quat[mocap_id] = [quat[3], quat[0], quat[1], quat[2]]
    mujoco.mj_step(model, data)

def test_camera_orientation(model, data, renderer, mocap_id):
    print("\n>>> 校准相机视线方向...")
    target = get_checkerboard_center(model, data)
    test_pos = target + np.array([0.5, 0.5, 1.0])
    modes, best_mode, max_b = [False, True], False, -1
    for mode in modes:
        pose = look_at_matrix(test_pos, target, flip_z=mode)
        move_rig(model, data, mocap_id, pose)
        renderer.update_scene(data, camera="camera1_view")
        gray = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2GRAY)
        if np.mean(gray) > max_b: max_b = np.mean(gray); best_mode = mode
    print(f"   >>> 锁定模式: {'Flip-Z' if best_mode else 'Standard'}")
    return best_mode, target

# ================= 4. FIM 规划器 =================
class FIMPlanner:
    def __init__(self, flip_mode=False):
        self.flip_mode = flip_mode
        self.candidates = []
        self._pool_generated = False
        self.selected_indices = set()
        self.H_accum = np.eye(4) * 1e-6 

    def ensure_pool(self, target_pos):
        if self._pool_generated: return
        radii = [0.8, 1.0, 1.25]
        elevations = np.deg2rad([35, 50, 65, 80])
        azimuths = np.linspace(np.deg2rad(-45), np.deg2rad(45), 9)
        for r in radii:
            for el in elevations:
                for az in azimuths:
                    z = r * np.sin(el) + target_pos[2]
                    h = r * np.cos(el)
                    x, y = h * np.sin(az) + target_pos[0], -h * np.cos(az) + target_pos[1]
                    pose = look_at_matrix(np.array([x,y,z]), target_pos, flip_z=self.flip_mode)
                    self.candidates.append(pose)
        self._pool_generated = True

    def _simulate_fim(self, obj_points, pose_matrix):
        inv_pose = np.linalg.inv(pose_matrix)
        rvec, _ = cv2.Rodrigues(inv_pose[:3, :3])
        tvec = inv_pose[:3, 3]
        K_sim = np.array([[GT_FX, 0, CAM_WIDTH/2], [0, GT_FX, CAM_HEIGHT/2], [0, 0, 1]])
        img_pts, _ = cv2.projectPoints(obj_points, rvec, tvec, K_sim, np.zeros(5))
        img_pts = img_pts.reshape(-1, 2)
        mask = (img_pts[:,0]>=0) & (img_pts[:,0]<CAM_WIDTH) & (img_pts[:,1]>=0) & (img_pts[:,1]<CAM_HEIGHT)
        if np.sum(mask) < 10: return np.zeros((4,4))
        
        R_mat, _ = cv2.Rodrigues(rvec)
        pts_cam = (R_mat @ obj_points.T).T + tvec.T
        Z = np.abs(pts_cam[mask, 2]) + 1e-6
        X, Y = pts_cam[mask, 0], pts_cam[mask, 1]
        J = np.zeros((2*len(Z), 4))
        J[0::2, 0] = X/Z; J[0::2, 2] = 1.0
        J[1::2, 1] = Y/Z; J[1::2, 3] = 1.0
        return J.T @ J

    def select_next(self, obj_points_base):
        if not self.selected_indices: return self.candidates[len(self.candidates)//2], len(self.candidates)//2
        best_idx, best_cond = -1, 1e20
        available = [i for i in range(len(self.candidates)) if i not in self.selected_indices]
        for idx in available:
            H_temp = self.H_accum + self._simulate_fim(obj_points_base, self.candidates[idx])
            vals = np.linalg.eigvalsh(H_temp)
            cond = np.max(vals)/np.min(vals) if np.min(vals)>1e-9 else 1e9
            if cond < best_cond: best_cond = cond; best_idx = idx
        return self.candidates[best_idx], best_idx

    def update(self, idx, obj_points_base):
        self.selected_indices.add(idx)
        self.H_accum += self._simulate_fim(obj_points_base, self.candidates[idx])

    def get_cond(self):
        vals = np.linalg.eigvalsh(self.H_accum)
        return np.max(vals) / np.min(vals) if np.min(vals)>1e-9 else 1e9
    
    def mark_failed(self, idx):
        self.selected_indices.add(idx)

# ================= 5. 主程序 =================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)
    try: mocap_id = model.body("stereo_rig").mocapid[0]
    except: print("Error: Mocap ID not found"); return

    correct_flip_mode, target_center = test_camera_orientation(model, data, renderer, mocap_id)
    
    objp_base = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp_base[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp_base *= SQUARE_SIZE 
    
    all_logs = []
    pose_history_for_plot = []  # 存 Pose
    corner_history_for_plot = [] # [新增] 存角点 (u, v)
    
    print("\n>>> 开始采集 (Greedy)...")
    for sigma in NOISE_LEVELS:
        for repeat in range(REPEATS_PER_LEVEL):
            mujoco.mj_resetData(model, data)
            planner = FIMPlanner(flip_mode=correct_flip_mode)
            planner.ensure_pool(target_center)
            
            objpoints, imgL_list, imgR_list = [], [], []
            valid_cnt = 0
            
            while valid_cnt < TARGET_VALID_PAIRS:
                pose, p_idx = planner.select_next(objp_base)
                move_rig(model, data, mocap_id, pose)
                
                renderer.update_scene(data, camera="camera1_view")
                grayL = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2GRAY)
                renderer.update_scene(data, camera="camera2_view")
                grayR = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2GRAY)
                
                # 加噪
                def add_n(img, s): 
                    if s<=0: return img
                    return np.clip(img + np.random.normal(0,s,img.shape),0,255).astype(np.uint8)
                grayL = add_n(grayL, sigma)
                grayR = add_n(grayR, sigma)
                
                flags_cb = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
                retL, corL = cv2.findChessboardCorners(grayL, CHECKERBOARD, flags_cb)
                retR, corR = cv2.findChessboardCorners(grayR, CHECKERBOARD, flags_cb)
                
                if retL and retR:
                    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    subL = cv2.cornerSubPix(grayL, corL, (11,11), (-1,-1), term)
                    subR = cv2.cornerSubPix(grayR, corR, (11,11), (-1,-1), term)
                    
                    objpoints.append(objp_base)
                    imgL_list.append(subL)
                    imgR_list.append(subR)
                    
                    if repeat == 0:
                        visL = cv2.cvtColor(grayL, cv2.COLOR_GRAY2BGR)
                        visR = cv2.cvtColor(grayR, cv2.COLOR_GRAY2BGR)
                        cv2.drawChessboardCorners(visL, CHECKERBOARD, subL, True)
                        cv2.drawChessboardCorners(visR, CHECKERBOARD, subR, True)
                        fname = f"corners_N{sigma}_View{valid_cnt+1:02d}.jpg"
                        cv2.imwrite(os.path.join(VIS_SAVE_DIR, fname), np.hstack((visL, visR)))

                    # [轨迹与角点保存] 仅 Noise=0 (无噪) 且 Repeat=0 (第一次)
                    if sigma == 0.0 and repeat == 0:
                        # 1. 保存 Pose
                        pose_history_for_plot.append(pose[:3, 3].tolist())
                        # 2. [新增] 保存对应左图的角点 (N, 1, 2) -> (N, 2)
                        corners_flat = subL.reshape(-1, 2).tolist()
                        corner_history_for_plot.extend(corners_flat)

                    planner.update(p_idx, objp_base)
                    valid_cnt += 1
                
                else:
                    planner.mark_failed(p_idx)
                    sys.stdout.write(f"\r[Warning] 视角 {p_idx} 失败 (当前: {valid_cnt})")
                    sys.stdout.flush()
                
                # === 评估 ===
                curr_cond = planner.get_cond()
                curr_base_err, curr_rms, curr_focal_err = np.nan, np.nan, np.nan

                if valid_cnt >= 5 and len(objpoints) >= 5:
                    try:
                        ret_val, mtx_val, _, _, _ = cv2.calibrateCamera(
                            objpoints, imgL_list, (CAM_WIDTH, CAM_HEIGHT), 
                            None, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT)
                        curr_rms = ret_val
                        curr_focal_err = abs(mtx_val[0,0] - GT_FX)
                        _, _, _, _, _, _, T, _, _ = cv2.stereoCalibrate(
                            objpoints, imgL_list, imgR_list, mtx_val, None, mtx_val, None, 
                            (CAM_WIDTH, CAM_HEIGHT), flags=cv2.CALIB_FIX_INTRINSIC)
                        curr_base_err = abs(np.linalg.norm(T) - GT_BASELINE)
                    except: pass
                
                all_logs.append({
                    "noise": sigma, "repeat": repeat, "views": valid_cnt, 
                    "cond_num": curr_cond, "base_err": curr_base_err, "focal_err": curr_focal_err, "rms": curr_rms
                })
                sys.stdout.write(f"\r Noise:{sigma} Rep:{repeat} View:{valid_cnt} | Cond:{curr_cond:.1e} | Err:{curr_base_err:.4f}")
                sys.stdout.flush() 

    df = pd.DataFrame(all_logs)
    df.fillna(value={"base_err":0.1, "focal_err":100.0, "rms":100.0}, inplace=True)
    df.to_csv(os.path.join(OUTPUT_ROOT, "greedy_convergence_analysis.csv"), index=False)
    print(f"\n[Done] CSV 保存完毕")

    # 保存轨迹和角点数据
    if pose_history_for_plot:
        with open(os.path.join(OUTPUT_ROOT, "greedy_poses.json"), "w") as f:
            json.dump(pose_history_for_plot, f)
        # [新增] 保存角点数据
        with open(os.path.join(OUTPUT_ROOT, "greedy_corners.json"), "w") as f:
            json.dump(corner_history_for_plot, f)
        print(f"[Info] 轨迹与角点数据已保存 (JSON)")

if __name__ == "__main__":
    main()