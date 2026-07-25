import os
import time
import pandas as pd
import cv2
import yaml
import numpy as np
import math
from ultralytics import YOLO
import torch
from scipy.optimize import linear_sum_assignment
import pyzed.sl as sl

# ===== LOAD CONFIG & GROUND-TRUTH =====
with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
model_custom = cfg['cameraConfig']['model']
conf_thresh  = cfg['cameraConfig'].get('conf')
iou_thresh   = cfg['cameraConfig'].get('iou')
gt_df = pd.read_csv('ground_truth.csv')
gt_map = dict(zip(gt_df['frame'], gt_df['present']))

# ===== PARAMETERS =====
svo_path    = r"D:\SEMESTER 7\TA\DATASET\VIDEO TESTING FIX\t50j120-280.svo2"
output_folder = 'hasil_testing_small_revisi'
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 't50j120-280.mp4')
MAX_DEPTH_M = 5.0  # max valid depth in meters
H_cam       = 0.30 # camera mounting height (m)
theta       = math.radians(-30)  # pitch angle downwards (deg)
R_pitch     = np.array([
    [1,               0,                0],
    [0, math.cos(theta), -math.sin(theta)],
    [0, math.sin(theta), math.cos(theta)]
], dtype=np.float32)

# ===== INITIALIZE ZED SVO =====
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.set_from_svo_file(svo_path)
init_params.svo_real_time_mode = False
init_params.depth_mode       = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError(f"Failed to open SVO: {status}")
runtime_params = sl.RuntimeParameters()

# Get resolution and setup writer
cam_info = zed.get_camera_information()
widthL = cam_info.camera_configuration.resolution.width
heightL = cam_info.camera_configuration.resolution.height
calibL   = cam_info.camera_configuration.calibration_parameters.left_cam
baseline = cam_info.camera_configuration.calibration_parameters.get_camera_baseline()
print(f"Baseline kamera: {baseline:.4f} meter")
fx = calibL.fx   # focal length x (px)
fy = calibL.fy   # focal length y (px)
Cx = calibL.cx   # principal point x (px)
Cy = calibL.cy   # principal point y (px)
full_w = widthL * 2
full_h = heightL
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out_writer = cv2.VideoWriter(output_path, fourcc, 30, (full_w, full_h))

# ===== INITIALIZE YOLO =====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = YOLO(model_custom).to(device)

# ===== PROCESS FRAMES =====n
mapping_log = []
frame_idx = 0
max_frame =150
prev_frame_time = time.time()
pre_times   = []
inf_times   = []
post_times  = []
metrics_log = []
total_frames      = 0
total_TP = total_FP = total_FN = total_TN = 0

print("=== Start processing SVO frames ===")
while frame_idx < max_frame:
    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        break

    # retrieve
    left_mat  = sl.Mat(); right_mat    = sl.Mat()
    depth_mat = sl.Mat(); point_cloud  = sl.Mat()
    zed.retrieve_image(left_mat,  sl.VIEW.LEFT)
    zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
    zed.retrieve_measure(depth_mat,   sl.MEASURE.DEPTH)
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

    # BGR images
    left  = cv2.cvtColor(left_mat.get_data(),  cv2.COLOR_BGRA2BGR)
    right = cv2.cvtColor(right_mat.get_data(), cv2.COLOR_BGRA2BGR)
    # PC and depth
    pc_np    = point_cloud.get_data()    # HxWx4
    XYZ      = pc_np[..., :3]            # HxWx3
    depth_np = depth_mat.get_data()      # HxW

    # YOLO detect
    t0 = time.time()
    resL = model.predict(left,  device=device, conf=conf_thresh, iou=iou_thresh, stream=False)
    resR = model.predict(right, device=device, conf=conf_thresh, iou=iou_thresh, stream=False)
    boxesL = resL[0].boxes.xyxy.cpu().numpy().astype(int) if resL and resL[0].boxes else []
    boxesR = resR[0].boxes.xyxy.cpu().numpy().astype(int) if resR and resR[0].boxes else []
    t1 = time.time()

    # latency breakdown
    spL, spR = getattr(resL[0], 'speed', None), getattr(resR[0], 'speed', None)
    if spL and spR:
        pre_times.append(spL['preprocess']   + spR['preprocess'])
        inf_times.append(spL['inference']    + spR['inference'])
        post_times.append(spL['postprocess'] + spR['postprocess'])
    else:
        tot_ms = (t1 - t0) * 1000
        pre_times.append(0); inf_times.append(tot_ms); post_times.append(0)

    print(f"[Frame {frame_idx}] #L={len(boxesL)}, #R={len(boxesR)}")
    
    # draw raw detections
    for (x1,y1,x2,y2) in boxesL:
        cv2.rectangle(left,  (x1,y1),(x2,y2), (255,0,0), 2)
    for (x1,y1,x2,y2) in boxesR:
        cv2.rectangle(right, (x1,y1),(x2,y2), (255,0,0), 2)

    # stereo matching hungarian algorithm
    centersL = [( (b[0]+b[2])//2, (b[1]+b[3])//2 ) for b in boxesL]
    centersR = [( (b[0]+b[2])//2, (b[1]+b[3])//2 ) for b in boxesR]
    matches = []
    if centersL and centersR:
        N,M = len(centersL), len(centersR)
        cost = np.full((N,M), np.inf)
        for i,(xcL,ycL) in enumerate(centersL):
            for j,(xcR,ycR) in enumerate(centersR):
                cost[i,j] = abs(xcL-xcR) + abs(ycL-ycR)
        row, col = linear_sum_assignment(cost)
        matches = [(i,j) for i,j in zip(row,col) if cost[i,j] < 700]

    # compute & annotate distances
    for i,j in matches:
        x1,y1,x2,y2   = boxesL[i]
        x1r,y1r,x2r,y2r = boxesR[j]

         # hitung disparitas dan piksel center
        uL, vL = centersL[i]
        uR, vR = centersR[j]
        disparity = abs(uL - uR)
        
        # collect all valid 3D points in ROI
        roi = XYZ[y1:y2, x1:x2, :].reshape(-1,3)
        mask = (np.isfinite(roi[:,2]) & (roi[:,2]>0) & (roi[:,2]< MAX_DEPTH_M))
        valid = roi[mask]
        if valid.shape[0] == 0:
            # fallback to depth map (ambil median depth)
            patch = depth_np[y1:y2, x1:x2]
            candidates = patch[np.isfinite(patch) & (patch>0) & (patch<MAX_DEPTH_M)]
            if candidates.size == 0:
                continue
            # Z_slant ≈ median depth
            Z_slant = np.median(candidates)
        else:
            # Rotasi ke world frame & tambahkan ketinggian kamera point cloud
            p_cam = valid.T                                   # shape: (3, N)
            p_w   = (R_pitch @ p_cam).T
            p_w_rep = np.median(p_w, axis=0)
            X_rep,Y_rep,Z_rep = p_w_rep
        
        # Hitung slant range point clud
        slants = np.linalg.norm(p_w, axis=1)
        Z_slant = slants.mean()
        
        # 1b) Validasi menggunakan triangulasi stereo
        Z_est = (fx * baseline) / disparity
        X_est = ((uL - Cx) * Z_est) / fx
        Y_est = ((vL - Cy) * Z_est) / fy    
            
        # Rotasi + translasi
        Y_world = np.cos(theta) * Y_est - np.sin(theta) * Z_est 
        Z_world = np.sin(theta) * Y_est + np.cos(theta) * Z_est
        
        # Validasi triangulasi dan hasil XYZ
        disparity_valid = 2 < disparity < 500
        xyz_valid = all(np.isfinite([X_est, Y_world, Z_world])) and all(np.abs([X_est, Y_world, Z_world]) < 2.0)

        if not (disparity_valid and xyz_valid):
            slant = 0
            d_ground = 0
        else:
            slant = np.linalg.norm([X_est, Y_world, Z_world])
            d_ground = np.sqrt(np.clip(slant**2 - H_cam**2, a_min=0, a_max=None))


        d_ground_pointcloud = np.sqrt(np.clip(Z_slant**2 - H_cam**2, a_min=0, a_max=None))
        
        # Konversi ke cm
        dist_cm = d_ground * 100
        
        dist_cm_pointcloud = d_ground_pointcloud * 100
        
        # draw matched boxes & labels
        cv2.rectangle(left,  (x1,y1),(x2,y2),   (0,255,0), 2)
        cv2.putText(left,  f"{i}-{dist_cm:.2f}cm", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.rectangle(right, (x1r,y1r),(x2r,y2r), (0,0,255), 2)
        cv2.putText(right, f"{i}-{dist_cm:.2f}cm", (x1r,y1r-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        mapping_log.append({
            'frame': frame_idx,
            'ID': i,
            'left_xmin': x1, 'left_ymin': y1,
            'left_xmax': x2, 'left_ymax': y2,
            'f':        fx,           # focal length (px)
            'C_x':      Cx,           # principal point x (px)
            'C_y':      Cy,           # principal point y (px)
            'x_il':     uL,           # x kiri (px)
            'v_il':     vL,           # y kiri (px)
            'x_ir':     uR,           # x kanan (px)
            'v_ir':     vR,           # y kanan (px)
            'disparitas': disparity,
            'X_pc' : X_rep,
            'Y_pc': Y_rep,
            'Z_pc':Z_rep,
            'Z_slant_pc': Z_slant,
            'cm_pc': round(dist_cm_pointcloud,2),
            'X_ts': X_est,
            'Y_ts': Y_world,
            'Z_ts': Z_world,
            'Z_slant_ts': slant,
            'cm_ts': round(dist_cm,2),
        })

    # write & show
    out = np.hstack([left, right])
    
    # instance‐level metrics
    TP = len(matches)
    FP = len(boxesL) - TP
    FN = len(boxesR) - TP
    total_TP += TP; total_FP += FP; total_FN += FN

    # --- log per‐frame untuk CSV ---
    metrics_log.append({
        'i': frame_idx+1,     # 1-based frame index; hapus +1 kalau mau 0-based
        'L_i': len(boxesL),
        'R_i': len(boxesR),
        'M_i': TP,
        'FP_i': FP,
        'FN_i': FN
    })

    
    # ===== hitung dan gambar FPS =====
    new_frame_time = time.time()
    fps = 1.0 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    fps_text = f"FPS: {fps:.1f}"
    text_size, _ = cv2.getTextSize(fps_text,
                                   cv2.FONT_HERSHEY_SIMPLEX,
                                   1, 2)
    text_x = full_w - text_size[0] - 10
    text_y = 30
    cv2.putText(out, fps_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,255), 2, cv2.LINE_AA)
    
    out_writer.write(out)
    cv2.imshow("SVO ZED2", cv2.resize(out, (full_w//2, full_h//2)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# cleanup & save
zed.close()
out_writer.release()
cv2.destroyAllWindows()

# setelah zed.close(), out_writer.release(), dst.:
metrics_df = pd.DataFrame(metrics_log)
metrics_df.to_csv(os.path.join(output_folder, 't50j120-280.csv'),  sep=',', float_format='%.6f', index=False)
print(f"Per‐frame metrics saved to {os.path.join(output_folder, 't50j120-280.csv')}")

# save log
output_folder = 'result_distance_small_revisi'
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, 't50j120-280.csv')
pd.DataFrame(mapping_log).to_csv(csv_path, index=False)
print(f"Finished. Output video -> {output_path}")

# final metrics & latency
filtered_data = [d for d in mapping_log if 160 <= d['cm_ts'] <= 169]

TP = len(filtered_data)  
FP = total_FP  
FN = total_FN

den = TP + FP + FN
acc = TP / den * 100 if den > 0 else 0.0

print("\n=== Hasil Evaluasi ===")
print(f"Frames diproses : {frame_idx}")
print(f"TP:{total_TP}  FP:{total_FP}  FN:{total_FN}  Acc:{acc:.2f}%")

if device=='cuda':
    a = torch.cuda.memory_allocated()/1024**2
    p = torch.cuda.max_memory_allocated()/1024**2
    print(f"GPU Mem (MB): Alloc {a:.1f}, Peak {p:.1f}")

print(f"\nSelesai! Video saved to {output_path}")
