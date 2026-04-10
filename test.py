import torch
import os
import cv2
import numpy as np
import warnings
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import cfg
from adapter import collate_dexycb_to_mvgformer
from train import build_model, compute_mpjpe

from datasets.DexYCB import DEXYCBDatasets
from datasets.DriverHOI import DriverHOIDatasets

HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# 手指分组 (用于按手指统计 MPJPE)
FINGER_GROUPS = {
    'Wrist':  [0],
    'Thumb':  [1, 2, 3, 4],
    'Index':  [5, 6, 7, 8],
    'Middle': [9, 10, 11, 12],
    'Ring':   [13, 14, 15, 16],
    'Pinky':  [17, 18, 19, 20],
}


# ------------------------------------------------------------------------------------------------
# 评价指标计算工具
# ------------------------------------------------------------------------------------------------

def compute_pa_mpjpe(pred, gt):
    """
    计算 PA-MPJPE (Procrustes Aligned MPJPE)。
    通过 Procrustes 对齐消除全局旋转、平移和缩放差异后计算关节误差。

    Args:
        pred (np.ndarray): (N, 21, 3) 预测三维坐标
        gt (np.ndarray): (N, 21, 3) 真值三维坐标

    Returns:
        float: 所有样本的 PA-MPJPE 误差之和 (单位: mm)
        int: 有效关节点总数
    """
    total_err = 0.0
    total_count = 0

    for i in range(pred.shape[0]):
        p = pred[i]  # (21, 3)
        g = gt[i]    # (21, 3)

        # 1. 中心化
        mu_p = p.mean(axis=0, keepdims=True)
        mu_g = g.mean(axis=0, keepdims=True)
        p_centered = p - mu_p
        g_centered = g - mu_g

        # 2. 计算最优旋转 (SVD)
        H = p_centered.T @ g_centered  # (3, 3)
        U, S, Vt = np.linalg.svd(H)

        # 处理反射情况
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.eye(3)
        sign_matrix[2, 2] = np.sign(d)

        R = Vt.T @ sign_matrix @ U.T

        # 3. 计算最优缩放
        var_p = np.sum(p_centered ** 2)
        scale = np.trace(np.diag(S) @ sign_matrix) / (var_p + 1e-8)

        # 4. 对齐
        p_aligned = scale * (p_centered @ R.T) + mu_g

        # 5. 计算误差 (转换为 mm)
        err = np.linalg.norm(p_aligned - g, axis=-1) * 1000.0
        total_err += err.sum()
        total_count += err.shape[0]

    return total_err, total_count


def compute_pck(pred, gt, thresholds_mm):
    """
    计算 PCK (Percentage of Correct Keypoints)。

    Args:
        pred (np.ndarray): (N, 21, 3) 预测三维坐标
        gt (np.ndarray): (N, 21, 3) 真值三维坐标
        thresholds_mm (list[float]): 阈值列表 (单位: mm)

    Returns:
        dict: {threshold: pck_value} 各阈值下的 PCK 值
        int: 总关节点数
    """
    per_joint_err = np.linalg.norm(pred - gt, axis=-1) * 1000.0  # (N, 21), mm
    total_joints = per_joint_err.size

    pck_dict = {}
    for thr in thresholds_mm:
        correct = (per_joint_err < thr).sum()
        pck_dict[thr] = correct

    return pck_dict, total_joints


def compute_auc(pck_curve, thresholds_mm):
    """
    计算 PCK 曲线下面积 (AUC)。

    Args:
        pck_curve (list[float]): 各阈值下的 PCK 值 (0~1)
        thresholds_mm (list[float]): 对应的阈值列表

    Returns:
        float: AUC 值 (0~1)
    """
    thresholds = np.array(thresholds_mm)
    pck_values = np.array(pck_curve)

    # 梯形积分法, 归一化到 [0, 1]
    auc = np.trapz(pck_values, thresholds) / (thresholds[-1] - thresholds[0])
    return auc


def compute_reprojection_error(pred_3d, gt_3d, cameras, img_size):
    """
    计算二维重投影误差。

    Args:
        pred_3d (np.ndarray): (21, 3) 预测三维坐标
        gt_3d (np.ndarray): (21, 3) 真值三维坐标
        cameras (dict): 单个样本的相机参数
        img_size (tuple): (H, W)

    Returns:
        float: 该样本在所有有效视角上的平均重投影误差 (单位: pixel)
        int: 有效投影点数量
    """
    H, W = img_size
    total_err = 0.0
    total_count = 0

    V = cameras['camera_R'].shape[0]

    for v in range(V):
        K = cameras['camera_Intri'][v]
        R = cameras['camera_R'][v]
        T = cameras['camera_T'][v]

        # 投影预测
        pred_cam = pred_3d @ R.T + T[np.newaxis, :]
        pred_z = np.maximum(pred_cam[:, 2], 0.01)
        pred_u = pred_cam[:, 0] * K[0, 0] / pred_z + K[0, 2]
        pred_v = pred_cam[:, 1] * K[1, 1] / pred_z + K[1, 2]

        # 投影真值
        gt_cam = gt_3d @ R.T + T[np.newaxis, :]
        gt_z = np.maximum(gt_cam[:, 2], 0.01)
        gt_u = gt_cam[:, 0] * K[0, 0] / gt_z + K[0, 2]
        gt_v = gt_cam[:, 1] * K[1, 1] / gt_z + K[1, 2]

        # 有效性掩码: 深度 > 0.05 且在图像范围内
        mask = (gt_cam[:, 2] > 0.05) & \
               (gt_u >= 0) & (gt_u < W) & \
               (gt_v >= 0) & (gt_v < H)

        if mask.sum() == 0:
            continue

        err = np.sqrt((pred_u[mask] - gt_u[mask]) ** 2 +
                      (pred_v[mask] - gt_v[mask]) ** 2)
        total_err += err.sum()
        total_count += mask.sum()

    return total_err, total_count


def compute_per_finger_mpjpe(pred, gt):
    """
    按手指分组统计 MPJPE。

    Args:
        pred (np.ndarray): (N, 21, 3)
        gt (np.ndarray): (N, 21, 3)

    Returns:
        dict: {finger_name: mpjpe_mm}
    """
    per_joint_err = np.linalg.norm(pred - gt, axis=-1) * 1000.0  # (N, 21), mm
    results = {}
    for name, indices in FINGER_GROUPS.items():
        results[name] = per_joint_err[:, indices].mean()
    return results


# ------------------------------------------------------------------------------------------------
# 可视化工具 (保持原有实现)
# ------------------------------------------------------------------------------------------------

def project_3d_to_2d(pts_3d, K, R, T):
    pts_cam = np.dot(pts_3d, R.T) + T[np.newaxis, :]
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    z = np.maximum(z, 0.01)
    u = x * K[0, 0] / z + K[0, 2]
    v = y * K[1, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def draw_skeleton(img, pts_2d, color=(0, 0, 255), radius=4, thickness=2):
    for p in pts_2d:
        cv2.circle(img, (int(p[0]), int(p[1])), radius, color, -1)
    for (s, e) in HAND_SKELETON:
        if s < len(pts_2d) and e < len(pts_2d):
            pt1 = (int(pts_2d[s][0]), int(pts_2d[s][1]))
            pt2 = (int(pts_2d[e][0]), int(pts_2d[e][1]))
            cv2.line(img, pt1, pt2, color, thickness)
    return img


def draw_3d_skeleton_to_image(pred_3d, gt_3d, mpjpe_val, target_size=(512, 512)):
    fig = plt.figure(figsize=(target_size[0] / 100, target_size[1] / 100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], c='g', marker='o', s=15, label='GT')
    for s, e in HAND_SKELETON:
        ax.plot([gt_3d[s, 0], gt_3d[e, 0]],
                [gt_3d[s, 1], gt_3d[e, 1]],
                [gt_3d[s, 2], gt_3d[e, 2]], color='g', linewidth=1.5)

    ax.text(gt_3d[0, 0], gt_3d[0, 1], gt_3d[0, 2],
            f'GT_Root\n({gt_3d[0, 0]:.3f}, {gt_3d[0, 1]:.3f}, {gt_3d[0, 2]:.3f})',
            color='darkgreen', fontsize=8, fontweight='bold')

    ax.scatter(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2], c='r', marker='^', s=15, label='Pred')
    for s, e in HAND_SKELETON:
        ax.plot([pred_3d[s, 0], pred_3d[e, 0]],
                [pred_3d[s, 1], pred_3d[e, 1]],
                [pred_3d[s, 2], pred_3d[e, 2]], color='r', linewidth=1.5)

    all_pts = np.vstack([pred_3d, gt_3d])
    max_range = np.array([all_pts[:, 0].max() - all_pts[:, 0].min(),
                          all_pts[:, 1].max() - all_pts[:, 1].min(),
                          all_pts[:, 2].max() - all_pts[:, 2].min()]).max() / 2.0
    mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) * 0.5
    mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) * 0.5
    mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_title(f'3D Pose (MPJPE: {mpjpe_val:.2f}mm)\nGreen: GT, Red: Pred', fontsize=12)
    ax.legend(loc='upper right', fontsize='small')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel('X', fontsize=9)
    ax.set_ylabel('Y', fontsize=9)
    ax.set_zlabel('Z', fontsize=9)

    fig.tight_layout()
    fig.canvas.draw()
    img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, target_size)
    return img_bgr


# ------------------------------------------------------------------------------------------------
# 主测试流程
# ------------------------------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ckpt_path = cfg.TEST.TEST_CKPT
    if not ckpt_path:
        print("[Error] cfg.TEST.TEST_CKPT is empty. Please specify a checkpoint path.")
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Testing Model: {cfg.MODEL.NAME} | Dataset: {cfg.DATASET.NAME}")

    dataset_name = cfg.DATASET.NAME.lower()
    root_dir = cfg.DATASET.ROOT_DEXYCB if dataset_name == 'dexycb' else cfg.DATASET.ROOT_DRIVERHOI
    DatasetClass = DEXYCBDatasets if dataset_name == 'dexycb' else DriverHOIDatasets

    test_dataset = DatasetClass(
        root_dir=root_dir,
        split='test',
        split_strategy=cfg.DATASET.SPLIT_STRATEGY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    model = build_model(cfg, device)

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict']

    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    viz_save_dir = f"{cfg.TEST.VIZ_DIR}_{cfg.MODEL.NAME}"
    net_h, net_w = cfg.NETWORK.IMAGE_SIZE

    # -------------------------------------------------------
    # 指标累加器
    # -------------------------------------------------------
    # 1. MPJPE
    total_mpjpe_sum = 0.0
    total_mpjpe_count = 0

    # 2. PA-MPJPE
    total_pa_mpjpe_sum = 0.0
    total_pa_mpjpe_count = 0

    # 3. PCK (阈值范围: 0~50mm, 步长1mm)
    pck_thresholds = list(range(1, 51))
    pck_correct = {thr: 0 for thr in pck_thresholds}
    pck_total_joints = 0

    # 4. 二维重投影误差
    total_reproj_sum = 0.0
    total_reproj_count = 0

    # 5. 按手指分组的逐样本误差收集
    finger_errors_all = {name: [] for name in FINGER_GROUPS.keys()}

    print(">>> Start Evaluation <<<")
    with torch.no_grad():
        for i, (batch_inputs, batch_targets, meta_batch) in enumerate(tqdm(test_loader, desc="Testing")):

            views, meta = collate_dexycb_to_mvgformer(batch_inputs, batch_targets, device)
            outputs, _ = model(views, meta)

            if 'final_pred_poses' in outputs:
                pred_poses = outputs['final_pred_poses']
            elif 'pred_poses' in outputs and 'outputs_coord' in outputs['pred_poses']:
                pred_poses = outputs['pred_poses']['outputs_coord']
            else:
                raise ValueError("Output format not recognized")

            if pred_poses.dim() == 4:
                pred_poses = pred_poses.squeeze(1)

            gt_poses = torch.stack([m['joints_3d'] for m in meta])
            if gt_poses.dim() == 4:
                gt_poses = gt_poses.squeeze(1)
            gt_poses = gt_poses.to(device)

            B = pred_poses.shape[0]
            V = len(views)
            img_size = (net_h, net_w)

            pred_np = pred_poses.cpu().numpy()  # (B, 21, 3)
            gt_np = gt_poses.cpu().numpy()      # (B, 21, 3)

            # ------- 1. MPJPE -------
            err_sum, valid_count = compute_mpjpe(pred_poses, gt_poses)
            total_mpjpe_sum += err_sum
            total_mpjpe_count += valid_count

            # ------- 2. PA-MPJPE -------
            pa_err_sum, pa_count = compute_pa_mpjpe(pred_np, gt_np)
            total_pa_mpjpe_sum += pa_err_sum
            total_pa_mpjpe_count += pa_count

            # ------- 3. PCK -------
            batch_pck, batch_joints = compute_pck(pred_np, gt_np, pck_thresholds)
            for thr in pck_thresholds:
                pck_correct[thr] += batch_pck[thr]
            pck_total_joints += batch_joints

            # ------- 4. 二维重投影误差 -------
            for b in range(B):
                cam_b = {
                    'camera_R': meta[b]['camera_R'].cpu().numpy(),
                    'camera_T': meta[b]['camera_T'].cpu().numpy(),
                    'camera_Intri': meta[b]['camera_Intri'].cpu().numpy(),
                }
                reproj_err, reproj_cnt = compute_reprojection_error(
                    pred_np[b], gt_np[b], cam_b, img_size
                )
                total_reproj_sum += reproj_err
                total_reproj_count += reproj_cnt

            # ------- 5. 按手指分组 MPJPE -------
            finger_mpjpe = compute_per_finger_mpjpe(pred_np, gt_np)
            for name, val in finger_mpjpe.items():
                finger_errors_all[name].append(val)

            # ------- 可视化 -------
            if cfg.TEST.VIZ and (i % cfg.TEST.VIZ_FREQ == 0):
                os.makedirs(viz_save_dir, exist_ok=True)
                b = 0
                sample_vis_imgs = []

                cur_pred = pred_np[b]
                cur_gt = gt_np[b]
                sample_mpjpe = np.linalg.norm(cur_pred - cur_gt, axis=-1).mean() * 1000.0

                is_driverhoi = dataset_name == 'driverhoi'
                viz_size = 1024 if is_driverhoi else net_w  # DriverHOI 用更大的展示尺寸
                viz_scale = viz_size / net_w  # 坐标从 256 空间放大到展示空间的倍率

                for v in range(V):
                    img_path = meta_batch['img_paths'][v][b]
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        img_bgr = np.zeros((viz_size, viz_size, 3), dtype=np.uint8)
                        sample_vis_imgs.append(img_bgr)
                        continue

                    # 缩放到展示尺寸
                    img_resized = cv2.resize(img_bgr, (viz_size, viz_size))

                    K = meta[b]['camera_Intri'][v].cpu().numpy()
                    R = meta[b]['camera_R'][v].cpu().numpy()
                    T = meta[b]['camera_T'][v].cpu().numpy()

                    # K 在 256x256 空间, 投影后坐标需放大到展示尺寸
                    pred_2d = project_3d_to_2d(cur_pred, K, R, T) * viz_scale
                    gt_2d = project_3d_to_2d(cur_gt, K, R, T) * viz_scale

                    r_pred = 4 if is_driverhoi else 3
                    r_gt = 3 if is_driverhoi else 2
                    th_pred = 2 if is_driverhoi else 2
                    th_gt = 1

                    img_vis = draw_skeleton(img_resized, pred_2d, color=(0, 0, 255), radius=r_pred, thickness=th_pred)
                    img_vis = draw_skeleton(img_vis, gt_2d, color=(0, 255, 0), radius=r_gt, thickness=th_gt)

                    reproj_err = np.linalg.norm(pred_2d - gt_2d, axis=-1).mean() / viz_scale  # 误差转回 256 空间
                    font_scale = 1.5 if is_driverhoi else 0.6
                    font_th = 2
                    cv2.putText(img_vis, f"View {v}", (8, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_th)
                    cv2.putText(img_vis, f"Err: {reproj_err:.1f} px", (8, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_th)

                    sample_vis_imgs.append(img_vis)

                if len(sample_vis_imgs) > 0:
                    rows = 2
                    cols = int(math.ceil(V / 2))
                    grid_h = rows * viz_size
                    grid_w = cols * viz_size

                    if is_driverhoi:
                        # DriverHOI: 仅拼接多视角图像, 不绘制 3D 姿态
                        final_canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
                        for idx, img in enumerate(sample_vis_imgs):
                            r = idx // cols
                            c_idx = idx % cols
                            final_canvas[r * viz_size:(r + 1) * viz_size,
                                         c_idx * viz_size:(c_idx + 1) * viz_size] = img
                    else:
                        # DexYCB: 多视角图像 + 3D 姿态可视化
                        w_3d = viz_size * 2
                        h_3d = viz_size * 2
                        img_3d_vis = draw_3d_skeleton_to_image(cur_pred, cur_gt, sample_mpjpe,
                                                               target_size=(w_3d, h_3d))
                        final_canvas = np.zeros((grid_h, grid_w + w_3d, 3), dtype=np.uint8)
                        for idx, img in enumerate(sample_vis_imgs):
                            r = idx // cols
                            c_idx = idx % cols
                            final_canvas[r * viz_size:(r + 1) * viz_size,
                                         c_idx * viz_size:(c_idx + 1) * viz_size] = img
                        final_canvas[0:h_3d, grid_w:grid_w + w_3d] = img_3d_vis

                    save_name = f"batch_{i}_sample_{b}_grid.jpg"
                    cv2.imwrite(os.path.join(viz_save_dir, save_name), final_canvas)

    # -------------------------------------------------------
    # 汇总所有指标
    # -------------------------------------------------------
    avg_mpjpe = total_mpjpe_sum / total_mpjpe_count if total_mpjpe_count > 0 else 0.0
    avg_pa_mpjpe = total_pa_mpjpe_sum / total_pa_mpjpe_count if total_pa_mpjpe_count > 0 else 0.0
    avg_reproj = total_reproj_sum / total_reproj_count if total_reproj_count > 0 else 0.0

    # PCK 曲线
    pck_curve = [pck_correct[thr] / pck_total_joints if pck_total_joints > 0 else 0.0
                 for thr in pck_thresholds]

    # AUC (0~50mm 区间)
    auc_score = compute_auc(pck_curve, pck_thresholds)

    # 特定阈值 PCK
    pck_5  = pck_correct[5]  / pck_total_joints * 100 if pck_total_joints > 0 else 0.0
    pck_10 = pck_correct[10] / pck_total_joints * 100 if pck_total_joints > 0 else 0.0
    pck_20 = pck_correct[20] / pck_total_joints * 100 if pck_total_joints > 0 else 0.0
    pck_30 = pck_correct[30] / pck_total_joints * 100 if pck_total_joints > 0 else 0.0
    pck_50 = pck_correct[50] / pck_total_joints * 100 if pck_total_joints > 0 else 0.0

    # 按手指分组 MPJPE
    finger_avg = {name: np.mean(vals) for name, vals in finger_errors_all.items()}

    # -------------------------------------------------------
    # 打印结果
    # -------------------------------------------------------
    print(f"\n{'=' * 55}")
    print(f"  Evaluation Results  ({cfg.MODEL.NAME} on {cfg.DATASET.NAME})")
    print(f"{'=' * 55}")

    print(f"\n  [3D Metrics]")
    print(f"    MPJPE:        {avg_mpjpe:.2f} mm")
    print(f"    PA-MPJPE:     {avg_pa_mpjpe:.2f} mm")

    print(f"\n  [2D Metrics]")
    print(f"    Reproj. Err:  {avg_reproj:.2f} px")

    print(f"\n  [PCK & AUC]")
    print(f"    PCK@5mm:      {pck_5:.1f}%")
    print(f"    PCK@10mm:     {pck_10:.1f}%")
    print(f"    PCK@20mm:     {pck_20:.1f}%")
    print(f"    PCK@30mm:     {pck_30:.1f}%")
    print(f"    PCK@50mm:     {pck_50:.1f}%")
    print(f"    AUC (0-50mm): {auc_score:.4f}")

    print(f"\n  [Per-Finger MPJPE (mm)]")
    for name in ['Wrist', 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky']:
        print(f"    {name:8s}:     {finger_avg[name]:.2f}")

    print(f"\n{'=' * 55}")

    if cfg.TEST.VIZ:
        print(f"  Visualizations saved to: {viz_save_dir}/")

    # -------------------------------------------------------
    # 保存 PCK 曲线图
    # -------------------------------------------------------
    if cfg.TEST.VIZ:
        os.makedirs(viz_save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(pck_thresholds, [p * 100 for p in pck_curve], 'b-', linewidth=2)
        ax.set_xlabel('Threshold (mm)', fontsize=12)
        ax.set_ylabel('PCK (%)', fontsize=12)
        ax.set_title(f'3D PCK Curve  (AUC={auc_score:.4f})', fontsize=14)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # 标注关键阈值
        for thr, pck_val in [(5, pck_5), (10, pck_10), (20, pck_20), (30, pck_30), (50, pck_50)]:
            ax.axvline(x=thr, color='gray', linestyle='--', alpha=0.5)
            ax.annotate(f'{pck_val:.1f}%', xy=(thr, pck_val), fontsize=9,
                        xytext=(thr + 1.5, pck_val - 5),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

        fig.tight_layout()
        fig.savefig(os.path.join(viz_save_dir, 'pck_curve.png'), dpi=150)
        plt.close(fig)
        print(f"  PCK curve saved to: {viz_save_dir}/pck_curve.png")

    print()


if __name__ == "__main__":
    main()