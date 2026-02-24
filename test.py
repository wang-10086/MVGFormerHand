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

    # 1. 绘制 GT (绿色)
    ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], c='g', marker='o', s=15, label='GT')
    for s, e in HAND_SKELETON:
        ax.plot([gt_3d[s, 0], gt_3d[e, 0]],
                [gt_3d[s, 1], gt_3d[e, 1]],
                [gt_3d[s, 2], gt_3d[e, 2]], color='g', linewidth=1.5)

    # [新增] 标注 GT 腕部 (第0个点) 的 3D 坐标
    ax.text(gt_3d[0, 0], gt_3d[0, 1], gt_3d[0, 2],
            f'GT_Root\n({gt_3d[0, 0]:.3f}, {gt_3d[0, 1]:.3f}, {gt_3d[0, 2]:.3f})',
            color='darkgreen', fontsize=8, fontweight='bold')

    # 2. 绘制 预测 (红色)
    ax.scatter(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2], c='r', marker='^', s=15, label='Pred')
    for s, e in HAND_SKELETON:
        ax.plot([pred_3d[s, 0], pred_3d[e, 0]],
                [pred_3d[s, 1], pred_3d[e, 1]],
                [pred_3d[s, 2], pred_3d[e, 2]], color='r', linewidth=1.5)

    # # [新增] 标注 Pred 腕部 (第0个点) 的 3D 坐标
    # ax.text(pred_3d[0, 0], pred_3d[0, 1], pred_3d[0, 2],
    #         f'Pred_Root\n({pred_3d[0, 0]:.3f}, {pred_3d[0, 1]:.3f}, {pred_3d[0, 2]:.3f})',
    #         color='darkred', fontsize=8, fontweight='bold')

    # 3. 固定 3D 比例与视角范围
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

    # 保留坐标刻度，并添加轴标签
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
        num_workers=0
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

    total_mpjpe_sum = 0.0
    total_valid_joints = 0

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

            err_sum, valid_count = compute_mpjpe(pred_poses, gt_poses)
            total_mpjpe_sum += err_sum
            total_valid_joints += valid_count

            if cfg.TEST.VIZ and (i % cfg.TEST.VIZ_FREQ == 0):
                os.makedirs(viz_save_dir, exist_ok=True)
                b = 0
                V = len(views)
                sample_vis_imgs = []

                cur_pred = pred_poses[b].cpu().numpy()
                cur_gt = gt_poses[b].cpu().numpy()

                sample_mpjpe = np.linalg.norm(cur_pred - cur_gt, axis=-1).mean() * 1000.0

                # 处理所有的二维视角图像
                for v in range(V):
                    img_path = meta_batch['img_paths'][v][b]
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        # 遇到读取失败的图像填入黑色占位符
                        img_bgr = np.zeros((net_h, net_w, 3), dtype=np.uint8)
                        sample_vis_imgs.append(img_bgr)
                        continue

                    orig_h, orig_w = img_bgr.shape[:2]
                    scale_x = orig_w / net_w
                    scale_y = orig_h / net_h

                    K = meta[b]['camera_Intri'][v].cpu().numpy()
                    R = meta[b]['camera_R'][v].cpu().numpy()
                    T = meta[b]['camera_T'][v].cpu().numpy()

                    pred_2d = project_3d_to_2d(cur_pred, K, R, T)
                    pred_2d[:, 0] *= scale_x
                    pred_2d[:, 1] *= scale_y
                    img_vis = draw_skeleton(img_bgr, pred_2d, color=(0, 0, 255), radius=6, thickness=3)

                    gt_2d = project_3d_to_2d(cur_gt, K, R, T)
                    gt_2d[:, 0] *= scale_x
                    gt_2d[:, 1] *= scale_y
                    img_vis = draw_skeleton(img_vis, gt_2d, color=(0, 255, 0), radius=4, thickness=2)

                    reproj_err = np.linalg.norm(pred_2d - gt_2d, axis=-1).mean()

                    cv2.putText(img_vis, f"View {v}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(img_vis, f"Err: {reproj_err:.1f} px", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 255), 2)

                    sample_vis_imgs.append(cv2.resize(img_vis, (net_w, net_h)))

                if len(sample_vis_imgs) > 0:
                    # 按照 2 行布局左侧的 2D 图像
                    rows = 2
                    cols = int(math.ceil(V / 2))

                    grid_h = rows * net_h
                    grid_w = cols * net_w

                    # 设定 3D 结果图的尺寸，令其高度与左侧 2 行对齐，呈大的正方形或矩形
                    w_3d = net_w * 2
                    h_3d = net_h * 2

                    # 生成 3D 可视化
                    img_3d_vis = draw_3d_skeleton_to_image(cur_pred, cur_gt, sample_mpjpe, target_size=(w_3d, h_3d))

                    # 创建最终大画布：左边是 2D 网格，右边是 3D 图
                    final_canvas = np.zeros((grid_h, grid_w + w_3d, 3), dtype=np.uint8)

                    # 填入左侧的 2D 图片
                    for idx, img in enumerate(sample_vis_imgs):
                        r = idx // cols
                        c_idx = idx % cols
                        y_start = r * net_h
                        y_end = y_start + net_h
                        x_start = c_idx * net_w
                        x_end = x_start + net_w
                        final_canvas[y_start:y_end, x_start:x_end] = img

                    # 拼接 3D 图像到右侧区域
                    final_canvas[0:h_3d, grid_w:grid_w + w_3d] = img_3d_vis

                    save_name = f"batch_{i}_sample_{b}_grid.jpg"
                    cv2.imwrite(os.path.join(viz_save_dir, save_name), final_canvas)

    avg_mpjpe = total_mpjpe_sum / total_valid_joints if total_valid_joints > 0 else 0.0

    print(f"\n" + "=" * 40)
    print(f"Final Test MPJPE: {avg_mpjpe:.2f} mm")
    if cfg.TEST.VIZ:
        print(f"Visualizations saved to: {viz_save_dir}/")
    print(f"=" * 40)


if __name__ == "__main__":
    main()