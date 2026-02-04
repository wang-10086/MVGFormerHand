import torch
import os
import cv2
import numpy as np
import warnings
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. 导入配置与核心模块
from config import cfg
from adapter import collate_dexycb_to_mvgformer
from train import build_model, compute_mpjpe

# 2. 导入数据集
from datasets.DexYCB import DEXYCBDatasets
from datasets.DriverHOI import DriverHOIDatasets

# -----------------------------------------------------------------------------
# 1. 几何与可视化辅助函数
# -----------------------------------------------------------------------------

# 定义手部骨架连接关系 (21个关键点)
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


def project_3d_to_2d(pts_3d, K, R, T):
    """
    将 3D 点投影到 2D 图像平面。

    Args:
        pts_3d: (N, 3) 3D 坐标
        K: (3, 3) 内参矩阵
        R: (3, 3) 旋转矩阵
        T: (3,) 平移向量

    Returns:
        pts_2d: (N, 2) 2D 像素坐标 (u, v)
    """
    # World -> Camera
    pts_cam = np.dot(pts_3d, R.T) + T[np.newaxis, :]
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

    # 避免深度为 0 或负数导致的除法错误
    z = np.maximum(z, 0.01)

    # Camera -> Pixel
    u = x * K[0, 0] / z + K[0, 2]
    v = y * K[1, 1] / z + K[1, 2]

    return np.stack([u, v], axis=1)


def draw_skeleton(img, pts_2d, color=(0, 0, 255), radius=4, thickness=2):
    """
    在图像上绘制手部骨架。

    Args:
        img: cv2 图像 (BGR)
        pts_2d: (21, 2) 关键点坐标
        color: BGR 颜色
    """
    # 绘制关键点
    for p in pts_2d:
        cv2.circle(img, (int(p[0]), int(p[1])), radius, color, -1)

    # 绘制骨骼连线
    for (s, e) in HAND_SKELETON:
        if s < len(pts_2d) and e < len(pts_2d):
            pt1 = (int(pts_2d[s][0]), int(pts_2d[s][1]))
            pt2 = (int(pts_2d[e][0]), int(pts_2d[e][1]))
            cv2.line(img, pt1, pt2, color, thickness)
    return img


def create_grid_image(images, rows=2, cols=2):
    """
    将多张图片拼接成网格大图。

    Args:
        images: 图片列表
        rows: 行数
        cols: 列数
    """
    if not images:
        return None

    h, w, c = images[0].shape
    grid_h = rows * h
    grid_w = cols * w
    canvas = np.zeros((grid_h, grid_w, c), dtype=np.uint8)

    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break

        r = idx // cols
        c_idx = idx % cols

        y_start = r * h
        y_end = y_start + h
        x_start = c_idx * w
        x_end = x_start + w

        # 简单的尺寸安全检查
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))

        canvas[y_start:y_end, x_start:x_end] = img

    return canvas


# -----------------------------------------------------------------------------
# 2. 主测试逻辑
# -----------------------------------------------------------------------------

def main():
    # 忽略不必要的警告
    warnings.filterwarnings("ignore")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 检查权重路径
    ckpt_path = cfg.TEST.TEST_CKPT
    if not ckpt_path:
        print("[Error] cfg.TEST.TEST_CKPT is empty. Please specify a checkpoint path.")
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Testing Model: {cfg.MODEL.NAME} | Dataset: {cfg.DATASET.NAME}")

    # 1. 加载数据集
    dataset_name = cfg.DATASET.NAME.lower()
    root_dir = cfg.DATASET.ROOT_DEXYCB if dataset_name == 'dexycb' else cfg.DATASET.ROOT_DRIVERHOI
    DatasetClass = DEXYCBDatasets if dataset_name == 'dexycb' else DriverHOIDatasets

    test_dataset = DatasetClass(
        root_dir=root_dir,
        split='test',
        split_strategy=cfg.DATASET.SPLIT_STRATEGY
    )

    # num_workers=0 以避免调试时的多进程问题，生产环境可适当调大
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # 2. 构建模型并加载权重
    model = build_model(cfg, device)

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict']

    # 处理 DataParallel 可能引入的 'module.' 前缀
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # 3. 准备可视化
    viz_save_dir = f"{cfg.TEST.VIZ_DIR}_{cfg.MODEL.NAME}"
    net_h, net_w = cfg.NETWORK.IMAGE_SIZE  # e.g., [256, 256]

    # 指标统计
    total_mpjpe = 0.0
    num_batches = 0

    print(">>> Start Evaluation <<<")
    with torch.no_grad():
        for i, (batch_inputs, batch_targets, meta_batch) in enumerate(tqdm(test_loader, desc="Testing")):

            # A. 数据适配与前向传播
            views, meta = collate_dexycb_to_mvgformer(batch_inputs, batch_targets, device)
            outputs, _ = model(views, meta)

            # B. 提取预测结果 (兼容不同模型的输出格式)
            if 'final_pred_poses' in outputs:
                pred_poses = outputs['final_pred_poses']
            elif 'pred_poses' in outputs and 'outputs_coord' in outputs['pred_poses']:
                pred_poses = outputs['pred_poses']['outputs_coord']
            else:
                raise ValueError("Output format not recognized (missing 'final_pred_poses' or 'pred_poses')")

            # 确保维度一致 (B, 21, 3)
            if pred_poses.dim() == 4:
                pred_poses = pred_poses.squeeze(1)

            # C. 提取 GT
            gt_poses = torch.stack([m['joints_3d'] for m in meta])
            if gt_poses.dim() == 4:
                gt_poses = gt_poses.squeeze(1)
            gt_poses = gt_poses.to(device)

            # D. 计算 MPJPE 指标
            batch_error = compute_mpjpe(pred_poses, gt_poses)
            total_mpjpe += batch_error
            num_batches += 1

            # E. 可视化逻辑
            if cfg.TEST.VIZ and (i % cfg.TEST.VIZ_FREQ == 0):
                os.makedirs(viz_save_dir, exist_ok=True)

                # 仅对 Batch 中的第 0 个样本进行可视化
                b = 0
                V = len(views)  # 视角数量

                sample_vis_imgs = []

                for v in range(V):
                    # 获取原始图像路径 (从 DataLoader 的 meta_batch 中获取)
                    # 注意：meta_batch['img_paths'] 结构通常是 list of lists/tuples
                    img_path = meta_batch['img_paths'][v][b]

                    # 1. 读取原图
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        print(f"[Warning] Failed to read image: {img_path}")
                        continue

                    orig_h, orig_w = img_bgr.shape[:2]

                    # 2. 计算缩放比例 (原图 -> 网络输入尺寸)
                    # 因为投影是在网络输入尺寸对应的内参下进行的，或者需要反算到原图
                    # 这里为了方便，我们把 3D 点投影回原图尺寸，需要注意内参的匹配
                    # 如果 meta 中的 Intri 已经被 adapter 缩放过 (对应 256x256)，
                    # 那么 project_3d_to_2d 得到的坐标是 256x256 下的。
                    # 我们需要将其放大回原图尺寸绘制。

                    scale_x = orig_w / net_w
                    scale_y = orig_h / net_h

                    # 3. 获取相机参数
                    K = meta[b]['camera_Intri'][v].cpu().numpy()
                    R = meta[b]['camera_R'][v].cpu().numpy()
                    T = meta[b]['camera_T'][v].cpu().numpy()

                    # 4. 投影并绘制
                    # 绘制预测 (红色)
                    cur_pred = pred_poses[b].cpu().numpy()
                    pred_2d = project_3d_to_2d(cur_pred, K, R, T)
                    # 坐标缩放回原图
                    pred_2d[:, 0] *= scale_x
                    pred_2d[:, 1] *= scale_y
                    img_vis = draw_skeleton(img_bgr, pred_2d, color=(0, 0, 255), radius=6, thickness=3)

                    # 绘制 GT (绿色)
                    cur_gt = gt_poses[b].cpu().numpy()
                    gt_2d = project_3d_to_2d(cur_gt, K, R, T)
                    # 坐标缩放回原图
                    gt_2d[:, 0] *= scale_x
                    gt_2d[:, 1] *= scale_y
                    img_vis = draw_skeleton(img_vis, gt_2d, color=(0, 255, 0), radius=4, thickness=2)

                    # 标记视角
                    cv2.putText(img_vis, f"View {v}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    sample_vis_imgs.append(img_vis)

                # 5. 拼接多视角大图
                if len(sample_vis_imgs) > 0:
                    # 自动计算网格布局
                    if V == 4:
                        rows, cols = 2, 2
                    elif V == 8:
                        rows, cols = 2, 4
                    else:
                        cols = int(math.ceil(math.sqrt(V)))
                        rows = int(math.ceil(V / cols))

                    final_canvas = create_grid_image(sample_vis_imgs, rows, cols)

                    if final_canvas is not None:
                        save_name = f"batch_{i}_sample_{b}_grid.jpg"
                        cv2.imwrite(os.path.join(viz_save_dir, save_name), final_canvas)

    # 计算全局平均 MPJPE
    avg_mpjpe = total_mpjpe / num_batches if num_batches > 0 else 0.0

    print(f"\n" + "=" * 40)
    print(f"Final Test MPJPE: {avg_mpjpe:.2f} mm")
    if cfg.TEST.VIZ:
        print(f"Visualizations saved to: {viz_save_dir}/")
    print(f"=" * 40)


if __name__ == "__main__":
    main()