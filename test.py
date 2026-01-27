import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# 项目模块引用
from config_hand import _C as cfg
from datasets.DexYCB import DEXYCBDatasets
from datasets.DriverHOI import DriverHOIDatasets
from adapter import collate_dexycb_to_mvgformer
from model import get_mvp


# -----------------------------------------------------------------------------
# 1. 可视化工具
# -----------------------------------------------------------------------------
def visualize_batch(img_tensor, pred_uv, gt_uv, batch_idx, save_dir, mpjpe_val):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std + mean).clip(0, 1)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)

    # 画 GT (红色)
    if gt_uv is not None:
        plt.scatter(gt_uv[:, 0], gt_uv[:, 1], s=15, edgecolors='red', facecolors='none', linewidth=1, label='GT')

    # 画 预测 (绿色)
    if pred_uv is not None:
        label = f'Pred ({mpjpe_val:.1f}mm)'
        plt.scatter(pred_uv[:, 0], pred_uv[:, 1], s=5, c='lime', label=label)

        # 连线
        bones = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        u, v = pred_uv[:, 0], pred_uv[:, 1]
        for s, e in bones:
            # 简单的边界保护，防止画线报错
            if -1000 < u[s] < 2000 and -1000 < u[e] < 2000:
                plt.plot([u[s], u[e]], [v[s], v[e]], 'lime', linewidth=0.3)

    plt.legend()
    plt.axis('off')
    title_str = f"Sample {batch_idx} | MPJPE: {mpjpe_val:.1f}mm"
    plt.title(title_str)

    save_path = os.path.join(save_dir, f"vis_batch_{batch_idx}.png")
    plt.savefig(save_path)
    plt.close()


# -----------------------------------------------------------------------------
# 2. 核心测试逻辑 (完全对齐 train.py)
# -----------------------------------------------------------------------------
def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 目录 ---
    save_dir = os.path.join("test_results", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = cfg.DATASET.NAME.lower()
    print(f"Initializing Test Dataset: {dataset_name} ...")

    if dataset_name == 'dexycb':
        root_dir = cfg.DATASET.ROOT_DEXYCB
        # 测试模式：加载未见过的 Subject + 未见过的后4个视角
        test_dataset = DEXYCBDatasets(root_dir=root_dir, split='test')

    elif dataset_name == 'driverhoi':
        root_dir = cfg.DATASET.ROOT_DRIVERHOI
        test_dataset = DriverHOIDatasets(root_dir=root_dir, split='test')

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    print(f"Testing on {len(test_dataset)} samples...")

    # num_workers=0 防止多进程报错
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # --- 模型 ---
    print("Loading Model...")
    # is_train=False 关闭 pretrained 下载
    model = get_mvp(cfg, is_train=False).to(device)

    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'state_dict' in checkpoint:
            # 处理多卡训练的 'module.' 前缀
            state_dict = checkpoint['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    model.eval()

    # 获取 Matcher
    if hasattr(model, 'matcher'):
        matcher = model.matcher
    elif hasattr(model.module, 'matcher'):
        matcher = model.module.matcher
    else:
        raise ValueError("Matcher not found!")

    total_mpjpe = 0.0
    count = 0

    # 这里的 R_fix 仅用于【可视化】，不影响 MPJPE 计算
    # 如果出来的图是倒的或者飞的，可以尝试修改这里
    # 常见的修正: torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]...)
    # 但为了先跑通数值，我们先用单位矩阵，或者依靠 Dataset 自带的 R
    R_fix = torch.eye(3, device=device)

    pbar = tqdm(dataloader, desc="Testing")

    with torch.no_grad():
        for i, (batch_inputs, batch_targets, _) in enumerate(pbar):
            # 1. 适配数据
            views, meta = collate_dexycb_to_mvgformer(batch_inputs, batch_targets, device)

            # 2. 前向传播
            outputs, _ = model(views, meta)

            # 3. 准备数据
            pred_all = outputs['pred_poses']['outputs_coord']
            B = pred_all.shape[0]  # B=1
            pred_reshaped = pred_all.view(B, cfg.DECODER.num_instance, 21, 3)  # (1, N, 21, 3)

            pred_logits = outputs['pred_logits']
            pred_probs = pred_logits.view(B, cfg.DECODER.num_instance, 21, 1).mean(dim=2).sigmoid()

            gt_poses = torch.stack([m['joints_3d'] for m in meta]).squeeze(1).to(device)  # (1, 21, 3)

            # 4. 匹配 (和 Train 保持一致)
            indices = matcher(pred_probs, pred_reshaped, gt_poses)
            src_idx, tgt_idx = indices[0]
            best_idx = src_idx.item()

            best_pred_3d = pred_reshaped[0, best_idx]

            # 5. 计算 3D MPJPE (最核心指标)
            # 这里直接算 3D 距离，完全不依赖投影逻辑，绝对准确
            error = torch.norm(best_pred_3d - gt_poses[0], dim=-1).mean().item() * 1000.0

            # --- [关键修改] 移除所有 valid 检查，强制计入 ---
            total_mpjpe += error
            count += 1

            avg_error = total_mpjpe / count
            pbar.set_postfix({"Avg MPJPE": f"{avg_error:.2f}mm"})

            # 6. 可视化 (仅作为 debug 参考，不影响指标)
            if i % args.vis_freq == 0:
                v_idx = 0
                K = meta[0]['camera'][v_idx]['K']
                R = meta[0]['camera'][v_idx]['R']
                T = meta[0]['camera'][v_idx]['T']

                # 投影公式
                # pred_cam = (R * pred_3d) + T
                pred_cam = torch.matmul(best_pred_3d, R.t()) + T

                z_s = pred_cam[:, 2].clamp(min=0.01)
                pu = pred_cam[:, 0] * K[0, 0] / z_s + K[0, 2]
                pv = pred_cam[:, 1] * K[1, 1] / z_s + K[1, 2]
                pred_uv = torch.stack([pu, pv], dim=-1).cpu().numpy()

                # GT 投影
                gt_cam = torch.matmul(gt_poses[0], R.t()) + T
                gz_s = gt_cam[:, 2].clamp(min=0.01)
                gu = gt_cam[:, 0] * K[0, 0] / gz_s + K[0, 2]
                gv = gt_cam[:, 1] * K[1, 1] / gz_s + K[1, 2]
                gt_uv = torch.stack([gu, gv], dim=-1).cpu().numpy()

                img_tensor = views[0][0]
                visualize_batch(img_tensor, pred_uv, gt_uv, i, save_dir, error)

    print("=" * 50)
    print(f"Test Finished!")
    print(f"Total Samples: {count}")
    # 这里的数值应该和 Train 的 Validation 结果 (8mm左右) 非常接近
    print(f"Final MPJPE: {total_mpjpe / count:.4f} mm")
    print(f"Visualizations saved to: {save_dir}")
    print("=" * 50)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--exp_name', type=str, default='eval_final', help='Result folder name')
    parser.add_argument('--vis_freq', type=int, default=1, help='Visualize frequency')
    args = parser.parse_args()
    test(args)