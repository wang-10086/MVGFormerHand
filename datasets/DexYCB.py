import os
import collections
import warnings
import yaml
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as standard
import random

# 本地模块引用
from datasets.datasets_utils import (
    get_camera_color_intrinsics,
    construct_projection_matrix,
    camera2world,
    xyz2uvd,
    visualize_3d_joints
)
from config import cfg


class DEXYCBDatasets(Dataset):
    def __init__(self, root_dir, split='train', split_strategy='subject'):
        super(DEXYCBDatasets, self).__init__()
        self.root_dir = root_dir
        self.calibration_dir = os.path.join(root_dir, 'calibration/calibration')
        self.data_split = split
        self.split_strategy = split_strategy

        # ... transforms ...
        self.transform = standard.Compose([
            standard.Resize(cfg.NETWORK.IMAGE_SIZE if hasattr(cfg, 'NETWORK') else (256, 256)),
            standard.ToTensor(),
            standard.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 收集样本
        self.samples = self._collect_samples()

        print(f"[{split.upper()}] DexYCB Dataset initialized")
        print(f"  Strategy: {self.split_strategy}")
        print(f"  Samples: {len(self.samples)}")

    def _collect_samples(self):
        # 1. 获取所有 Subject
        all_subjects = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d != "calibration"
        ])

        # ---------------------------------------------------------------------
        # [核心修改] 划分逻辑
        # ---------------------------------------------------------------------
        target_subjects = []

        if self.split_strategy == 'subject':
            # --- 策略 A: 按被试划分 ---
            total_subj = len(all_subjects)
            if total_subj > 0:
                split_idx = int(0.85 * total_subj)
                if split_idx == total_subj and total_subj > 1: split_idx = total_subj - 1
                if self.data_split == 'train':
                    target_subjects = all_subjects[:split_idx]
                else:
                    target_subjects = all_subjects[split_idx:]

            # 收集这些被试的所有样本
            final_samples = self._scan_subjects(target_subjects)

        elif self.split_strategy == 'random':
            # --- 策略 B: 随机混合划分 ---
            # 1. 先收集所有人的所有样本
            all_samples = self._scan_subjects(all_subjects)

            # 2. 固定随机打乱
            rnd = random.Random(42)
            rnd.shuffle(all_samples)

            # 3. 切分
            split_point = int(0.9 * len(all_samples))
            if self.data_split == 'train':
                final_samples = all_samples[:split_point]
            else:
                final_samples = all_samples[split_point:]

        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

        # 重建索引 idx
        for i, s in enumerate(final_samples):
            s['idx'] = i

        return final_samples

    def _scan_subjects(self, subject_list):
        """辅助函数：扫描指定 Subject 列表下的所有样本"""
        samples = []
        global_idx = 0  # 临时 ID
        MAX_SEQ = None

        for subject_dir in subject_list:
            subject_path = os.path.join(self.root_dir, subject_dir)
            all_seqs = sorted([d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))])
            if MAX_SEQ: all_seqs = all_seqs[:MAX_SEQ]

            for seq_dir in all_seqs:
                seq_path = os.path.join(subject_path, seq_dir)
                count = 0
                # 寻找任一有效视角计算帧数
                possible_views = sorted(os.listdir(seq_path))
                for v in possible_views:
                    vp = os.path.join(seq_path, v)
                    if os.path.isdir(vp):
                        count = sum(1 for f in os.listdir(vp) if f.endswith('.jpg'))
                        if count > 0: break

                for sample in range(count):
                    samples.append({
                        "subject": subject_dir,
                        "seq": seq_dir,
                        "sample": sample,
                        "idx": 0  # 稍后重置
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs = collections.defaultdict(list)
        targets = collections.defaultdict(list)
        meta_info = collections.defaultdict(list)

        sample_info = self.samples[idx]
        seq_path = os.path.join(self.root_dir, sample_info["subject"], sample_info["seq"])
        sample_idx = sample_info["sample"]

        # 读取元数据和外参配置
        meta_path = os.path.join(seq_path, 'meta.yml')
        with open(meta_path, 'r') as file:
            meta_data = yaml.load(file, Loader=yaml.FullLoader)

        ext_id = meta_data["extrinsics"]
        extrinsic_path = os.path.join(self.calibration_dir, f'extrinsics_{ext_id}/extrinsics.yml')
        with open(extrinsic_path, 'r') as file:
            extrinsic_config = yaml.load(file, Loader=yaml.FullLoader)["extrinsics"]

        all_view_dirs = sorted([d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))])

        # if self.data_split == 'train':
        #     target_view_dirs = all_view_dirs[:4]
        # else:
        #     # 如果是 test，取后 4 个
        #     # 注意：要确保视角总数够 8 个，不够的话代码会取剩下的所有
        #     target_view_dirs = all_view_dirs[4:]
        #
        #     # 兜底：万一数据有问题导致不足 4 个，就取全部，避免报错
        #     if len(target_view_dirs) == 0:
        #         target_view_dirs = all_view_dirs

        # if self.data_split == 'train':
        #     # 训练：取偶数视角 [0, 2, 4, 6]
        #     target_view_dirs = [all_view_dirs[i] for i in range(0, 8, 2)]
        # else:
        #     # 测试：取奇数视角 [1, 3, 5, 7]
        #     target_view_dirs = [all_view_dirs[i] for i in range(1, 8, 2)]
        #
        #     # 兜底
        #     if len(target_view_dirs) == 0:
        #         target_view_dirs = all_view_dirs

        target_view_dirs = all_view_dirs

        for view_idx in target_view_dirs:
            view_path = os.path.join(seq_path, view_idx)
            rgb_path = os.path.join(view_path, f'color_{sample_idx:06d}.jpg')

            if not os.path.exists(rgb_path): continue

            # --- 1. 图像处理 ---
            img_pil = Image.open(rgb_path).convert('RGB')
            ori_img = np.array(img_pil)
            orig_width, orig_height = img_pil.size

            # 变换得到 256x256 的 Tensor (C, H, W)
            img_tensor = self.transform(img_pil)
            img_np = np.array(img_tensor)

            # 计算缩放因子
            scale_x = cfg.input_img_shape[0] / orig_width
            scale_y = cfg.input_img_shape[1] / orig_height

            # --- 2. 参数读取 ---
            intrinsic_path = os.path.join(self.calibration_dir, f'intrinsics/{view_idx}_640x480.yml')
            intrinsic_data = get_camera_color_intrinsics(intrinsic_path)
            extrinsic_data = extrinsic_config[view_idx]

            proj_matrix, intrinsic, extrinsic = construct_projection_matrix(intrinsic_data, extrinsic_data)

            # --- 3. 标签与坐标变换 ---
            pose_path = os.path.join(view_path, f'labels_{sample_idx:06d}.npz')
            pose_data = np.load(pose_path)
            cam_joints_3d = pose_data['joint_3d'].squeeze(0)  # [21, 3]

            world_joints_3d = camera2world(cam_joints_3d, extrinsic)
            joints_uvd = xyz2uvd(cam_joints_3d, intrinsic)

            # --- 4. 核心调整: Resize 适配 ---
            joints_uvd[:, 0] *= scale_x
            joints_uvd[:, 1] *= scale_y

            intrinsic[0, 0] *= scale_x
            intrinsic[1, 1] *= scale_y
            intrinsic[0, 2] *= scale_x
            intrinsic[1, 2] *= scale_y

            # --- 5. 存储 ---
            inputs['img'].append(img_np)
            inputs['extrinsic'].append(extrinsic.astype(np.float32))
            inputs['intrinsic'].append(intrinsic.astype(np.float32))

            targets['mesh_pose_uvd'].append(joints_uvd.astype(np.float32))
            targets['mesh_pose_xyz'].append(cam_joints_3d.astype(np.float32))
            targets['intrinsic'].append(intrinsic.astype(np.float32))
            targets['extrinsic'].append(extrinsic.astype(np.float32))
            targets['world_coord'].append(world_joints_3d.astype(np.float32))
            targets['proj_matrix'].append(proj_matrix.astype(np.float32))
            targets['ori_img'].append(ori_img)

            meta_info['subject'].append(sample_info["subject"])
            meta_info['seq'].append(sample_info["seq"])
            meta_info['sample_idx'].append(sample_idx)
            meta_info['view_idx'].append(view_idx)

        final_inputs = {k: np.stack(v, axis=0) for k, v in inputs.items()}
        final_targets = {}
        for k, v in targets.items():
            arr = np.stack(v, axis=0)
            if k == 'ori_img':
                final_targets[k] = arr
            else:
                final_targets[k] = np.float32(arr)

        return final_inputs, final_targets, meta_info

    def print_gt_stats(self, sample_stride=50):
        """
        [工具函数] 统计 GT 3D 坐标分布，并在控制台打印建议的 SPACE_CENTER 和 SPACE_SIZE。

        Args:
            sample_stride (int): 采样步长。默认为 50，即每 50 个样本统计一次。
                                 既能保证速度，又具有统计学意义。
        """
        import torch
        print(f"\n>>> 正在扫描数据集以计算 GT 统计信息 (每 {sample_stride} 个样本采样一次)...")

        all_x, all_y, all_z = [], [], []

        # 遍历数据集索引
        for i in range(0, len(self), sample_stride):
            # 直接调用内部获取数据的逻辑
            # 注意：这里我们调用 __getitem__，它通常返回 (img, target, ...)
            data = self.__getitem__(i)

            target = None

            # --- 自动寻找包含坐标的字典 ---
            if isinstance(data, dict):
                target = data
            elif isinstance(data, (list, tuple)):
                # 如果返回的是 tuple，遍历寻找含有 'world_coord' 的字典
                for item in data:
                    if isinstance(item, dict) and 'world_coord' in item:
                        target = item
                        break

            if target is None or 'world_coord' not in target:
                continue

            # 提取坐标
            # world_coord 通常是 Tensor 或 Numpy
            world_coord = target['world_coord']

            # 转为 Tensor 并拍平
            if not isinstance(world_coord, torch.Tensor):
                coords = torch.from_numpy(world_coord).float()
            else:
                coords = world_coord.float()

            coords = coords.view(-1, 3)  # (N_points, 3)

            all_x.append(coords[:, 0])
            all_y.append(coords[:, 1])
            all_z.append(coords[:, 2])

        if not all_x:
            print("错误: 未找到任何有效的 world_coord 数据！")
            return

        # 拼接所有采样点
        all_x = torch.cat(all_x)
        all_y = torch.cat(all_y)
        all_z = torch.cat(all_z)

        # 计算统计量
        min_x, max_x, mean_x = all_x.min().item(), all_x.max().item(), all_x.mean().item()
        min_y, max_y, mean_y = all_y.min().item(), all_y.max().item(), all_y.mean().item()
        min_z, max_z, mean_z = all_z.min().item(), all_z.max().item(), all_z.mean().item()

        # 计算建议的配置
        center = [mean_x, mean_y, mean_z]

        # 计算跨度
        span_x = max_x - min_x
        span_y = max_y - min_y
        span_z = max_z - min_z

        # 建议尺寸：最大跨度 * 1.2 (留 20% 余量)
        size_val = max(span_x, span_y, span_z) * 1.2

        print("=" * 50)
        print(f" [数据集 GT 统计结果]")
        print(f"  X 轴: 范围 [{min_x:.3f}, {max_x:.3f}], 均值 {mean_x:.3f}")
        print(f"  Y 轴: 范围 [{min_y:.3f}, {max_y:.3f}], 均值 {mean_y:.3f}")
        print(f"  Z 轴: 范围 [{min_z:.3f}, {max_z:.3f}], 均值 {mean_z:.3f}")
        print("-" * 50)
        print(f" [建议修改 Config.py]")
        print(f"  SPACE_CENTER = [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
        print(f"  SPACE_SIZE   = [{size_val:.4f}, {size_val:.4f}, {size_val:.4f}]")
        print("=" * 50)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2  # 如果没有 cv2，可以用 PIL 或 plt 代替

    # 忽略警告，保持输出清爽
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1. 初始化数据集
    # ------------------------------------------------------------------
    root_path = cfg.root_dir if hasattr(cfg, 'root_dir') else '/home/wk/wk/wk/datasets/DexYCB'
    print(f"Initializing dataset from: {root_path}")
    dataset = DEXYCBDatasets(root_dir=root_path, split='train')

    if len(dataset) > 0:
        # 2. 随机取一个样本 (建议取稍微靠后一点的，避开可能的空数据)
        idx = min(100, len(dataset) - 1)
        print(f"\nLoading sample index: {idx}")

        # 获取由 Dataset __getitem__ 返回的数据
        # inputs['img']: (V, 3, 256, 256)
        # targets['world_coord']: (V, 21, 3)
        # targets['extrinsic']: (V, 4, 4) 或 (V, 3, 4) -- 这里通常是 Camera Pose (C2W)
        inputs, targets, meta_info = dataset[idx]

        # 选择第 0 个视角进行验证
        view_idx = 0

        # ------------------------------------------------------------------
        # 3. 提取核心数据
        # ------------------------------------------------------------------
        # 图像 (C, H, W) -> (H, W, C)
        img_vis = inputs['img'][view_idx].transpose(1, 2, 0).copy()  # copy 用于画图

        # 3D 世界坐标 (21, 3)
        world_points = targets['world_coord'][view_idx]

        # 相机内参 (3, 3)
        K = targets['intrinsic'][view_idx]

        # 相机外参 (通常是 Camera-to-World, 需验证)
        extrinsic_c2w = targets['extrinsic'][view_idx]

        # 数据集自带的 2D 标签 (作为 Ground Truth 参考)
        gt_uv = targets['mesh_pose_uvd'][view_idx]

        # ------------------------------------------------------------------
        # 4. 核心验证：手动投影 (World -> Camera -> Image)
        # ------------------------------------------------------------------
        print(f"Extrinsic Shape: {extrinsic_c2w.shape}")

        # [步骤 A] 处理外参：确保是 4x4 矩阵
        if extrinsic_c2w.shape == (3, 4):
            ext_4x4 = np.eye(4)
            ext_4x4[:3, :] = extrinsic_c2w
        else:
            ext_4x4 = extrinsic_c2w

        # # [步骤 B] 外参矩阵：本身就是World-to-Camera不需要求逆
        w2c_matrix = ext_4x4

        R_w2c = w2c_matrix[:3, :3]
        T_w2c = w2c_matrix[:3, 3]

        # [步骤 C] 坐标变换: World -> Camera
        # P_cam = R * P_world + T
        # world_points: (21, 3)
        camera_points = np.dot(world_points, R_w2c.T) + T_w2c

        # [步骤 D] 透视投影: Camera -> Pixel
        # u = fx * x / z + cx
        # v = fy * y / z + cy
        # 或者 P_pixel = K @ P_cam
        # (21, 3) @ (3, 3).T -> (21, 3)
        pixel_coords_homo = np.dot(camera_points, K.T)

        # 归一化 (除以 Z)
        projected_uv = pixel_coords_homo[:, :2] / pixel_coords_homo[:, 2:3]

        # ------------------------------------------------------------------
        # 5. 可视化对比
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 10))

        # 显示图片
        # 如果 Dataset 里做了 ImageNet Normalize，这里图片可能会很黑/奇怪，但不影响点的位置
        ax.imshow(img_vis)

        # A. 画数据集自带的 GT 2D 点 (红色空心圆)
        ax.scatter(gt_uv[:, 0], gt_uv[:, 1], s=80, edgecolors='red', facecolors='none', linewidths=2,
                   label='Dataset GT (uvd)')

        # B. 画我们要验证的 手动投影点 (绿色实心点)
        ax.scatter(projected_uv[:, 0], projected_uv[:, 1], s=20, c='green', label='Manual Proj (World->Cam->Img)')

        # 连线 (骨架) 方便观察
        bones = [
            (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        # 画绿色的骨架 (手动投影)
        x_proj, y_proj = projected_uv[:, 0], projected_uv[:, 1]
        for start, end in bones:
            ax.plot([x_proj[start], x_proj[end]], [y_proj[start], y_proj[end]], 'g-', linewidth=1, alpha=0.7)

        ax.legend()
        ax.set_title(f"Projection Verification (View {view_idx})\nRed Circle: GT, Green Dot: Your Calculation")
        ax.axis('off')

        plt.show()

    else:
        print("Dataset is empty.")