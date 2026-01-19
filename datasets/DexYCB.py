import os
import collections
import warnings
import yaml
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as standard

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
    def __init__(self, root_dir, split='train'):
        """
        初始化 DexYCB 数据集 (无归一化版本, 适配 256x256 输入)
        """
        super(DEXYCBDatasets, self).__init__()
        self.root_dir = root_dir
        self.calibration_dir = os.path.join(root_dir, 'calibration/calibration')
        self.data_split = split

        # 仅保留 Resize 和 ToTensor，移除 Normalize
        # 这样 img 数据范围是 [0, 1]，方便直接可视化
        self.transform = standard.Compose([
            standard.Resize(cfg.input_img_shape),
            standard.ToTensor(),
        ])

        self.samples = self._collect_samples()
        print(f"Dataset initialized from: {root_dir}")
        print(f"Split: {split}, Collected samples: {len(self.samples)}")

    def _collect_samples(self):
        # -----------------------------------------------------------
        # [控制配置] 在这里硬编码控制加载的被试数量
        # 设置为 None 则加载所有被试
        # 设置为 整数 (如 2) 则只加载前 2 个被试
        MAX_SUBJECTS = 1
        # -----------------------------------------------------------

        samples = []
        global_idx = 0

        # 获取所有可能的被试目录
        subjects = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d != "calibration"
        ])

        # 应用数量限制
        if MAX_SUBJECTS is not None and isinstance(MAX_SUBJECTS, int):
            original_count = len(subjects)
            subjects = subjects[:MAX_SUBJECTS]
            print(
                f"!!! [Dataset Limitation] Restricting to first {len(subjects)} subjects (Original: {original_count}): {subjects}")

        for subject_dir in subjects:
            subject_path = os.path.join(self.root_dir, subject_dir)
            for seq_dir in sorted(os.listdir(subject_path)):
                seq_path = os.path.join(subject_path, seq_dir)
                if not os.path.isdir(seq_path): continue

                count = 0
                for view_dir in sorted(os.listdir(seq_path)):
                    view_path = os.path.join(seq_path, view_dir)
                    if os.path.isdir(view_path):
                        count = sum(1 for f in os.listdir(view_path) if f.endswith('.jpg'))
                        break

                for sample in range(count):
                    samples.append({
                        "subject": subject_dir,
                        "seq": seq_dir,
                        "sample": sample,
                        "idx": global_idx
                    })
                    global_idx += 1
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

        view_dirs = sorted([d for d in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, d))])

        for view_idx in view_dirs:
            view_path = os.path.join(seq_path, view_idx)
            rgb_path = os.path.join(view_path, f'color_{sample_idx:06d}.jpg')

            if not os.path.exists(rgb_path): continue

            # --- 1. 图像处理 ---
            img_pil = Image.open(rgb_path).convert('RGB')
            # 保留原始图像
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

            # --- 4. 核心调整: 仅做 Resize 适配 ---
            # 像素坐标适配 256x256
            joints_uvd[:, 0] *= scale_x
            joints_uvd[:, 1] *= scale_y

            # 内参适配 256x256
            intrinsic[0, 0] *= scale_x
            intrinsic[1, 1] *= scale_y
            intrinsic[0, 2] *= scale_x
            intrinsic[1, 2] *= scale_y

            # --- 5. 存储 ---
            inputs['img'].append(img_np)
            inputs['extrinsic'].append(extrinsic.astype(np.float32))
            inputs['intrinsic'].append(intrinsic.astype(np.float32))

            targets['mesh_pose_uvd'].append(joints_uvd.astype(np.float32))  # 存的是 resize 后的像素坐标
            targets['mesh_pose_xyz'].append(cam_joints_3d.astype(np.float32))  # 存的是绝对相机坐标
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore", category=UserWarning)

    root_path = cfg.root_dir if hasattr(cfg, 'root_dir') else '/home/wk/wk/wk/datasets/DexYCB'
    print(f"Initializing dataset from: {root_path}")
    dataset = DEXYCBDatasets(root_dir=root_path, split='train')

    if len(dataset) > 0:
        idx = min(800, len(dataset) - 1)
        print(f"\nLoading sample index: {idx}")
        inputs, targets, meta_info = dataset[idx]

        view_idx = 0

        # --- 可视化核心修改 ---
        # 1. 直接使用 Network Input Image (256x256)
        # inputs['img'] 形状是 (V, C, H, W)，需要取出第0个视角并转置为 (H, W, C)
        img_vis = inputs['img'][view_idx].transpose(1, 2, 0)

        # 2. 直接使用 2D Keypoints (256x256)
        # 这里的 uvd 已经是乘过 scale_x/y 的，直接对应 img_vis
        uvd = targets['mesh_pose_uvd'][view_idx]

        fig, ax = plt.subplots(figsize=(6, 6))

        # 显示图片 (img_vis 是 float 0-1 范围，imshow 可以直接处理)
        ax.imshow(img_vis)

        # 绘制点
        x = uvd[:, 0]
        y = uvd[:, 1]
        ax.scatter(x, y, c='r', s=20, label='Joints')

        # 绘制骨架
        bones = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

        for start, end in bones:
            if start < len(x) and end < len(x):
                ax.plot([x[start], x[end]], [y[start], y[end]], 'b-', linewidth=1.5)

        ax.set_title(f"View {view_idx} (256x256 Direct Vis)")
        ax.axis('off')
        plt.show()

        # 打印信息验证
        print(f"Image Shape: {img_vis.shape}")
        print(f"Joint Range X: {x.min():.1f} - {x.max():.1f}")
        print(f"Joint Range Y: {y.min():.1f} - {y.max():.1f}")

        # 3D 验证 (世界坐标)
        print("Visualizing World Coordinates...")
        visualize_3d_joints(targets['world_coord'][view_idx])

    else:
        print("Dataset is empty.")