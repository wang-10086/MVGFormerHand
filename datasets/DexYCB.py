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

        # 定义标准 ImageNet 归一化参数
        self.normalize = standard.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 仅保留 Resize 和 ToTensor，移除 Normalize
        # 这样 img 数据范围是 [0, 1]，方便直接可视化
        self.transform = standard.Compose([
            standard.Resize(cfg.input_img_shape),
            standard.ToTensor(),
            self.normalize,
        ])

        self.samples = self._collect_samples()
        print(f"Dataset initialized from: {root_dir}")
        print(f"Split: {split}, Collected samples: {len(self.samples)}")

    def _collect_samples(self):
        # -----------------------------------------------------------
        # [控制配置] 硬编码调试参数
        # 1. 限制加载的被试数量 (None 表示全部)
        MAX_SUBJECTS = 2

        # 2. [新增] 限制每个被试下的序列数量 (None 表示全部)
        # 例如设置为 1，则每个被试只加载它的第 1 个视频序列
        MAX_SEQUENCES_PER_SUBJECT = 20
        # -----------------------------------------------------------

        samples = []
        global_idx = 0

        # --- 1. 获取并限制被试 ---
        subjects = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and d != "calibration"
        ])

        if MAX_SUBJECTS is not None and isinstance(MAX_SUBJECTS, int):
            original_cnt = len(subjects)
            subjects = subjects[:MAX_SUBJECTS]
            print(f"!!! [Limit] Subjects restricted: {len(subjects)}/{original_cnt}")

        for subject_dir in subjects:
            subject_path = os.path.join(self.root_dir, subject_dir)

            # --- 2. 获取并限制序列 ---
            # 先过滤出所有有效的序列目录
            all_seqs = sorted([
                d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d))
            ])

            # 应用序列数量限制
            if MAX_SEQUENCES_PER_SUBJECT is not None and isinstance(MAX_SEQUENCES_PER_SUBJECT, int):
                target_seqs = all_seqs[:MAX_SEQUENCES_PER_SUBJECT]
            else:
                target_seqs = all_seqs

            # print(f"  - Subject {subject_dir}: Loading {len(target_seqs)} sequences")

            for seq_dir in target_seqs:
                seq_path = os.path.join(subject_path, seq_dir)

                # 计算该序列下的帧数 (通过任意一个视角文件夹判断)
                count = 0
                # 遍历所有视角文件夹找到一个存在的来计算帧数
                possible_views = sorted(os.listdir(seq_path))
                for view_dir in possible_views:
                    view_path = os.path.join(seq_path, view_dir)
                    if os.path.isdir(view_path):
                        # 简单统计 jpg 数量
                        count = sum(1 for f in os.listdir(view_path) if f.endswith('.jpg'))
                        if count > 0:
                            break

                # 添加样本
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

        # --- 可视化 ---
        img_vis = inputs['img'][view_idx].transpose(1, 2, 0)
        uvd = targets['mesh_pose_uvd'][view_idx]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_vis)

        x = uvd[:, 0]
        y = uvd[:, 1]
        ax.scatter(x, y, c='r', s=20, label='Joints')

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

        print(f"Image Shape: {img_vis.shape}")
        print("Visualizing World Coordinates...")
        visualize_3d_joints(targets['world_coord'][view_idx])

    else:
        print("Dataset is empty.")