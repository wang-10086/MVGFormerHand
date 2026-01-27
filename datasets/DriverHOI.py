import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import cv2


class DriverHOIDatasets(Dataset):
    """
    适配 MVGFormer 的 DriverHOI 数据集加载器
    包含左右手自动选择逻辑
    """

    def __init__(self,
                 root_dir: str,
                 split: str = 'train',  # 'train' or 'test'
                 camera_views: Optional[List[str]] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 **kwargs):

        self.data_root = Path(root_dir)
        self.split = split
        self.target_size = target_size

        # 默认四视角
        self.available_views = ['MBP25030012', 'MBP25030014', 'MBP25030016', 'MBP25030017']
        self.camera_views = camera_views if camera_views is not None else self.available_views

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 加载相机参数
        self.camera_intrinsics, self.camera_extrinsics = self._load_calibration_files(self.data_root / "calibration")

        # 1. 获取所有 Subject
        all_subjects = sorted(
            [d.name for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("subject")])

        # 2. [核心修改] 根据 split 参数划分 Subject
        # 确定性划分：按名称排序，前 ~85% 为训练集，后 ~15% 为测试集
        # 例如 10 个 subject: train=[0..8], test=[9]
        total_subj = len(all_subjects)
        if total_subj > 0:
            split_idx = int(0.85 * total_subj)
            # 确保至少留一个给测试集 (如果总数大于1)
            if split_idx == total_subj and total_subj > 1:
                split_idx = total_subj - 1

            if self.split == 'train':
                self.subjects = all_subjects[:split_idx]
            else:
                self.subjects = all_subjects[split_idx:]
        else:
            self.subjects = []

        # 3. 构建索引
        self.data_index = self._build_data_index()

        # 定义需要用右手的 Device ID 集合
        self.right_hand_devices = set(range(1, 18)) | set(range(26, 30))

        print(f"[{self.split.upper()}] DriverHOI Dataset Initialized")
        print(f"  Root: {self.data_root}")
        print(f"  Subjects ({len(self.subjects)}): {self.subjects}")
        print(f"  Total Frames: {len(self.data_index)}")

    def _load_calibration_files(self, calib_dir: Path):
        intri_path = calib_dir / "intri.yml"
        extri_path = calib_dir / "extri.yml"
        intrinsics, extrinsics = {}, {}

        if intri_path.exists():
            fs = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
            names_node = fs.getNode("names")
            cam_names = [names_node.at(i).string() for i in
                         range(int(names_node.size()))] if not names_node.empty() else []
            for cam in cam_names:
                K = fs.getNode(f"K_{cam}").mat()
                D = fs.getNode(f"dist_{cam}").mat()
                if K is not None:
                    intrinsics[cam] = {"K": np.array(K, dtype=np.float32).reshape(3, 3),
                                       "D": D.flatten().astype(np.float32) if D is not None else np.zeros(5,
                                                                                                          np.float32)}
            fs.release()

        if extri_path.exists():
            fs = cv2.FileStorage(str(extri_path), cv2.FILE_STORAGE_READ)
            names_node = fs.getNode("names")
            cam_names = [names_node.at(i).string() for i in
                         range(int(names_node.size()))] if not names_node.empty() else []
            for cam in cam_names:
                Rm = fs.getNode(f"Rot_{cam}").mat()
                rvec = fs.getNode(f"R_{cam}").mat()
                T = fs.getNode(f"T_{cam}").mat()
                R = np.eye(3, dtype=np.float32)
                if Rm is not None:
                    R = Rm.astype(np.float32).reshape(3, 3)
                elif rvec is not None:
                    R, _ = cv2.Rodrigues(rvec)
                Tv = np.zeros(3, dtype=np.float32)
                if T is not None: Tv = T.astype(np.float32).flatten()[:3]
                extrinsics[cam] = {"R": R, "T": Tv}
            fs.release()
        return intrinsics, extrinsics

    def _build_data_index(self) -> List[Dict]:
        data_index = []
        for subj in self.subjects:
            subj_path = self.data_root / subj
            for act_dir in [d for d in subj_path.iterdir() if d.is_dir()]:
                action_name = act_dir.name

                for dev_dir in [d for d in act_dir.iterdir() if d.is_dir() and d.name.startswith("device")]:
                    try:
                        device_id = int(dev_dir.name.replace("device", "").replace("_", ""))
                    except:
                        continue

                    json_dir = dev_dir / "post_json"
                    base_view = self.camera_views[0]
                    base_view_path = dev_dir / base_view
                    if not base_view_path.exists(): continue

                    img_names = sorted(
                        [f.name for f in base_view_path.iterdir() if f.suffix.lower() in ['.jpg', '.png']])

                    # 降采样
                    img_names = img_names[::5]

                    for img_name in img_names:
                        frame_stem = Path(img_name).stem
                        view_paths = {}
                        valid_frame = True
                        for view in self.camera_views:
                            p = dev_dir / view / img_name
                            if not p.exists():
                                valid_frame = False
                                break
                            view_paths[view] = p

                        if not valid_frame: continue

                        json_path = json_dir / (frame_stem + ".json")

                        data_index.append({
                            "view_paths": view_paths,
                            "json_path": json_path,
                            "subject": subj,
                            "action": action_name,  # 动作
                            "device_id": device_id,  # 设备ID
                            "frame_id": frame_stem
                        })
        return data_index

    def _load_keypoints(self, json_path: Path, action: str, device_id: int):
        """
        根据动作和设备ID，选择加载左手或右手。
        仅选择，不做任何翻转。
        """
        if not json_path.exists():
            return None

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            kp_data = data.get("keypoints3d", {})

            # 1. 获取原始数据
            right_hand = np.array(kp_data.get("RightHand", []), dtype=np.float32)
            left_hand = np.array(kp_data.get("LeftHand", []), dtype=np.float32)

            # 2. 判定目标手
            target_is_right = True  # 默认为右手

            if action in ['push', 'swing']:
                # 规则1: push/swing 始终是右手
                target_is_right = True
            elif action in ['point', 'press']:
                # 规则2: point/press 根据设备ID判断
                if device_id in self.right_hand_devices:
                    target_is_right = True
                else:
                    target_is_right = False

            # 3. 选择数据
            hand = None
            if target_is_right:
                if right_hand.shape == (21, 3):
                    hand = right_hand
            else:
                if left_hand.shape == (21, 3):
                    hand = left_hand

            # 如果没找到数据，返回空
            if hand is None:
                return None

            # 4. 坐标轴变换 [2, 0, 1] (保持 Z, X, Y 轴对齐)
            hand = hand[..., [2, 0, 1]]

            return torch.from_numpy(hand)

        except Exception:
            return None

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        item = self.data_index[idx]

        # 根据逻辑加载对应的手（不翻转）
        joints_3d = self._load_keypoints(
            item['json_path'],
            item['action'],
            item['device_id']
        )

        if joints_3d is None:
            joints_3d = torch.zeros((21, 3), dtype=torch.float32)
            valid_joints = 0.0
        else:
            valid_joints = 1.0

        list_imgs = []
        list_intrinsics = []
        list_extrinsics = []
        list_world_coords = []

        for view_name in self.camera_views:
            img_path = item['view_paths'][view_name]

            try:
                img_pil = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img_pil.size
                img_tensor = self.transform(img_pil)
            except:
                img_tensor = torch.zeros((3, self.target_size[0], self.target_size[1]))
                orig_w, orig_h = self.target_size

            # Load Params
            K_raw = self.camera_intrinsics.get(view_name, {}).get('K', np.eye(3)).copy()
            extri = self.camera_extrinsics.get(view_name, {})
            R_data = extri.get('R', np.eye(3))
            T_data = extri.get('T', np.zeros(3))

            # K Matrix Scaling
            target_h, target_w = self.target_size
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            K_raw[0, 0] *= scale_x
            K_raw[0, 2] *= scale_x
            K_raw[1, 1] *= scale_y
            K_raw[1, 2] *= scale_y

            ext_4x4 = np.eye(4, dtype=np.float32)
            ext_4x4[:3, :3] = R_data
            ext_4x4[:3, 3] = T_data

            list_imgs.append(img_tensor)
            list_intrinsics.append(torch.from_numpy(K_raw).float())
            list_extrinsics.append(torch.from_numpy(ext_4x4).float())
            list_world_coords.append(joints_3d)

        inputs = {
            'img': torch.stack(list_imgs, dim=0),
            'intrinsic': torch.stack(list_intrinsics, dim=0),
            'extrinsic': torch.stack(list_extrinsics, dim=0)
        }

        targets = {
            'world_coord': torch.stack(list_world_coords, dim=0),
            'intrinsic': torch.stack(list_intrinsics, dim=0),
            'extrinsic': torch.stack(list_extrinsics, dim=0),
        }

        meta_info = {
            'subject': item['subject'],
            'sample_idx': idx,
            'frame_id': item['frame_id']
        }

        return inputs, targets, meta_info


# =============================================================================
# 单元测试与可视化主函数 (适配 Adapter 格式: inputs, targets, meta)
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import warnings
    import torch
    import numpy as np

    # 忽略警告，保持输出清爽
    warnings.filterwarnings("ignore")

    print(">>> 正在启动 DriverHOI 数据集自检程序 (Adapter适配版)...")

    # 1. 配置路径 (请修改为你自己的实际路径)
    DATA_ROOT = "/home/wk/wk/wk/datasets/DriverHOI3D"

    # 2. 初始化数据集
    try:
        # 注意: 这里使用 train split，会加载所有 subjects
        dataset = DriverHOIDatasets(root_dir=DATA_ROOT, split='train', target_size=(256, 256))
    except Exception as e:
        print(f"数据集初始化失败: {e}")
        exit()

    if len(dataset) == 0:
        print("数据集为空！请检查路径。")
        exit()

    # 3. 随机寻找一个有效样本 (包含有效 3D 关键点)
    idx = 0
    found_valid = False

    # 为了演示，我们遍历一些样本直到找到有数据的
    print("正在寻找包含有效骨架的样本...")
    for i in range(0, len(dataset), 50):  # 步长50快速搜索
        inputs, targets, meta = dataset[i]

        # 检查 validity: 看 world_coord 是否全为0
        # targets['world_coord'] shape: (V, 21, 3)
        # 取第一个视角的第一帧判断
        if targets['world_coord'][0].abs().sum() > 1e-6:
            idx = i
            found_valid = True
            break

    if not found_valid:
        print("警告: 未找到包含有效 3D 关键点的样本，将展示第0个样本（可能为空）。")
        idx = 0

    # 4. 加载目标样本
    inputs, targets, meta = dataset[idx]

    print(f"\n>>> 加载样本 ID: {idx}")
    print(f"    Subject: {meta.get('subject', 'N/A')}")
    print(f"    Frame:   {meta.get('frame_id', 'N/A')}")

    # 打印数据形状以供检查
    imgs = inputs['img']  # (V, 3, H, W)
    joints_all = targets['world_coord']  # (V, 21, 3)
    Ks = targets['intrinsic']  # (V, 3, 3)
    Exts = targets['extrinsic']  # (V, 4, 4)

    print(f"    Images Shape: {imgs.shape}")
    print(f"    Joints Shape: {joints_all.shape}")
    print(f"    Intrinsics Shape: {Ks.shape}")

    # 5. 可视化所有视角
    num_views = len(imgs)
    fig, axes = plt.subplots(1, num_views, figsize=(5 * num_views, 5))
    if num_views == 1: axes = [axes]

    # 定义手部骨架连接
    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
    ]

    for v_idx in range(num_views):
        ax = axes[v_idx]

        # --- A. 图像反归一化 (Tensor -> Numpy) ---
        img_tensor = imgs[v_idx]
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        img_np = (img_tensor.numpy() * std + mean).transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)

        ax.imshow(img_np)
        h, w = img_np.shape[:2]
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # 图像坐标系

        # --- B. 投影 3D 关键点 ---
        # 提取参数
        K = Ks[v_idx]  # (3, 3)
        ext = Exts[v_idx]  # (4, 4)
        joints_3d = joints_all[v_idx]  # (21, 3)

        # 检查是否全0 (无效数据)
        if joints_3d.abs().sum() < 1e-6:
            ax.text(10, 20, "Invalid/No 3D Data", color='white', backgroundcolor='red')
            ax.set_title(f"View {v_idx}: No Data")
            ax.axis('off')
            continue

        # 提取 R, T
        R = ext[:3, :3]
        T = ext[:3, 3]

        # 1. World -> Camera: P_cam = P_world @ R^T + T
        cam_points = torch.matmul(joints_3d, R.t()) + T

        # 2. Camera -> Image: u = fx * X/Z + cx
        z_safe = cam_points[:, 2].clamp(min=0.01)
        u = cam_points[:, 0] * K[0, 0] / z_safe + K[0, 2]
        v = cam_points[:, 1] * K[1, 1] / z_safe + K[1, 2]

        uv_np = torch.stack([u, v], dim=1).numpy()

        # --- C. 绘图 ---
        ax.scatter(uv_np[:, 0], uv_np[:, 1], s=20, c='red', label='Proj 3D')

        for start, end in bones:
            # 简单的边界检查，防止画到图外太远
            if (0 <= uv_np[start, 0] < w * 1.5) and (0 <= uv_np[start, 1] < h * 1.5):
                ax.plot([uv_np[start, 0], uv_np[end, 0]],
                        [uv_np[start, 1], uv_np[end, 1]],
                        color='lime', linewidth=1)

        ax.set_title(f"View {v_idx}")
        ax.axis('off')

    plt.tight_layout()
    save_path = "dataset_verify_adapter_ready.jpg"
    plt.savefig(save_path)