import os
import yaml
import torch
import warnings
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as standard
from datasets.datasets_utils import get_camera_color_intrinsics, construct_projection_matrix, camera2world, xyz2uvd, visualize_3d_joints
from config import cfg
import collections


class DEXYCBDatasets(Dataset):
    def __init__(self, root_dir, split='train'):
        """
        初始化DexYCB数据集
        参数:
            root_dir (str): 数据集根目录路径
            split (str): 数据集分割('train'或'test')
        """
        super(DEXYCBDatasets, self).__init__()
        self.root_dir = root_dir
        self.calibration_dir = os.path.join(root_dir, 'calibration/calibration')
        self.data_split = split
        # self.select_view_idx = cfg.view_idx  # 从配置获取要使用的视角索引

        # 设置图像标准化参数(ImageNet均值和标准差)
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = standard.Compose([
            standard.Resize((256, 256)),
            standard.ToTensor(),
            standard.Normalize(*mean_std)
        ])

        # 构建样本字典列表
        self.samples = self._collect_samples()
        print(f"Created dataset from: {root_dir}")
        print(f"Collected {len(self.samples)} samples for {split} split.")

    def _collect_samples(self):
        """
        收集所有样本信息，形成字典类型的列表
        """
        samples = []
        global_idx = 0

        # 遍历根目录下的每个被试文件夹
        for subject_dir in sorted(os.listdir(self.root_dir)):
            subject_path = os.path.join(self.root_dir, subject_dir)
            if not os.path.isdir(subject_path) or subject_dir == "calibration":
                continue

            # 遍历被试文件夹中的所有序列文件夹
            for seq_dir in sorted(os.listdir(subject_path)):
                seq_path = os.path.join(subject_path, seq_dir)
                if not os.path.isdir(seq_path):
                    continue

                # 统计样本数量
                count = 0
                for view_dir in sorted(os.listdir(seq_path)):
                    view_path = os.path.join(seq_path, view_dir)
                    if not os.path.isdir(view_path):
                        continue
                    count = sum(1 for filename in os.listdir(view_path) if filename.endswith('.jpg'))
                    break  # 假设每个视角文件夹的样本数量相同

                # 添加样本信息
                for sample in range(count):
                    sample_info = {
                        "subject": subject_dir,
                        "seq": seq_dir,
                        "sample": sample,
                        "idx": global_idx
                    }
                    samples.append(sample_info)
                    global_idx += 1

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        参数:
            idx: 样本索引
        返回:
            包含输入、目标和元信息的字典
        """
        # 初始化数据容器
        inputs = collections.defaultdict(list)
        targets = collections.defaultdict(list)
        meta_info = collections.defaultdict(list)

        # 获取当前样本的字典信息
        sample_info = self.samples[idx]
        subject_dir = sample_info["subject"]
        seq_dir = sample_info["seq"]
        sample_idx = sample_info["sample"]

        # 构造被试的序列文件夹路径
        seq_path = os.path.join(self.root_dir, subject_dir, seq_dir)

        # 读取meta文件
        meta_path = os.path.join(seq_path, 'meta.yml')
        with open(meta_path, 'r') as file:
            meta_data = yaml.load(file, Loader=yaml.FullLoader)
        extrinsic_path = os.path.join(self.calibration_dir, f'extrinsics_{meta_data["extrinsics"]}/extrinsics.yml')
        mano_sides = meta_data["mano_sides"][0]
        with open(extrinsic_path, 'r') as file:
            extrinsic_config = yaml.load(file, Loader=yaml.FullLoader)["extrinsics"]

        # # 读取3D关节点(世界坐标系)
        # pose_path = os.path.join(seq_path, 'pose.npz')
        # pose_data = np.load(pose_path)
        # world_joints_3d = generate_3d_joints(
        #     pose_m=pose_data['pose_m'][sample_idx, :, :],
        #     mano_model_path='/root/wk/code/Multi-view_pose_estimation/manopth/mano/models',
        #     side=mano_sides
        # )
        # world_joints_3d = world_joints_3d / 1000.0  # 将单位从毫米转换为米

        # 遍历所有视角
        for view_idx in sorted(os.listdir(seq_path)):
            # 判断是否为文件夹
            if not os.path.isdir(os.path.join(seq_path, view_idx)):
                continue

            # 读取RGB图片
            view_path = os.path.join(seq_path, view_idx)
            rgb_path = os.path.join(view_path, f'color_{sample_idx:06d}.jpg')
            img = Image.open(rgb_path).convert('RGB')
            ori_img = img.copy()

            # 应用图像变换(转换为张量并标准化)
            orig_height, orig_width = np.array(img).shape[:2]
            img = self.transform(img)
            scale_x = cfg.input_img_shape[0] / orig_width
            scale_y = cfg.input_img_shape[1] / orig_height
            img = np.array(img)

            # 读取相机参数
            intrinsic_path = os.path.join(self.calibration_dir, f'intrinsics/{view_idx}_640x480.yml')
            intrinsic_data = get_camera_color_intrinsics(intrinsic_path)  # 读取相机内参
            extrinsic_data = extrinsic_config[view_idx]  # 读取相机外参

            # 构造相机矩阵
            proj_matrix, intrinsic, extrinsic = construct_projection_matrix(intrinsic_data, extrinsic_data)

            # 直接读取该视角3D关节点标签
            pose_path = os.path.join(view_path, f'labels_{sample_idx:06d}.npz')
            pose_data = np.load(pose_path)
            cam_joints_3d = pose_data['joint_3d'].squeeze(0)

            # 将相机坐标系的3D关节点转换为世界坐标系
            # world_joints_3d = (np.dot(cam_joints_3d, extrinsic[:, :3].T) + extrinsic[:, 3])
            # world_joints_3d = np.matmul(cam_joints_3d, extrinsic[:, :3].T) + extrinsic[:, 3]
            # world_joints_3d = np.dot(extrinsic[:, :3].T, cam_joints_3d - extrinsic[:, 3])
            world_joints_3d = camera2world(cam_joints_3d, extrinsic)

            # 将3D关节点投影到图像平面(获取UVD坐标)
            joints_uvd = xyz2uvd(cam_joints_3d, intrinsic)

            # 将2D关键点坐标从原始图像空间转换到裁剪图像空间
            joints_uvd[:, 0] = joints_uvd[:, 0] * scale_x
            joints_uvd[:, 1] = joints_uvd[:, 1] * scale_y

            # 调整内参矩阵
            # 调整焦距
            intrinsic[0, 0] = intrinsic[0, 0] * scale_x  # fx
            intrinsic[1, 1] = intrinsic[1, 1] * scale_y  # fy
            # 调整主点坐标
            intrinsic[0, 2] = intrinsic[0, 2] * scale_x  # cx
            intrinsic[1, 2] = intrinsic[1, 2] * scale_y  # cy

            # 获取根关节深度
            root_joint_depth = joints_uvd[cfg.root_idx, 2:3]

            # 深度归一化(相对于根关节)
            joints_uvd_norm = joints_uvd.copy()
            joints_uvd_norm[:, 2] = joints_uvd_norm[:, 2] - root_joint_depth
            joints_uvd_norm[:, 2] = joints_uvd_norm[:, 2] / (cfg.bbox_3d_size / 2)

            # UV坐标归一化到[-1,1]范围
            joints_uvd_norm[:, 0] = joints_uvd_norm[:, 0] / (cfg.input_img_shape[0] / 2) - 1
            joints_uvd_norm[:, 1] = joints_uvd_norm[:, 1] / (cfg.input_img_shape[1] / 2) - 1

            # 3D坐标处理
            # 将3D坐标转换为相对于根关节的坐标
            cam_joints_3d_norm = cam_joints_3d - cam_joints_3d[cfg.root_idx]
            # 归一化3D坐标
            cam_joints_3d_norm = cam_joints_3d_norm / (cfg.bbox_3d_size / 2)

            # 将数据添加到相应的容器中
            inputs['img'].append(img)
            inputs['extrinsic'].append(np.float32(extrinsic))
            inputs['intrinsic'].append(np.float32(intrinsic))

            targets['mesh_pose_uvd'].append(joints_uvd_norm)
            targets['mesh_pose_xyz'].append(cam_joints_3d_norm)
            targets['intrinsic'].append(intrinsic)
            targets['extrinsic'].append(extrinsic)
            targets['world_coord'].append(world_joints_3d)
            targets['proj_matrix'].append(proj_matrix)
            targets['ori_img'].append(ori_img)

            meta_info['subject'].append(subject_dir)
            meta_info['seq'].append(seq_dir)
            meta_info['sample_idx'].append(sample_idx)
            meta_info['view_idx'].append(view_idx)

        # 将列表转换为numpy数组
        inputs = {k: np.stack(v, axis=0) for k, v in inputs.items()}
        targets = {k: np.float32(np.stack(v, axis=0)) for k, v in targets.items() if
                   k != 'result_file' and k != 'kps_file_name'}

        return inputs, targets, meta_info


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore", category=UserWarning)

    # 创建数据集实例
    dataset = DEXYCBDatasets(root_dir='/home/wk/wk/wk/datasets/DexYCB', split='train')
    # 获取一个样本
    inputs, targets, meta_info = dataset[800]
    # 打印数据形状
    print("\n数据形状:")
    print(f"输入数据: {[(k, v.shape) for k, v in inputs.items()]}")
    print(f"目标数据: {[(k, v.shape) for k, v in targets.items() if isinstance(v, np.ndarray)]}")

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(10, 8))
    view_idx = 0
    # 获取原始图像
    if isinstance(targets['ori_img'], list):
        ori_img = targets['ori_img'][view_idx]
    else:
        ori_img = targets['ori_img'][view_idx]
    # 如果需要，将浮点图像转换为uint8
    if ori_img.dtype != np.uint8:
        ori_img = (ori_img * 255).astype(np.uint8)
    # 检查颜色通道顺序
    if ori_img.shape[-1] == 3:  # RGB图像
        ori_img = ori_img[..., ::-1]  # 转换BGR到RGB
    # 在轴对象上显示图像
    ax.imshow(ori_img)
    # 获取2D关键点
    if isinstance(targets['mesh_pose_uvd'], list):
        uvd = targets['mesh_pose_uvd'][view_idx]
    else:
        uvd = targets['mesh_pose_uvd'][view_idx]
    img_height, img_width = ori_img.shape[:2]
    x = (uvd[:, 0] + 1) * img_width / 2
    y = (uvd[:, 1] + 1) * img_height / 2
    # 在轴对象上绘制关键点
    ax.scatter(x, y, c='r', s=30)
    # 定义手部骨架连接
    bones = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16),  # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
    ]
    # 在轴对象上绘制骨架连接线
    for start_idx, end_idx in bones:
        if start_idx < len(x) and end_idx < len(x):
            ax.plot([x[start_idx], x[end_idx]], [y[start_idx], y[end_idx]], 'b-', linewidth=2)
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

    visualize_3d_joints(targets['world_coord'][view_idx])
    # visualize_3d_joints(targets['mesh_pose_xyz'][view_idx])

    # 遍历8个视角坐标系下的3D关节点坐标
    for i in range(8):
        visualize_3d_joints(targets['mesh_pose_xyz'][i])


