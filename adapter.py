import torch


def collate_dexycb_to_mvgformer(batch_inputs, batch_targets, device):
    """
    将 DexYCB DataLoader 的输出转换为 MVGFormer 模型的输入格式。

    Args:
        batch_inputs (dict): 包含 'img', 'intrinsic', 'extrinsic' 等
        batch_targets (dict): 包含 'world_coord', 'mesh_pose_uvd' 等

    Returns:
        views (list of tensors): [ (B, 3, H, W), ... ] 共 V 个
        meta (list of dicts): 长度为 B，包含每个样本的相机参数和 GT
    """

    # 1. 拆分图像：(B, V, C, H, W) -> List of V * (B, C, H, W)
    imgs = batch_inputs['img'].to(device)
    B, V, C, H, W = imgs.shape
    views = [imgs[:, v, ...] for v in range(V)]

    # 2. 获取参数并移动到设备
    intrinsics = batch_inputs['intrinsic'].to(device)  # (B, V, 3, 3)
    extrinsics = batch_inputs['extrinsic'].to(device)  # (B, V, 3, 4) [已经是 World2Cam]

    gt_3d = batch_targets['world_coord'].to(device)  # (B, V, 21, 3)

    # 3. 构建 Meta 列表
    meta_list = []

    for b in range(B):
        sample_meta = {
            'camera': {},
            'num_person': [1],  # 只有一只手
            'joints_3d': gt_3d[b, 0:1, ...],  # (1, 21, 3)
            'joints_3d_vis': torch.ones(1, 21, 3).to(device),
            'roots_3d': gt_3d[b, 0, 0, :],  # 根关节 (Wrist)
            'joints_vis': [],
            'center': [],
            'scale': [],
            'affine_trans': [],
        }

        cam_Rs = []
        cam_Ts = []
        cam_Ks = []
        affines = []

        for v in range(V):

            # --- 修正部分开始 ---
            # DexYCB 返回的 extrinsics 已经是 World -> Camera 的 3x4 矩阵
            # 因此不需要求逆，直接切片提取 R 和 T
            w2c = extrinsics[b, v, ...]  # (3, 4)

            # 兼容性处理：防止有些数据loader返回4x4
            if w2c.shape[0] == 4:
                w2c = w2c[:3, :]  # 取前3行

            # 提取 R (3x3) 和 T (3,)
            R_w2c = w2c[:3, :3]
            T_w2c = w2c[:3, 3]

            K = intrinsics[b, v, ...]
            # --- 修正部分结束 ---

            cam_Rs.append(R_w2c)
            cam_Ts.append(T_w2c)
            cam_Ks.append(K)

            affines.append(torch.eye(3).to(device))

            sample_meta['joints_vis'].append(torch.ones(21, 1).to(device))
            sample_meta['center'].append(torch.tensor([W / 2, H / 2]).to(device))
            sample_meta['scale'].append(torch.tensor([1.0]).to(device))

            sample_meta['camera'][v] = {
                'K': K,
                'R': R_w2c,
                'T': T_w2c,
                'fx': K[0, 0], 'fy': K[1, 1], 'cx': K[0, 2], 'cy': K[1, 2]
            }

        sample_meta['camera_R'] = torch.stack(cam_Rs)  # (V, 3, 3)
        sample_meta['camera_T'] = torch.stack(cam_Ts)  # (V, 3)
        sample_meta['camera_standard_T'] = sample_meta['camera_T']
        sample_meta['camera_Intri'] = torch.stack(cam_Ks)  # (V, 3, 3)
        sample_meta['affine_trans'] = torch.stack(affines)

        meta_list.append(sample_meta)

    return views, meta_list