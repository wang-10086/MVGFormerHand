import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np


# -----------------------------------------------------------------------------
# 1. 辅助函数 (DLT, Backbone, Fusion)
# -----------------------------------------------------------------------------

def dlt_triangulation(extrinsic, intrinsic, mesh_pose_uvd):
    """
    使用DLT方法从多视角的2D关节点坐标重建3D关节点坐标
    """
    num_views, num_joints, _ = mesh_pose_uvd.shape

    # 计算每个视角的投影矩阵 P = K[R|t]
    projection_matrices = np.zeros((num_views, 3, 4))
    for i in range(num_views):
        projection_matrices[i] = intrinsic[i] @ extrinsic[i]

    # 初始化结果数组
    joints_3d = np.zeros((num_joints, 3))

    # 对每个关节点进行三角测量
    for j in range(num_joints):
        A = np.zeros((2 * num_views, 4))
        for i in range(num_views):
            u, v = mesh_pose_uvd[i, j, 0:2]
            P = projection_matrices[i]
            A[2 * i] = u * P[2] - P[0]
            A[2 * i + 1] = v * P[2] - P[1]

        _, _, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1]
        # 加上 1e-8 防止除零
        joints_3d[j] = X_homogeneous[:3] / (X_homogeneous[3] + 1e-8)

    return joints_3d


class HandPoseResNet(nn.Module):
    """
    基于ResNet50的手部姿态估计网络
    """

    def __init__(self, num_joints=21):
        super(HandPoseResNet, self).__init__()
        # 加载预训练的ResNet50
        resnet = resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2D 坐标回归
        self.joints_regressor = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_joints * 2)
        )
        # 置信度回归
        self.confidence_regressor = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_joints)
        )

    def extract_joint_features(self, features, joints_2d, feature_size, window_size=7):
        batch_size, channels, height, width = features.shape
        num_joints = joints_2d.shape[1]
        device = features.device

        scaled_joints = joints_2d * feature_size
        normalized_x = (scaled_joints[:, :, 0] / (width - 1)) * 2 - 1
        normalized_y = (scaled_joints[:, :, 1] / (height - 1)) * 2 - 1
        normalized_coords = torch.stack([normalized_x, normalized_y], dim=-1)

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-window_size // 2, window_size // 2, window_size, device=device),
            torch.linspace(-window_size // 2, window_size // 2, window_size, device=device)
        )
        grid_offset = torch.stack([grid_y, grid_x], dim=-1).unsqueeze(0).unsqueeze(0)
        grid_offset = grid_offset / torch.tensor([width, height], device=device) * 2

        normalized_coords = normalized_coords.unsqueeze(-2).unsqueeze(-2)
        grid = normalized_coords + grid_offset
        grid = grid.view(batch_size * num_joints, window_size, window_size, 2)

        features_expanded = features.unsqueeze(1).repeat(1, num_joints, 1, 1, 1)
        features_expanded = features_expanded.view(batch_size * num_joints, channels, height, width)

        sampled_features = F.grid_sample(features_expanded, grid, mode='bilinear', padding_mode='zeros',
                                         align_corners=True)
        joint_features = sampled_features.view(batch_size, num_joints, channels, window_size, window_size)
        return joint_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat_64 = self.layer1(x)
        feat_32 = self.layer2(feat_64)
        feat_16 = self.layer3(feat_32)
        feat_8 = self.layer4(feat_16)

        features = self.avg_pool(feat_8)
        features = torch.flatten(features, 1)

        joints_2d_flat = self.joints_regressor(features)
        joints_2d = joints_2d_flat.view(-1, 21, 2)
        confidences = torch.sigmoid(self.confidence_regressor(features))

        features_dict = {'feat_64': feat_64, 'feat_32': feat_32, 'feat_16': feat_16, 'feat_8': feat_8}
        joint_features_dict = {
            'jaf_64': self.extract_joint_features(feat_64, joints_2d, feature_size=1 / 4),
            'jaf_32': self.extract_joint_features(feat_32, joints_2d, feature_size=1 / 8),
            'jaf_16': self.extract_joint_features(feat_16, joints_2d, feature_size=1 / 16),
            'jaf_8': self.extract_joint_features(feat_8, joints_2d, feature_size=1 / 32)
        }
        return joints_2d, confidences, features_dict, joint_features_dict


class HierarchicalFeatureFusion(nn.Module):
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=512):
        super().__init__()
        self.level_processors = nn.ModuleList([
            nn.Sequential(nn.Linear(c * 7 * 7, out_channels), nn.ReLU(), nn.LayerNorm(out_channels))
            for c in in_channels_list
        ])
        self.view_fusion = nn.MultiheadAttention(out_channels, 8)
        self.level_fusion = nn.MultiheadAttention(out_channels, 8)
        self.refine_regressor = nn.Sequential(nn.Linear(out_channels, 256), nn.ReLU(), nn.Linear(256, 3))

    def forward(self, joint_features_dict, joints_3d):
        batch_size, num_joints, _ = joints_3d.shape
        feature_keys = ['jaf_64', 'jaf_32', 'jaf_16', 'jaf_8']
        processed_features = []
        for i, key in enumerate(feature_keys):
            feats = joint_features_dict[key]
            B, V, J, C, H, W = feats.shape
            feats_flat = feats.reshape(B, V, J, -1)
            level_features = []
            for j in range(num_joints):
                joint_feats = feats_flat[:, :, j, :]
                processed_views = []
                for b in range(batch_size):
                    processed = self.level_processors[i](joint_feats[b])
                    processed_views.append(processed)
                processed_views = torch.stack(processed_views)
                views_t = processed_views.transpose(0, 1)
                attn_out, _ = self.view_fusion(views_t, views_t, views_t)
                fused_views = attn_out.transpose(0, 1).mean(dim=1)
                level_features.append(fused_views)
            level_features = torch.stack(level_features, dim=1)
            processed_features.append(level_features)

        refined_joints = []
        for j in range(num_joints):
            level_feats = [feat[:, j, :] for feat in processed_features]
            level_feats = torch.stack(level_feats, dim=0)
            attn_out, _ = self.level_fusion(level_feats, level_feats, level_feats)
            fused_feats = attn_out.mean(dim=0)
            delta = self.refine_regressor(fused_feats)
            refined_pos = joints_3d[:, j, :] + delta
            refined_joints.append(refined_pos)
        refined_joints = torch.stack(refined_joints, dim=1)
        return refined_joints


# -----------------------------------------------------------------------------
# 2. 主模型类 (重命名为 LAT)
# -----------------------------------------------------------------------------

class LAT(nn.Module):
    def __init__(self, cfg=None):
        super(LAT, self).__init__()
        self.cfg = cfg
        # 2D关节点预测模块
        self.pose_estimator = HandPoseResNet(num_joints=21)
        # 特征融合模块
        self.feature_fusion = HierarchicalFeatureFusion(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=512
        )

        # 定义损失函数
        self.criterion_joints = nn.L1Loss()
        self.criterion_confidence = nn.BCELoss()

    def forward(self, views, meta=None):
        """
        参数:
            views: list of tensors, [(B, 3, H, W), ...]
            meta: list of dicts, [{'camera_R':..., 'camera_T':..., 'camera_Intri':...}, ...]
        """
        device = views[0].device
        batch_size = views[0].shape[0]
        num_views = len(views)

        # 1. 整理图像输入 (B, V, 3, H, W) -> (B*V, 3, H, W)
        images = torch.stack(views, dim=1)
        images_reshaped = images.view(batch_size * num_views, 3, images.shape[3], images.shape[4])

        # 2. 整理相机参数
        if meta is not None:
            # Stack相机参数用于DLT和Loss
            Rs = torch.stack([m['camera_R'] for m in meta]).to(device)
            Ts = torch.stack([m['camera_T'] for m in meta]).to(device)
            # Concatenate [R|t]
            extrinsics = torch.cat([Rs, Ts.unsqueeze(-1)], dim=-1)
            intrinsics = torch.stack([m['camera_Intri'] for m in meta]).to(device)
        else:
            # 推理时防止空指针，但实际上推理必须有 meta
            extrinsics = torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1).to(device)
            intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1).to(device)

        # 3. 前向传播：预测 2D 关键点
        joints_2d, confidences, features_dict_list, joint_features_dict_list = self.pose_estimator(images_reshaped)

        # Reshape: (B*V, ...) -> (B, V, ...)
        joints_2d = joints_2d.view(batch_size, num_views, 21, 2)
        confidences = confidences.view(batch_size, num_views, 21)
        joints_2d_copy = joints_2d.clone()

        # 4. DLT 三角测量 (获取初始 3D)
        joints_3d_list = []
        for b in range(batch_size):
            sample_extrinsics = extrinsics[b].detach().cpu().numpy()
            sample_intrinsics = intrinsics[b].detach().cpu().numpy()
            sample_joints_2d = joints_2d_copy[b].detach().cpu().numpy()

            # 构建 uvd (d设为0)
            sample_joints_uvd = np.zeros((num_views, 21, 3))
            sample_joints_uvd[:, :, :2] = sample_joints_2d

            sample_joints_3d = dlt_triangulation(sample_extrinsics, sample_intrinsics, sample_joints_uvd)
            joints_3d_list.append(sample_joints_3d)

        joints_3d = torch.tensor(np.stack(joints_3d_list, axis=0), dtype=torch.float32, device=device)

        # 5. 特征融合 (Refine 3D)
        joint_features_dict_reshaped = {
            k: v.view(batch_size, num_views, 21, *v.shape[2:])
            for k, v in joint_features_dict_list.items()
        }
        refined_joints_3d = self.feature_fusion(joint_features_dict_reshaped, joints_3d)

        # 6. 整理输出
        outputs = {
            'joints_2d': joints_2d,
            'confidences': confidences,
            'joints_3d_initial': joints_3d,
            'final_pred_poses': refined_joints_3d,  # 适配 test.py
            'pred_poses': {'outputs_coord': refined_joints_3d}  # 适配 train.py
        }

        # 7. 计算 Loss (仅在训练且有 GT 时)
        loss_dict = {}
        if self.training and meta is not None and 'joints_3d' in meta[0]:
            # 提取 GT 3D (B, 21, 3)
            gt_joints_3d = torch.stack([m['joints_3d'] for m in meta]).to(device)
            if gt_joints_3d.dim() == 4: gt_joints_3d = gt_joints_3d.squeeze(1)

            # 生成 GT 2D (用于 Loss)
            gt_joints_2d = self.project_3d_to_2d_batch(gt_joints_3d, intrinsics, extrinsics)

            targets_simulated = {
                'mesh_pose_uvd': gt_joints_2d,
                'intrinsic': intrinsics,
                'extrinsic': extrinsics,
                'world_coord': gt_joints_3d
            }
            loss_dict = self.compute_loss(outputs, targets_simulated)

        return outputs, loss_dict

    def project_3d_to_2d_batch(self, points_3d, K, E):
        """
        辅助函数：批量将 3D 点投影为 2D 像素坐标
        """
        B, N, _ = points_3d.shape
        V = K.shape[1]

        pts = points_3d.unsqueeze(1).expand(-1, V, -1, -1)
        R = E[..., :3]
        T = E[..., 3]

        pts_cam = torch.matmul(pts, R.transpose(-1, -2)) + T.unsqueeze(2)

        z = pts_cam[..., 2].clamp(min=1e-5)
        x = pts_cam[..., 0]
        y = pts_cam[..., 1]

        fx = K[..., 0, 0].unsqueeze(2)
        fy = K[..., 1, 1].unsqueeze(2)
        cx = K[..., 0, 2].unsqueeze(2)
        cy = K[..., 1, 2].unsqueeze(2)

        u = x * fx / z + cx
        v = y * fy / z + cy

        return torch.stack([u, v, z], dim=-1)

    def _compute_reprojection_loss(self, joints_3d, gt_joints_2d, intrinsics, extrinsics):
        batch_size, num_joints, _ = joints_3d.shape
        num_views = gt_joints_2d.shape[1]
        device = joints_3d.device

        joints_3d_homogeneous = torch.cat(
            [joints_3d, torch.ones(batch_size, num_joints, 1, device=device)],
            dim=2
        )
        total_loss = 0
        valid_projections = 0

        for b in range(batch_size):
            for v in range(num_views):
                proj_matrix = torch.matmul(intrinsics[b, v], extrinsics[b, v])
                projected_points = torch.matmul(joints_3d_homogeneous[b], proj_matrix.transpose(0, 1))

                valid_mask = projected_points[:, 2] > 1e-8
                if torch.sum(valid_mask) == 0: continue

                projected_points_valid = projected_points[valid_mask]
                gt_points_valid = gt_joints_2d[b, v, valid_mask]

                projected_uvs = projected_points_valid[:, :2] / projected_points_valid[:, 2:3]
                gt_uv_valid = gt_points_valid[:, :2]

                view_loss = F.l1_loss(projected_uvs, gt_uv_valid)
                total_loss += view_loss
                valid_projections += 1

        if valid_projections > 0:
            return total_loss / valid_projections
        else:
            return torch.tensor(0.0, device=device)

    def compute_loss(self, outputs, targets):
        """
        计算完整的损失函数，包含：
        1. 2D 关节点损失 (L1)
        2. 置信度损失 (BCE)
        3. 3D 关节点损失 (L1, 核心指标)
        4. 重投影损失 (Reprojection, 几何约束)
        """
        # 1. 提取预测值
        pred_joints_2d = outputs['joints_2d']  # (B, V, 21, 2)
        pred_confidences = outputs['confidences']  # (B, V, 21)
        pred_joints_3d = outputs['final_pred_poses']  # (B, 21, 3) Refined 3D

        # 2. 提取 Target
        # 2D 真值 (B, V, 21, 3) -> 取前两维 (u, v)
        gt_joints_uvd = targets['mesh_pose_uvd'].to(pred_joints_2d.device)
        gt_joints_2d = gt_joints_uvd[:, :, :, :2]

        # 3D 真值 (B, 21, 3)
        gt_joints_3d = targets['world_coord'].to(pred_joints_3d.device)

        # 相机参数 (用于重投影)
        intrinsics = targets['intrinsic']
        extrinsics = targets['extrinsic']

        # 置信度真值 (默认为全 1)
        gt_confidences = torch.ones_like(pred_confidences)

        # 3. 计算各项损失
        # (A) 2D Loss: 确保 ResNet 基础预测准确
        loss_joints_2d = self.criterion_joints(pred_joints_2d, gt_joints_2d)

        # (B) Confidence Loss: 确保置信度分支收敛
        loss_confidence = self.criterion_confidence(pred_confidences, gt_confidences)

        # (C) 3D Loss: 直接优化 MPJPE
        loss_joints_3d = self.criterion_joints(pred_joints_3d, gt_joints_3d)

        # (D) Reprojection Loss: 约束 Fusion 模块保持多视角几何一致性
        loss_reprojection = self._compute_reprojection_loss(
            pred_joints_3d, gt_joints_uvd, intrinsics, extrinsics
        )

        # 4. 汇总
        loss_dict = {
            'loss_joints_2d': loss_joints_2d,
            'loss_confidence': loss_confidence * 2,
            'loss_joints_3d': loss_joints_3d * 10,
            'loss_reprojection': loss_reprojection
        }

        return loss_dict


def get_lat_model(cfg):
    return LAT(cfg)