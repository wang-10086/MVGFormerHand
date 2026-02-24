import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import numpy as np


# -----------------------------------------------------------------------------
# 1. 辅助函数: 可微加权 DLT (Differentiable Weighted DLT)
# -----------------------------------------------------------------------------

def differentiable_weighted_dlt(intrinsic, extrinsic, uv, confidences):
    """
    [核心修复] 全流程可微的加权 DLT 三角测量 (Pure PyTorch Implementation)

    Args:
        intrinsic: (B, V, 3, 3)
        extrinsic: (B, V, 3, 4)
        uv: (B, V, J, 2)  归一化像素坐标 (通常在 0~W 之间)
        confidences: (B, V, J) 预测的置信度权重

    Returns:
        joints_3d: (B, J, 3) 世界坐标系下的 3D 坐标
    """
    B, V, J, _ = uv.shape
    device = uv.device

    # 1. 计算投影矩阵 P = K @ E
    # intrinsic: (B, V, 3, 3), extrinsic: (B, V, 3, 4) -> P: (B, V, 3, 4)
    P = torch.matmul(intrinsic, extrinsic)

    # 2. 扩展维度以支持广播: (B, V, 1, 3, 4) -> (B, V, J, 3, 4)
    P_exp = P.unsqueeze(2).expand(-1, -1, J, -1, -1)

    # 3. 准备约束方程所需的变量
    # uv: (B, V, J, 2)
    u = uv[..., 0].unsqueeze(-1)  # (B, V, J, 1)
    v = uv[..., 1].unsqueeze(-1)  # (B, V, J, 1)
    w = confidences.unsqueeze(-1)  # (B, V, J, 1) 使用置信度作为权重

    # 4. 构建 DLT 线性方程组 A * X = 0
    # 每一行方程: w * (u * P_row2 - P_row0)
    # P_exp shape: (B, V, J, 3, 4)

    # Row 1: x 轴约束
    row0 = w * (u * P_exp[..., 2, :] - P_exp[..., 0, :])  # (B, V, J, 4)
    # Row 2: y 轴约束
    row1 = w * (v * P_exp[..., 2, :] - P_exp[..., 1, :])  # (B, V, J, 4)

    # 堆叠所有视角的约束 -> A
    # Stack dim: (B, V, J, 2, 4)
    A = torch.stack([row0, row1], dim=3)

    # 变换形状以进行 SVD: (B, J, 2*V, 4)
    # 我们把 Batch 和 Joints 维度合并处理，把所有视角的约束拼在一起
    A = A.permute(0, 2, 1, 3, 4).reshape(B * J, 2 * V, 4)

    # 5. 可微 SVD 求解
    # 求解 A * X = 0 的最小二乘解，即 A 的最小奇异值对应的右奇异向量
    # 注意：如果 A 全为 0 (所有视角都没看见)，SVD 可能会不稳定，但 PyTorch 通常能处理
    try:
        # Vh shape: (B*J, 4, 4) (当 full_matrices=False)
        # 结果是 Vh 的最后一行 (对应 V 的最后一列)
        _, _, Vh = torch.linalg.svd(A)
        X_homo = Vh[:, -1, :]  # (B*J, 4)
    except RuntimeError:
        # 极少数情况 SVD 不收敛，返回 0
        X_homo = torch.zeros(B * J, 4, device=device)
        X_homo[:, 3] = 1.0

    # 6. 齐次坐标归一化
    w_homo = X_homo[:, 3:4]

    # 鲁棒性处理：检测 "盲点" (所有视角置信度都很低)
    # total_w: (B, J)
    total_w = confidences.sum(dim=1).view(B * J, 1)
    valid_mask = (total_w > 0.1) & (torch.abs(w_homo) > 1e-6)

    # 避免除以 0
    X_3d = X_homo[:, :3] / (w_homo + 1e-8)

    # 对于无效点，强制设为 0 (或者空间中心)，并不阻断梯度 (mask 只在数值上生效)
    X_3d = X_3d * valid_mask.float()

    return X_3d.reshape(B, J, 3)


# -----------------------------------------------------------------------------
# 2. ResNet Backbone (带 Soft-Argmax)
# -----------------------------------------------------------------------------

class HandPoseResNet(nn.Module):
    """
    [改进版] 使用 Soft-Argmax 进行 2D 坐标回归
    保留空间特征以提高 2D 精度
    """

    def __init__(self, num_joints=21, heatmap_size=64):
        super(HandPoseResNet, self).__init__()
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size

        # 1. Backbone
        resnet = resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 2. Head: 8x8 -> 64x64
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 256, 256],
            num_kernels=[4, 4, 4],
            in_channels=2048
        )

        self.final_layer = nn.Conv2d(256, num_joints, 1, 1, 0)

        # 3. Confidence Head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.confidence_regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_joints)
        )

        # 缓存网格
        self.register_buffer('grid_x', torch.arange(heatmap_size).float().unsqueeze(0))
        self.register_buffer('grid_y', torch.arange(heatmap_size).float().unsqueeze(1))

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, in_channels):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.ConvTranspose2d(in_channels, num_filters[i], num_kernels[i], 2, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(nn.ReLU(inplace=True))
            in_channels = num_filters[i]
        return nn.Sequential(*layers)

    def soft_argmax(self, heatmaps):
        B, K, H, W = heatmaps.shape
        heatmaps = heatmaps.reshape(B, K, -1)
        heatmaps = F.softmax(heatmaps, dim=2)
        heatmaps = heatmaps.reshape(B, K, H, W)

        acc_x = torch.sum(heatmaps * self.grid_x, dim=(2, 3))
        acc_y = torch.sum(heatmaps * self.grid_y, dim=(2, 3))

        norm_x = acc_x / (W - 1)
        norm_y = acc_y / (H - 1)
        return torch.stack([norm_x, norm_y], dim=-1)

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
        return sampled_features.view(batch_size, num_joints, channels, window_size, window_size)

    def forward(self, x):
        B_V, _, H_in, W_in = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat_64 = self.layer1(x)
        feat_32 = self.layer2(feat_64)
        feat_16 = self.layer3(feat_32)
        feat_8 = self.layer4(feat_16)

        # Branch 1: Heatmap
        heatmap_feat = self.deconv_layers(feat_8)
        heatmaps = self.final_layer(heatmap_feat)
        norm_coords = self.soft_argmax(heatmaps)

        joints_2d = torch.zeros_like(norm_coords)
        joints_2d[:, :, 0] = norm_coords[:, :, 0] * (W_in - 1)
        joints_2d[:, :, 1] = norm_coords[:, :, 1] * (H_in - 1)

        # Branch 2: Confidence
        pooled = self.avg_pool(feat_8)
        pooled = torch.flatten(pooled, 1)
        confidences = torch.sigmoid(self.confidence_regressor(pooled))

        features_dict = {'feat_64': feat_64, 'feat_32': feat_32, 'feat_16': feat_16, 'feat_8': feat_8}

        # 特征提取时detach坐标，防止此处的梯度干扰回归训练（可选策略，通常更稳）
        joint_features_dict = {
            'jaf_64': self.extract_joint_features(feat_64, joints_2d.detach(), feature_size=1 / 4),
            'jaf_32': self.extract_joint_features(feat_32, joints_2d.detach(), feature_size=1 / 8),
            'jaf_16': self.extract_joint_features(feat_16, joints_2d.detach(), feature_size=1 / 16),
            'jaf_8': self.extract_joint_features(feat_8, joints_2d.detach(), feature_size=1 / 32)
        }

        return joints_2d, confidences, features_dict, joint_features_dict


# -----------------------------------------------------------------------------
# 3. 特征融合模块 (保持不变)
# -----------------------------------------------------------------------------

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
# 4. 主模型类 (LAT) - 集成可微 DLT
# -----------------------------------------------------------------------------

class LAT(nn.Module):
    def __init__(self, cfg=None):
        super(LAT, self).__init__()
        self.cfg = cfg
        self.pose_estimator = HandPoseResNet(num_joints=21)
        self.feature_fusion = HierarchicalFeatureFusion(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=512
        )
        self.criterion_joints = nn.L1Loss()
        self.criterion_confidence = nn.BCELoss()

    def forward(self, views, meta=None):
        device = views[0].device
        batch_size = views[0].shape[0]
        num_views = len(views)

        images = torch.stack(views, dim=1)
        images_reshaped = images.view(batch_size * num_views, 3, images.shape[3], images.shape[4])

        if meta is not None:
            Rs = torch.stack([m['camera_R'] for m in meta]).to(device)
            Ts = torch.stack([m['camera_T'] for m in meta]).to(device)
            extrinsics = torch.cat([Rs, Ts.unsqueeze(-1)], dim=-1)
            intrinsics = torch.stack([m['camera_Intri'] for m in meta]).to(device)
        else:
            extrinsics = torch.eye(3, 4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1).to(device)
            intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1).to(device)

        # 1. 2D Prediction
        joints_2d, confidences, features_dict_list, joint_features_dict_list = self.pose_estimator(images_reshaped)

        joints_2d = joints_2d.view(batch_size, num_views, 21, 2)
        confidences = confidences.view(batch_size, num_views, 21)

        # 2. [关键修改] 可微加权 DLT
        # 直接传入 Tensor，不进行 detach，保持梯度流
        # 归一化 joints_2d 到 [0, 1] 区间还是保持像素坐标？
        # 注意：DLT 公式依赖 P=K@E，K通常是针对像素坐标的。所以这里传入像素坐标 joints_2d 是对的。
        joints_3d = differentiable_weighted_dlt(intrinsics, extrinsics, joints_2d, confidences)

        # 3. Fusion
        joint_features_dict_reshaped = {
            k: v.view(batch_size, num_views, 21, *v.shape[2:])
            for k, v in joint_features_dict_list.items()
        }
        refined_joints_3d = self.feature_fusion(joint_features_dict_reshaped, joints_3d)

        outputs = {
            'joints_2d': joints_2d,
            'confidences': confidences,
            'joints_3d_initial': joints_3d,
            'final_pred_poses': refined_joints_3d,
            'pred_poses': {'outputs_coord': refined_joints_3d}
        }

        # 4. Loss
        loss_dict = {}
        if self.training and meta is not None and 'joints_3d' in meta[0]:
            gt_joints_3d = torch.stack([m['joints_3d'] for m in meta]).to(device)
            if gt_joints_3d.dim() == 4: gt_joints_3d = gt_joints_3d.squeeze(1)

            gt_joints_2d_proj = self.project_3d_to_2d_batch(gt_joints_3d, intrinsics, extrinsics)

            targets_simulated = {
                'mesh_pose_uvd': gt_joints_2d_proj,
                'intrinsic': intrinsics,
                'extrinsic': extrinsics,
                'world_coord': gt_joints_3d
            }
            loss_dict = self.compute_loss_robust(outputs, targets_simulated)

        return outputs, loss_dict

    def project_3d_to_2d_batch(self, points_3d, K, E):
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

    def _compute_reprojection_loss(self, joints_3d, gt_joints_2d, intrinsics, extrinsics, vis_mask=None):
        batch_size, num_joints, _ = joints_3d.shape
        num_views = gt_joints_2d.shape[1]
        device = joints_3d.device

        joints_3d_homogeneous = torch.cat(
            [joints_3d, torch.ones(batch_size, num_joints, 1, device=device)], dim=2
        )
        total_loss = 0
        valid_projections = 0

        for b in range(batch_size):
            for v in range(num_views):
                if vis_mask is not None and vis_mask[b, v].sum() < 1.0:
                    continue

                proj_matrix = torch.matmul(intrinsics[b, v], extrinsics[b, v])
                projected_points = torch.matmul(joints_3d_homogeneous[b], proj_matrix.transpose(0, 1))

                valid_z = projected_points[:, 2] > 1e-8
                if torch.sum(valid_z) == 0: continue

                projected_points_valid = projected_points[valid_z]
                gt_points_valid = gt_joints_2d[b, v, valid_z]

                projected_uvs = projected_points_valid[:, :2] / projected_points_valid[:, 2:3]
                gt_uv_valid = gt_points_valid[:, :2]

                view_loss = F.l1_loss(projected_uvs, gt_uv_valid)
                total_loss += view_loss
                valid_projections += 1

        if valid_projections > 0:
            return total_loss / valid_projections
        else:
            return torch.tensor(0.0, device=device)

    def compute_loss_robust(self, outputs, targets):
        pred_joints_2d = outputs['joints_2d']
        pred_confidences = outputs['confidences']
        pred_joints_3d = outputs['final_pred_poses']

        gt_joints_uvd = targets['mesh_pose_uvd'].to(pred_joints_2d.device)
        gt_joints_2d = gt_joints_uvd[:, :, :, :2]
        gt_joints_3d = targets['world_coord'].to(pred_joints_3d.device)

        intrinsics = targets['intrinsic']
        extrinsics = targets['extrinsic']

        # Visibility Mask
        gt_z = gt_joints_uvd[:, :, :, 2]
        vis_mask_z = (gt_z > 0.05).float()
        H, W = 256, 256
        gt_u = gt_joints_2d[:, :, :, 0]
        gt_v = gt_joints_2d[:, :, :, 1]
        vis_mask_box = (gt_u > 0) & (gt_u < W) & (gt_v > 0) & (gt_v < H)
        vis_mask = vis_mask_z * vis_mask_box.float()

        # 2D Loss
        loss_joints_2d_raw = F.l1_loss(pred_joints_2d, gt_joints_2d, reduction='none')
        loss_joints_2d_masked = loss_joints_2d_raw * vis_mask.unsqueeze(-1)
        loss_joints_2d = loss_joints_2d_masked.sum() / (vis_mask.sum() + 1e-6)

        # Confidence Loss
        gt_confidences = vis_mask
        loss_confidence = self.criterion_confidence(pred_confidences, gt_confidences)

        # 3D Loss
        loss_joints_3d = self.criterion_joints(pred_joints_3d, gt_joints_3d) * 100.0

        # Reprojection Loss
        loss_reprojection = self._compute_reprojection_loss(
            pred_joints_3d, gt_joints_2d, intrinsics, extrinsics, vis_mask=vis_mask
        )

        loss_dict = {
            'loss_joints_2d': loss_joints_2d,
            'loss_confidence': loss_confidence * 5.0,
            'loss_joints_3d': loss_joints_3d,
            'loss_reprojection': loss_reprojection
        }
        return loss_dict


def get_lat_model(cfg):
    return LAT(cfg)