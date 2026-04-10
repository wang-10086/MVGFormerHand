import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# -----------------------------------------------------------------------------
# 1. 2D Backbone (带 Soft-Argmax 和 2D 监督)
# -----------------------------------------------------------------------------
class PoseResNet(nn.Module):
    def __init__(self, name='resnet50', pretrained=True, num_joints=21, heatmap_size=64):
        super().__init__()
        self.num_joints = num_joints
        self.heatmap_size = heatmap_size

        if name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            self.in_channels = 2048
        else:
            backbone = torchvision.models.resnet18(pretrained=pretrained)
            self.in_channels = 512

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.deconv_layers = self._make_deconv_layer(
            num_layers=3, num_filters=[256, 256, 256], num_kernels=[4, 4, 4]
        )

        self.final_layer = nn.Conv2d(256, num_joints, 1, 1, 0)

        self.register_buffer('grid_x', torch.arange(heatmap_size).float().unsqueeze(0))
        self.register_buffer('grid_y', torch.arange(heatmap_size).float().unsqueeze(1))

        self._init_weights()

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            layers.append(nn.ConvTranspose2d(self.in_channels, num_filters[i], num_kernels[i], 2, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(num_filters[i]))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = num_filters[i]
        return nn.Sequential(*layers)

    def _init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

    def forward(self, x):
        H_in, W_in = x.shape[2], x.shape[3]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        features = self.deconv_layers(x)

        heatmaps = self.final_layer(features)
        norm_coords = self.soft_argmax(heatmaps)

        joints_2d = torch.zeros_like(norm_coords)
        joints_2d[:, :, 0] = norm_coords[:, :, 0] * (W_in - 1)
        joints_2d[:, :, 1] = norm_coords[:, :, 1] * (H_in - 1)

        return features, joints_2d


# -----------------------------------------------------------------------------
# 2. V2VNet (4层 Hourglass, 适配 128^3 体素)
# -----------------------------------------------------------------------------
class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_planes)
        )
        self.skip_con = nn.Sequential()
        if in_planes != out_planes:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        return F.relu(self.res_branch(x) + self.skip_con(x), True)


class V2VModel(nn.Module):
    """
    4层 U-Net 结构，适配 96^3 输入。
    通道数设计: 为控制显存，使用较窄的通道 (16→32→64→128)。
    分辨率路径: 96→48→24→12→6 (编码) → 12→24→48→96 (解码)
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()

        # 编码器: 逐层降采样
        self.enc1 = Res3DBlock(input_channels, 16)   # 128³ → 16ch
        self.pool1 = nn.MaxPool3d(2, 2)

        self.enc2 = Res3DBlock(16, 32)                # 64³ → 32ch
        self.pool2 = nn.MaxPool3d(2, 2)

        self.enc3 = Res3DBlock(32, 64)                # 32³ → 64ch
        self.pool3 = nn.MaxPool3d(2, 2)

        self.enc4 = Res3DBlock(64, 128)               # 16³ → 128ch
        self.pool4 = nn.MaxPool3d(2, 2)

        # 瓶颈层: 8³, 全局感受野覆盖整只手
        self.bottleneck = Res3DBlock(128, 128)         # 8³ → 128ch

        # 解码器: 逐层上采样 + 跳跃连接
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec4 = Res3DBlock(128 + 128, 128)         # 16³

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec3 = Res3DBlock(128 + 64, 64)           # 32³

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec2 = Res3DBlock(64 + 32, 32)            # 64³

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.dec1 = Res3DBlock(32 + 16, 16)            # 128³

        self.output_layer = nn.Conv3d(16, output_channels, kernel_size=1)
        self._init_weights()

    def forward(self, x):
        from torch.utils.checkpoint import checkpoint

        # 编码 (梯度检查点: 不保存中间激活, 反向时重算)
        x1 = checkpoint(self.enc1, x, use_reentrant=False)
        x2 = checkpoint(self.enc2, self.pool1(x1), use_reentrant=False)
        x3 = checkpoint(self.enc3, self.pool2(x2), use_reentrant=False)
        x4 = checkpoint(self.enc4, self.pool3(x3), use_reentrant=False)
        x5 = checkpoint(self.bottleneck, self.pool4(x4), use_reentrant=False)

        # 解码 (128³ 和 64³ 的高分辨率层使用检查点, 其余正常计算)
        d4 = self.dec4(torch.cat([self.up4(x5), x4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x3], dim=1))
        d2 = checkpoint(self.dec2, torch.cat([self.up2(d3), x2], dim=1), use_reentrant=False)
        d1 = checkpoint(self.dec1, torch.cat([self.up1(d2), x1], dim=1), use_reentrant=False)

        return self.output_layer(d1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)


# -----------------------------------------------------------------------------
# 3. LVT 主类
# -----------------------------------------------------------------------------
class LVT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_joints = cfg.DECODER.num_keypoints

        self.backbone = PoseResNet(name='resnet50', pretrained=True, num_joints=self.num_joints)

        # 2D→3D 通道映射: 256→32 (96³体素下需要控制显存)
        self.feat_channels = 32
        self.process_features = nn.Sequential(
            nn.Conv2d(256, self.feat_channels, kernel_size=1),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(inplace=True)
        )

        self.volume_net = V2VModel(self.feat_channels, self.num_joints)
        self.vol_size = 96

        self.register_buffer('space_size', torch.tensor([0.4, 0.4, 0.4]).float())
        self.register_buffer('space_center', torch.tensor([0.0, 0.0, 0.0]).float())

    def _build_grid(self, batch_size, device):
        r = torch.arange(self.vol_size, device=device).float()
        z, y, x = torch.meshgrid(r, r, r, indexing='ij')
        grid = torch.stack([x, y, z], dim=0) / (self.vol_size - 1)

        size = self.space_size.view(3, 1, 1, 1)
        center = self.space_center.view(3, 1, 1, 1)
        grid_3d = (grid - 0.5) * size + center
        return grid_3d.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

    def _unproject(self, features, cameras, grid_3d, img_size):
        """
        显存优化版: 逐视角采样后立即累加并释放中间变量
        """
        B, V, C, Hf, Wf = features.shape
        D, H, W = grid_3d.shape[2:]
        img_w, img_h = img_size

        pts_3d = grid_3d.view(B, 3, -1).permute(0, 2, 1)  # (B, D*H*W, 3)

        volume_sum = torch.zeros(B, C, D, H, W, device=features.device)
        weight_sum = torch.zeros(B, 1, D, H, W, device=features.device)

        Rs, Ts, Ks = cameras['camera_R'], cameras['camera_T'], cameras['camera_Intri']

        for v in range(V):
            pts_cam = torch.matmul(pts_3d, Rs[:, v].transpose(1, 2)) + Ts[:, v].unsqueeze(1)

            z = pts_cam[..., 2].clamp(min=1e-5)
            fx, fy = Ks[:, v, 0, 0].unsqueeze(1), Ks[:, v, 1, 1].unsqueeze(1)
            cx, cy = Ks[:, v, 0, 2].unsqueeze(1), Ks[:, v, 1, 2].unsqueeze(1)

            u_norm = 2.0 * (pts_cam[..., 0] * fx / z + cx) / (img_w - 1) - 1.0
            v_norm = 2.0 * (pts_cam[..., 1] * fy / z + cy) / (img_h - 1) - 1.0

            valid_mask_flat = ((u_norm >= -1.0) & (u_norm <= 1.0) &
                               (v_norm >= -1.0) & (v_norm <= 1.0) &
                               (pts_cam[..., 2] > 1e-3)).float()

            sample_grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(1)
            del u_norm, v_norm, pts_cam, z  # 立即释放投影中间变量

            vol_v = F.grid_sample(features[:, v], sample_grid, align_corners=False, padding_mode='zeros')
            vol_v = vol_v.view(B, C, D, H, W)
            del sample_grid

            valid_mask = valid_mask_flat.view(B, 1, D, H, W)
            del valid_mask_flat

            # 原地累加, 避免创建新张量
            volume_sum.add_(vol_v * valid_mask)
            weight_sum.add_(valid_mask)
            del vol_v, valid_mask

        volume_fused = volume_sum / (weight_sum + 1e-5)
        del volume_sum, weight_sum
        return volume_fused

    def soft_argmax_3d(self, volumes):
        B, J, D, H, W = volumes.shape
        probs = F.softmax(volumes.view(B, J, -1), dim=-1).view(B, J, D, H, W)
        del volumes  # 释放原始 logit 体素

        device = probs.device
        r = torch.linspace(0, 1, self.vol_size, device=device)

        # 沿各轴逐次求期望, 避免同时持有三个完整的 128³ 网格
        # X 轴: 沿 W 维度加权求和后, 对 D,H 维度求和
        coord_x = torch.sum(probs * r.view(1, 1, 1, 1, -1), dim=(2, 3, 4))
        # Y 轴: 沿 H 维度加权求和
        coord_y = torch.sum(probs * r.view(1, 1, 1, -1, 1), dim=(2, 3, 4))
        # Z 轴: 沿 D 维度加权求和
        coord_z = torch.sum(probs * r.view(1, 1, -1, 1, 1), dim=(2, 3, 4))
        del probs

        coords_norm = torch.stack([coord_x, coord_y, coord_z], dim=-1)

        size = self.space_size.view(1, 1, 3)
        center = self.space_center.view(1, 1, 3)
        return (coords_norm - 0.5) * size + center

    def forward(self, views, meta=None):
        B = views[0].shape[0]
        V = len(views)
        device = views[0].device
        img_size = (views[0].shape[3], views[0].shape[2])

        # 1. 2D 特征与关节点提取
        images_reshaped = torch.cat(views, dim=0)
        feats, joints_2d = self.backbone(images_reshaped)

        feats = self.process_features(feats)
        _, C, Hf, Wf = feats.shape
        feats = feats.view(B, V, C, Hf, Wf)
        joints_2d = joints_2d.view(B, V, self.num_joints, 2)

        raw_R = torch.stack([m['camera_R'] for m in meta])
        raw_T = torch.stack([m['camera_T'] for m in meta])
        raw_K = torch.stack([m['camera_Intri'] for m in meta])

        if meta is not None and 'joints_3d' in meta[0]:
            gt_poses_world = torch.stack([m['joints_3d'] for m in meta])
            if gt_poses_world.dim() == 4:
                gt_poses_world = gt_poses_world.squeeze(1)
            gt_poses_world = gt_poses_world.to(device)
            root_world = gt_poses_world[:, 0, :]
        else:
            root_world = torch.zeros(B, 3).to(device)

        # 坐标系相对平移
        root_expanded = root_world.view(B, 1, 3, 1)
        T_rotated = torch.matmul(raw_R, root_expanded).squeeze(-1)
        new_T = T_rotated + raw_T

        relative_cameras = {'camera_R': raw_R, 'camera_T': new_T, 'camera_Intri': raw_K}

        # 2. 3D 体素构建与多视角特征投影融合
        grid_3d = self._build_grid(B, device)
        volume_fused = self._unproject(feats, relative_cameras, grid_3d, img_size)

        # 3. V2V 3D CNN 解码
        volume_out = self.volume_net(volume_fused)
        pred_3d_rel = self.soft_argmax_3d(volume_out)

        pred_3d_world = pred_3d_rel + root_world.unsqueeze(1)

        outputs = {
            'joints_2d': joints_2d,
            'pred_poses': {'outputs_coord': pred_3d_world.unsqueeze(1)},
            'final_pred_poses': pred_3d_world
        }

        # 4. 损失计算
        loss_dict = {}
        if self.training and meta is not None:
            # [A] 3D 相对坐标 L1 Loss
            gt_poses_rel = gt_poses_world - root_world.unsqueeze(1)
            weight_3d = self.cfg.DECODER.loss_pose_perjoint
            loss_dict['loss_pose_perjoint'] = F.l1_loss(pred_3d_rel, gt_poses_rel) * weight_3d

            # [B] 2D 辅助监督 Loss
            gt_joints_2d_proj = self._project_3d_to_2d_batch(
                gt_poses_world, raw_K,
                torch.cat([raw_R, raw_T.unsqueeze(-1)], dim=-1)
            )
            gt_u = gt_joints_2d_proj[..., 0]
            gt_v = gt_joints_2d_proj[..., 1]
            gt_z = gt_joints_2d_proj[..., 2]

            vis_mask = ((gt_u > 0) & (gt_u < img_size[0]) &
                        (gt_v > 0) & (gt_v < img_size[1]) &
                        (gt_z > 0.05)).float()

            loss_joints_2d_raw = F.l1_loss(joints_2d, gt_joints_2d_proj[..., :2], reduction='none')
            loss_joints_2d_masked = loss_joints_2d_raw * vis_mask.unsqueeze(-1)
            loss_dict['loss_joints_2d'] = loss_joints_2d_masked.sum() / (vis_mask.sum() + 1e-6) * 5.0

        return outputs, loss_dict

    def _project_3d_to_2d_batch(self, points_3d, K, E):
        B, N, _ = points_3d.shape
        V = K.shape[1]
        pts = points_3d.unsqueeze(1).expand(-1, V, -1, -1)
        R = E[..., :3]
        T = E[..., 3]
        pts_cam = torch.matmul(pts, R.transpose(-1, -2)) + T.unsqueeze(2)
        z = pts_cam[..., 2].clamp(min=1e-5)
        x, y = pts_cam[..., 0], pts_cam[..., 1]
        fx, fy = K[..., 0, 0].unsqueeze(2), K[..., 1, 1].unsqueeze(2)
        cx, cy = K[..., 0, 2].unsqueeze(2), K[..., 1, 2].unsqueeze(2)
        u = x * fx / z + cx
        v = y * fy / z + cy
        return torch.stack([u, v, z], dim=-1)


def get_lvt_model(cfg):
    return LVT(cfg)