import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


class PoseResNet(nn.Module):
    def __init__(self, name='resnet50', pretrained=True):
        super().__init__()
        # 1. 加载预训练 ResNet
        if name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            self.in_channels = 2048
        else:
            backbone = torchvision.models.resnet18(pretrained=pretrained)
            self.in_channels = 512

        # 保留 layer0 到 layer4
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 2. [核心改进] 3层反卷积，将 8x8 上采样到 64x64
        # 结构参考: Simple Baselines for Human Pose Estimation
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 256, 256],
            num_kernels=[4, 4, 4],
        )

        # 3. 初始化权重 (至关重要!)
        self._init_weights()

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = 4, 1, 0
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        # 反卷积层初始化
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)  # [关键] 小方差正态分布
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 上采样: (B, 2048, 8, 8) -> (B, 256, 64, 64)
        x = self.deconv_layers(x)
        return x


# --- V2VNet 3D卷积 (保持不变) ---
class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x): return self.block(x)


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

    def forward(self, x): return F.relu(self.res_branch(x) + self.skip_con(x), True)


class V2VModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        # 使用 7x7 大核增强感受野
        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Res3DBlock(16, 32),
            Res3DBlock(32, 32)
        )
        self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1)
        self._init_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.output_layer(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d): nn.init.xavier_normal_(m.weight)


# --- LVT 主类 ---
class LVT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_joints = cfg.DECODER.num_keypoints

        # 1. 使用高分辨率 Backbone (256x64x64)
        self.backbone = PoseResNet(name='resnet50', pretrained=True)

        # 2. 降维 (256 -> 32)
        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 3. 体素网络
        self.volume_net = V2VModel(32, self.num_joints)

        # 4. 空间参数 (硬编码为相对空间的参数，不受 Config 干扰)
        self.vol_size = 64

        # [关键] 强制设定一个以 0 为中心的小盒子 (40cm)
        # 这确保了无论 Config 怎么写，模型内部都只在这个小范围内搜索
        self.register_buffer('space_size', torch.tensor([0.2, 0.2, 0.2]).float())
        self.register_buffer('space_center', torch.tensor([0.0, 0.0, 0.0]).float())

    def _build_grid(self, batch_size, device):
        r = torch.arange(self.vol_size, device=device).float()
        z, y, x = torch.meshgrid(r, r, r, indexing='ij')
        grid = torch.stack([x, y, z], dim=0) / (self.vol_size - 1)  # (3, D, H, W)

        # (3, 1, 1, 1)
        size = self.space_size.view(3, 1, 1, 1)
        center = self.space_center.view(3, 1, 1, 1)

        grid_3d = (grid - 0.5) * size + center
        return grid_3d.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)

    def _unproject(self, features, cameras, grid_3d):
        B, V, C, Hf, Wf = features.shape
        D, H, W = grid_3d.shape[2:]

        # Flatten Grid: (B, N, 3)
        pts_3d = grid_3d.view(B, 3, -1).permute(0, 2, 1)

        volume_sum = 0

        # 注意：这里的 cameras 必须是已经转换到“相对坐标系”的参数
        Rs = cameras['camera_R']
        Ts = cameras['camera_T']
        Ks = cameras['camera_Intri']

        for v in range(V):
            # World (Relative) -> Cam
            pts_cam = torch.matmul(pts_3d, Rs[:, v].transpose(1, 2)) + Ts[:, v].unsqueeze(1)

            z = pts_cam[..., 2].clamp(min=1e-5)
            x, y = pts_cam[..., 0], pts_cam[..., 1]

            fx, fy = Ks[:, v, 0, 0].unsqueeze(1), Ks[:, v, 1, 1].unsqueeze(1)
            cx, cy = Ks[:, v, 0, 2].unsqueeze(1), Ks[:, v, 1, 2].unsqueeze(1)

            u = x * fx / z + cx
            v_coord = y * fy / z + cy

            # Norm [-1, 1] (假设输入 256x256)
            # 为了更稳健，这里应该从 cfg 读取，或者作为参数传入
            # 这里暂时硬编码 256.0
            u_norm = 2.0 * u / 255.0 - 1.0
            v_norm = 2.0 * v_coord / 255.0 - 1.0

            # Grid Sample
            sample_grid = torch.stack([u_norm, v_norm], dim=-1).view(B, D * H, W, 2)
            feat_v = features[:, v]

            vol_v = F.grid_sample(feat_v, sample_grid, align_corners=False)
            vol_v = vol_v.view(B, C, D, H, W)

            volume_sum += vol_v

        return volume_sum / V

    def soft_argmax_3d(self, volumes):
        B, J, D, H, W = volumes.shape
        volumes = volumes.view(B, J, -1)
        probs = F.softmax(volumes, dim=-1).view(B, J, D, H, W)

        device = volumes.device
        r = torch.linspace(0, 1, self.vol_size, device=device)
        z_g, y_g, x_g = torch.meshgrid(r, r, r, indexing='ij')

        coord_x = torch.sum(probs * x_g.unsqueeze(0).unsqueeze(0), dim=(2, 3, 4))
        coord_y = torch.sum(probs * y_g.unsqueeze(0).unsqueeze(0), dim=(2, 3, 4))
        coord_z = torch.sum(probs * z_g.unsqueeze(0).unsqueeze(0), dim=(2, 3, 4))

        coords_norm = torch.stack([coord_x, coord_y, coord_z], dim=-1)

        size = self.space_size.view(1, 1, 3)
        center = self.space_center.view(1, 1, 3)

        # 得到的是相对坐标
        pred_3d_rel = (coords_norm - 0.5) * size + center
        return pred_3d_rel

    def forward(self, views, meta=None):
        B = views[0].shape[0]
        V = len(views)
        device = views[0].device

        # 1. 特征提取
        x = torch.cat(views, dim=0)
        feats = self.backbone(x)
        feats = self.process_features(feats)
        _, C, Hf, Wf = feats.shape
        feats = feats.view(B, V, C, Hf, Wf)

        # ---------------------------------------------------------------------
        # [核心] 内部处理：Root-Relative 坐标变换
        # ---------------------------------------------------------------------

        # 1. 从 meta 中提取原始相机参数和 GT
        # 注意：这里不需要 adapter 改动，读进来的就是原始世界坐标
        raw_R = torch.stack([m['camera_R'] for m in meta])  # (B, V, 3, 3)
        raw_T = torch.stack([m['camera_T'] for m in meta])  # (B, V, 3)
        raw_K = torch.stack([m['camera_Intri'] for m in meta])

        # 2. 获取 Root (手腕) 坐标
        # 假设 GT 存在 (Training 阶段) 或者使用一个 Mock Root (Testing 阶段)
        if meta is not None and 'joints_3d' in meta[0]:
            gt_poses_world = torch.stack([m['joints_3d'] for m in meta])  # (B, 21, 3)
            if gt_poses_world.dim() == 4: gt_poses_world = gt_poses_world.squeeze(1)
            gt_poses_world = gt_poses_world.to(device)

            # 取手腕 (第0个关节)
            root_world = gt_poses_world[:, 0, :]  # (B, 3)
        else:
            # 推理阶段如果没有 GT，默认设为 0 (或者你需要外挂一个检测器传进来)
            # 这里为了跑通先设为 0
            root_world = torch.zeros(B, 3).to(device)

        # 3. 计算相对相机的平移 T_new
        # 公式: T_new = R * Root + T
        # 扩展维度以便广播: R(B, V, 3, 3), Root(B, 1, 3, 1)
        root_expanded = root_world.view(B, 1, 3, 1)  # (B, 1, 3, 1)
        T_rotated = torch.matmul(raw_R, root_expanded).squeeze(-1)  # (B, V, 3)
        new_T = T_rotated + raw_T  # (B, V, 3)

        relative_cameras = {
            'camera_R': raw_R,
            'camera_T': new_T,  # 使用新的 T
            'camera_Intri': raw_K
        }

        # ---------------------------------------------------------------------
        # 2. 几何计算 (完全在相对坐标系下进行)
        # ---------------------------------------------------------------------

        grid_3d = self._build_grid(B, device)  # 这是一个以 (0,0,0) 为中心的 40cm 盒子

        volume_raw = self._unproject(feats, relative_cameras, grid_3d)
        volume_out = self.volume_net(volume_raw)
        pred_3d_rel = self.soft_argmax_3d(volume_out)  # 得到相对坐标

        # ---------------------------------------------------------------------
        # 3. 结果还原 & 输出
        # ---------------------------------------------------------------------

        # 如果需要输出世界坐标，加回去: Pred_World = Pred_Rel + Root
        pred_3d_world = pred_3d_rel + root_world.unsqueeze(1)

        outputs = {
            'pred_poses': {'outputs_coord': pred_3d_world.unsqueeze(1)},
            'final_pred_poses': pred_3d_world
        }

        loss_dict = {}
        if self.training and meta is not None:
            # 计算 Loss 时，可以直接比较相对坐标，也可以比较世界坐标
            # 这里比较相对坐标更数值稳定
            gt_poses_rel = gt_poses_world - root_world.unsqueeze(1)

            weight = self.cfg.DECODER.loss_pose_perjoint
            loss_dict['loss_pose_perjoint'] = F.l1_loss(pred_3d_rel, gt_poses_rel) * weight

        return outputs, loss_dict


def get_lvt_model(cfg): return LVT(cfg)