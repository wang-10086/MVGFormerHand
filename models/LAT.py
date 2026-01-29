import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# -----------------------------------------------------------------------------
# 1. 骨干网络 (Backbone)
# -----------------------------------------------------------------------------
# 为了保持独立性，这里包含了一个标准的 ResNet Backbone 定义。
# 如果您想复用原 model.py 的 Backbone，可以注释掉下面这块并使用: from model import Backbone
class Backbone(nn.Module):
    def __init__(self, name='resnet50', pretrained=True):
        super().__init__()
        # 加载预训练的 ResNet
        if name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            self.out_channels = 2048
        else:
            backbone = torchvision.models.resnet18(pretrained=pretrained)
            self.out_channels = 512

        # 移除最后两层 (GlobalAvgPool 和 FC)
        self.body = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x):
        # 返回最后的特征图 (B, C, H, W)
        return self.body(x)


# -----------------------------------------------------------------------------
# 2. 几何工具函数 (Geometry Utils)
# -----------------------------------------------------------------------------

def soft_argmax_2d(heatmaps, temperature=100.0):
    """
    可微 Soft-Argmax 操作：从热图中提取 (u, v) 坐标和置信度
    Args:
        heatmaps: (B, N_joints, H, W)
    Returns:
        coords: (B, N_joints, 2)  [x, y] 格式，未归一化（像素坐标）
        confidence: (B, N_joints, 1)
    """
    B, N, H, W = heatmaps.shape
    # 平展空间维度
    heatmaps_flat = heatmaps.view(B, N, -1)

    # Softmax 计算概率分布
    probs = F.softmax(heatmaps_flat * temperature, dim=-1)

    # 提取置信度 (最大概率值)
    confidence, _ = torch.max(probs, dim=-1, keepdim=True)  # (B, N, 1)

    # 生成网格坐标
    device = heatmaps.device
    y_range = torch.arange(H, device=device).float()
    x_range = torch.arange(W, device=device).float()
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

    # 计算期望坐标
    grid_x = grid_x.view(-1)
    grid_y = grid_y.view(-1)

    pred_x = torch.sum(probs * grid_x, dim=-1)
    pred_y = torch.sum(probs * grid_y, dim=-1)

    coords = torch.stack([pred_x, pred_y], dim=-1)  # (B, N, 2)

    return coords, confidence


def weighted_algebraic_triangulation(uv, conf, cameras):
    """
    加权代数三角剖分 (Weighted DLT)
    论文核心：利用网络预测的 confidence 对不同视角的方程进行加权，
    从而减弱被遮挡视角的影响。

    Args:
        uv: (B, V, J, 2)  2D 像素坐标
        conf: (B, V, J, 1)  置信度权重
        cameras: 包含相机参数的字典
    Returns:
        X_3d: (B, J, 3)  3D 世界坐标
    """
    B, V, J, _ = uv.shape
    device = uv.device

    # 1. 构建投影矩阵 P = K [R | T]
    # R: (B, V, 3, 3), T: (B, V, 3), K: (B, V, 3, 3)
    R = cameras['camera_R']
    T = cameras['camera_T'].unsqueeze(-1)  # (B, V, 3, 1)
    K = cameras['camera_Intri']

    # 拼接外参 [R|T] -> (B, V, 3, 4)
    Rt = torch.cat([R, T], dim=-1)
    # 投影矩阵 P = K @ [R|T] -> (B, V, 3, 4)
    P = torch.matmul(K, Rt)

    # 2. 构建线性方程组 A X = 0
    # 对于每个关节 j，我们有 2*V 个方程。
    # 方程形式: w * (u * P_row2 - P_row0) * X = 0
    #          w * (v * P_row2 - P_row1) * X = 0

    # 调整维度以支持广播: (B, V, J, ...)
    P = P.unsqueeze(2).expand(-1, -1, J, -1, -1)  # (B, V, J, 3, 4)

    u = uv[..., 0].unsqueeze(-1)  # (B, V, J, 1)
    v = uv[..., 1].unsqueeze(-1)  # (B, V, J, 1)

    # 使用 sqrt(conf) 作为权重，因为最小二乘法是在平方误差上优化的
    w = torch.sqrt(conf + 1e-6)  # (B, V, J, 1)

    # 行 1: u * P[2] - P[0]
    row1 = w * (u * P[..., 2:3, :] - P[..., 0:1, :])  # (B, V, J, 1, 4)
    # 行 2: v * P[2] - P[1]
    row2 = w * (v * P[..., 2:3, :] - P[..., 1:2, :])  # (B, V, J, 1, 4)

    # 拼接构建矩阵 A: (B, V, J, 2, 4)
    A = torch.cat([row1, row2], dim=-2)

    # 调整形状为 (B, J, 2*V, 4) 以便对每个关节单独求解
    A = A.permute(0, 2, 1, 3, 4).reshape(B, J, 2 * V, 4)

    # 3. SVD 求解最小二乘问题
    # 为了批量计算，合并前两个维度: (B*J, 2*V, 4)
    A_flat = A.reshape(B * J, 2 * V, 4)

    # SVD: A = U S V^T
    # 最小奇异值对应的右奇异向量即为解
    # 注意: torch.linalg.svd 返回的 Vh 是 V 的转置 (V^H)
    try:
        _, _, Vh = torch.linalg.svd(A_flat)
        X_homo = Vh[:, -1, :]  # 取最后一行 (B*J, 4)
    except RuntimeError:
        # SVD 可能不收敛的兜底方案
        X_homo = torch.tensor([0., 0., 0., 1.], device=device).expand(B * J, 4)

    # 4. 齐次坐标转欧氏坐标
    X_homo = X_homo.view(B, J, 4)
    X_3d = X_homo[..., :3] / (X_homo[..., 3:4] + 1e-6)

    return X_3d


# -----------------------------------------------------------------------------
# 3. LAT 模型主类 (Model Class)
# -----------------------------------------------------------------------------

class LAT(nn.Module):
    """
    Learnable Algebraic Triangulation (LAT)
    论文: Learnable Triangulation of Human Pose (ICCV 2019) - Method 1
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1. 骨干网络
        self.backbone = Backbone()
        feat_dim = self.backbone.out_channels

        # 2. 2D 热图预测头
        # 简单的卷积层将特征映射为 J 个关节点的热图
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, cfg.DECODER.num_keypoints, kernel_size=1)
        )

        self.num_joints = cfg.DECODER.num_keypoints

    def forward(self, views, meta):
        """
        Args:
            views: List[Tensor], 长度为 V, 每个元素 shape (B, 3, H, W)
            meta: List[Dict], 长度为 V, 包含相机参数
        Returns:
            Dict: {'pred_poses': {'outputs_coord': (B, 1, J, 3)}, ...}
        """
        # --- 准备数据 ---
        B = views[0].shape[0]
        V = len(views)

        # 将所有视角的图像堆叠到 Batch 维度进行并行处理
        # (B*V, 3, H, W)
        x = torch.cat(views, dim=0)

        # --- 1. 特征提取 & 2D 热图预测 ---
        feats = self.backbone(x)
        heatmaps = self.heatmap_head(feats)  # (B*V, J, H_f, W_f)

        _, _, H_feat, W_feat = heatmaps.shape

        # --- 2. 软最大化 (Soft-Argmax) ---
        # 提取像素坐标 (u, v) 和置信度
        uv_flat, conf_flat = soft_argmax_2d(heatmaps)  # uv: (B*V, J, 2)

        # [关键步骤] 坐标缩放
        # Heatmap 的尺寸通常比原图小 (如 1/32)。
        # 投影矩阵 K 是基于原图尺寸定义的，所以必须把预测的 uv 映射回原图尺度。
        # 假设输入图是 256x256, Backbone (ResNet) 下采样 32 倍 -> 特征图 8x8
        # 我们需要根据特征图和原图的比例进行缩放
        orig_H, orig_W = views[0].shape[2], views[0].shape[3]
        scale_factor_x = orig_W / W_feat
        scale_factor_y = orig_H / H_feat

        uv_flat[..., 0] *= scale_factor_x
        uv_flat[..., 1] *= scale_factor_y

        # 恢复 (B, V, J, ...) 维度
        uv = uv_flat.view(B, V, self.num_joints, 2)
        conf = conf_flat.view(B, V, self.num_joints, 1)

        # --- 3. 准备相机参数 ---
        # 堆叠所有 meta 中的参数
        cameras = {
            'camera_R': torch.stack([m['camera_R'] for m in meta]),  # (B, V, 3, 3)
            'camera_T': torch.stack([m['camera_T'] for m in meta]),  # (B, V, 3)
            'camera_Intri': torch.stack([m['camera_Intri'] for m in meta])  # (B, V, 3, 3)
        }

        # --- 4. 可微代数三角剖分 ---
        pred_3d = weighted_algebraic_triangulation(uv, conf, cameras)  # (B, J, 3)

        # --- 5. 格式化输出 ---
        # 为了与 MVGFormer 接口保持一致: (B, num_instance=1, J, 3)
        return {
            'pred_poses': {'outputs_coord': pred_3d.unsqueeze(1)},
            # 这里的 logits 设为全 1，因为 LAT 不是检测模型，没有背景类
            'pred_logits': torch.ones(B, 1, 1).to(pred_3d.device)
        }


# 用于实例化模型的便捷函数
def get_lat_model(cfg):
    model = LAT(cfg)
    return model