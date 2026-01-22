import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------------------------------------
# 1. 基础组件与几何工具 (Geometry Utils)
# ------------------------------------------------------------------------------------------------

class MLP(nn.Module):
    """简单的多层感知机"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def batch_project_points(points_3d, cameras, img_size):
    """
    将 3D 点投影到 2D 图像平面并归一化到 [-1, 1] (供 grid_sample 使用)
    Args:
        points_3d: (B, N_query, 3) 绝对坐标
        cameras: List of dicts, len=Batch. 每个 dict 包含 'R', 'T', 'K'
        img_size: (H, W)
    Returns:
        sampling_grid: (B, N_view, N_query, 1, 2) 归一化坐标 (x, y)
        mask: (B, N_view, N_query, 1) 点是否在图像内的掩码
    """
    B = points_3d.shape[0]
    N_q = points_3d.shape[1]

    # 假设所有样本的视角数量相同
    # 提取相机参数堆叠
    # camera_R: (B, V, 3, 3)
    # camera_T: (B, V, 3)
    # camera_K: (B, V, 3, 3)

    # 这里为了通用性，我们需要从 meta list 中重新堆叠参数
    # 如果 adapter 已经处理好 stacked tensor，可以直接传入
    # 这里假设 cameras 参数已经是堆叠好的 Tensor (B, V, ...)
    Rs = cameras['camera_R']  # (B, V, 3, 3)
    Ts = cameras['camera_T']  # (B, V, 3)
    Ks = cameras['camera_Intri']  # (B, V, 3, 3)

    V = Rs.shape[1]

    # 1. 扩展 3D 点: (B, 1, N_q, 3) -> (B, V, N_q, 3)
    pts = points_3d.unsqueeze(1).expand(-1, V, -1, -1)

    # 2. World -> Camera: X_cam = (X_world - T) @ R.T  或者 X_cam = X_world @ R.T + T
    # 注意 MVGFormer 原版逻辑通常是: X_cam = (X_world - T) @ R (若R是row-major) 或 X_cam = (X_world) @ R.T + T
    # 这里的实现取决于 Adapter 中传进来的 R, T 定义。
    # 假设 Adapter 传进来的是 World2Cam: X_cam = R * X_world + T
    # 对应 Tensor 运算: X_cam = (R @ X_world.T).T + T
    # 为了高效，通常写成: X_cam = X_world @ R.T + T

    # pts: (B, V, N, 3), Rs: (B, V, 3, 3)
    # 矩阵乘法: (B, V, N, 3) x (B, V, 3, 3)^T -> (B, V, N, 3)
    pts_cam = torch.matmul(pts, Rs.transpose(-1, -2)) + Ts.unsqueeze(2)

    # 3. Camera -> Pixel: u = fx*X/Z + cx
    # 避免除以 0
    eps = 0.1
    z = pts_cam[..., 2].clamp(min=eps)
    x = pts_cam[..., 0]
    y = pts_cam[..., 1]

    fx = Ks[..., 0, 0].unsqueeze(2)
    fy = Ks[..., 1, 1].unsqueeze(2)
    cx = Ks[..., 0, 2].unsqueeze(2)
    cy = Ks[..., 1, 2].unsqueeze(2)

    u = x * fx / z + cx
    v = y * fy / z + cy

    # 4. 归一化到 [-1, 1]
    H, W = img_size
    u_norm = 2.0 * u / (W - 1) - 1.0
    v_norm = 2.0 * v / (H - 1) - 1.0

    # 5. 生成 Mask (判断点是否在图像范围内)
    in_view = (u_norm > -1.0) & (u_norm < 1.0) & (v_norm > -1.0) & (v_norm < 1.0) & (pts_cam[..., 2] > 0)

    # Stack 为 grid: (B, V, N_q, 2)
    grid = torch.stack((u_norm, v_norm), dim=-1)

    return grid, in_view


# ------------------------------------------------------------------------------------------------
# 2. 投影注意力模块 (Projective Attention) - 纯 PyTorch 实现
# ------------------------------------------------------------------------------------------------

class ProjectiveAttention(nn.Module):
    """
    纯 PyTorch 实现的投影注意力。
    替代了原版的 CUDA Deformable Attention。
    通过 grid_sample 直接在投影点采样特征，并结合 Self-Attention 融合多视角信息。
    """

    def __init__(self, d_model, n_heads, n_levels, n_views, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_views = n_views

        # 特征融合层
        self.attention_weights = nn.Linear(d_model, n_levels * n_views)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # 这是一个简化版的实现：直接采样投影点，不使用可学习的 offsets (Deformable)
        # 如果需要 Deformable，可以在 grid 上加一个 learned offset

    def forward(self, query, reference_points_3d, views_feats, cameras, img_size):
        """
        Args:
            query: (B, N_query, C) - 3D Query Features
            reference_points_3d: (B, N_query, 3) - 3D Coordinates
            views_feats: List of tensors [Level1(B, V, C, H1, W1), Level2...]
            cameras: Meta info containing K, R, T
        """
        B, N_q, C = query.shape

        # 1. 将 3D 参考点投影到所有视角的 2D 平面
        # grid: (B, V, N_q, 2)
        grid_base, mask = batch_project_points(reference_points_3d, cameras, img_size)

        all_sampled_feats = []

        # 2. 多尺度特征采样
        for lvl, feats in enumerate(views_feats):
            # feats: (B, V, C, H, W) -> Collapse B*V -> (B*V, C, H, W)
            V = feats.shape[1]
            H, W = feats.shape[3], feats.shape[4]

            feats_reshaped = feats.flatten(0, 1)

            # grid: (B, V, N_q, 2) -> (B*V, N_q, 1, 2)
            grid_lvl = grid_base.flatten(0, 1).unsqueeze(2)

            # 使用 grid_sample 采样特征
            # sampled: (B*V, C, N_q, 1)
            sampled = F.grid_sample(feats_reshaped, grid_lvl, align_corners=False, padding_mode='zeros')

            # Reshape back: (B, V, C, N_q) -> (B, V, N_q, C)
            sampled = sampled.view(B, V, C, N_q).permute(0, 1, 3, 2)

            all_sampled_feats.append(sampled)

        # 3. 融合多尺度特征
        # 简单求和或者拼接，这里我们简单平均多尺度，重点在多视角融合
        # (B, V, N_q, C)
        multi_scale_feat = torch.stack(all_sampled_feats, dim=0).mean(0)

        # 4. 视角融合 (View Fusion)
        # 原版有多种融合方式，这里使用 Query-Aware 的加权融合

        # Mask out points outside image
        # mask: (B, V, N_q) -> (B, V, N_q, 1)
        valid_mask = mask.unsqueeze(-1).float()
        multi_scale_feat = multi_scale_feat * valid_mask

        # 简单平均融合 (也可以用 Attention)
        # (B, N_q, C)
        fused_feat = multi_scale_feat.sum(dim=1) / (valid_mask.sum(dim=1).clamp(min=1.0))

        # Residual Connection
        output = query + self.dropout(self.output_proj(fused_feat))
        output = self.norm(output)

        return output


# ------------------------------------------------------------------------------------------------
# 3. 解码器层与解码器 (Transformer Decoder)
# ------------------------------------------------------------------------------------------------

class MVGDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # Self Attention (Query 之间的交互)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Projective Attention (Query 与 图像特征的交互)
        # 参数将在模型构建时传入
        self.proj_attn = None  # 稍后初始化

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, reference_points, src_feats, cameras, img_size):
        # 1. Self Attention
        # tgt: (B, N_q, C), query_pos: (B, N_q, C)
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2. Projective Attention (Cross Attention with Image)
        # reference_points: (B, N_q, 3) 绝对坐标
        tgt2 = self.proj_attn(tgt, reference_points, src_feats, cameras, img_size)
        # Residual 在 proj_attn 内部已做 (通常), 但为了规范这里假设 proj_attn 返回的是 fused features
        # 上面的 ProjectiveAttention 实现里已经加了 residual 和 norm，所以直接赋值
        tgt = tgt2

        # 3. FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


class MVGDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.DECODER.d_model
        self.n_layers = cfg.DECODER.num_decoder_layers

        # 构建层
        self.layers = nn.ModuleList([
            MVGDecoderLayer(d_model=self.d_model) for _ in range(self.n_layers)
        ])

        # 初始化 Projective Attention 模块
        for layer in self.layers:
            layer.proj_attn = ProjectiveAttention(
                d_model=self.d_model,
                n_heads=cfg.DECODER.nhead,
                n_levels=len(cfg.DECODER.use_feat_level),
                n_views=cfg.DATASET.CAMERA_NUM
            )

        # 预测头 (Pose Regression Head)
        # 用于每一层预测 offset 更新 reference points
        self.pose_embed = MLP(self.d_model, self.d_model, 3, 3)
        # 分类头 (用于判断是不是手/关节置信度)
        self.class_embed = nn.Linear(self.d_model, 1)  # 手部任务通常是 1 类

    def forward(self, tgt, query_pos, reference_points, src_feats, cameras, img_size):
        output = tgt

        all_pred_poses = []
        all_pred_logits = []

        # 迭代更新
        for layer in self.layers:
            # 1. Transformer Layer Forward
            output = layer(output, query_pos, reference_points, src_feats, cameras, img_size)

            # 2. 预测坐标更新 (Delta)
            # output: (B, N_q, C)
            delta_pose = self.pose_embed(output)

            # 更新 Reference Points (绝对坐标)
            reference_points = reference_points + delta_pose

            # 3. 预测分类 Logits
            logits = self.class_embed(output)

            all_pred_poses.append(reference_points)
            all_pred_logits.append(logits)

        return torch.stack(all_pred_logits), torch.stack(all_pred_poses)


# ------------------------------------------------------------------------------------------------
# 4. Backbone (ResNet)
# ------------------------------------------------------------------------------------------------

class Backbone(nn.Module):
    def __init__(self, name='resnet50', pretrained=True):
        super().__init__()
        # 加载标准 ResNet
        if name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
        else:
            backbone = torchvision.models.resnet18(pretrained=pretrained)

        # 提取特征层: Layer 2, 3, 4 (Stride 8, 16, 32)
        self.body = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1,
                          backbone.layer2),
            backbone.layer3,
            backbone.layer4
        ])

        # 1x1 Conv 统一通道数到 d_model
        # ResNet50 channels: 512, 1024, 2048
        # ResNet18 channels: 128, 256, 512
        if name == 'resnet50':
            in_channels = [512, 1024, 2048]
        else:
            in_channels = [128, 256, 512]

        self.input_proj = nn.ModuleList([
            nn.Conv2d(c, 256, kernel_size=1) for c in in_channels
        ])

    def forward(self, x):
        # x: (B*V, 3, H, W)
        features = []
        for i, layer in enumerate(self.body):
            x = layer(x)
            features.append(self.input_proj[i](x))
        return features  # List of [C=256, H, W] tensors


class HungarianMatcher(nn.Module):
    """
    基于匈牙利算法的二分图匹配器
    Cost = cost_class * (-prob) + cost_pose * L1_dist
    """

    def __init__(self, cost_class=2.0, cost_pose=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_pose = cost_pose

    @torch.no_grad()
    def forward(self, pred_logits, pred_poses, gt_poses):
        """
        Args:
            pred_logits: (B, N_inst, 1) 每个实例的平均置信度 (经过 Sigmoid)
            pred_poses: (B, N_inst, 21, 3) 预测的 3D 姿态
            gt_poses: (B, 21, 3) 真实的 3D 姿态 (假设单手)
        Returns:
            indices: List[Tuple], 长度为 B. [(src_idx, tgt_idx), ...]
        """
        bs, num_inst = pred_logits.shape[:2]
        indices = []

        # 循环处理 batch (scipy 不支持 batch)
        for b in range(bs):
            # 1. 准备数据
            # prob: (N_inst, 1)
            p_prob = pred_logits[b]
            # pose: (N_inst, 21, 3)
            p_pose = pred_poses[b]
            # gt: (21, 3) -> (1, 21, 3) 用于广播
            t_pose = gt_poses[b].unsqueeze(0)

            # 2. 计算代价 (Cost)
            # A. 分类代价: 我们希望 prob 越大越好 -> -prob 越小越好
            C_class = -p_prob

            # B. 姿态代价: L1 距离
            # sum(dim=(1,2)) 将 21 个关节和 xyz 维度求和 -> (N_inst,)
            C_pose = torch.abs(p_pose - t_pose).sum(dim=(1, 2)).unsqueeze(-1)  # (N_inst, 1)

            # C. 总代价
            C = self.cost_class * C_class + self.cost_pose * C_pose

            # 3. 匈牙利匹配
            # linear_sum_assignment 寻找最小代价的指派
            # C squeeze 后变成 (N_inst,) 向量，scipy 会自动找最小的那个索引
            C_cpu = C.squeeze(-1).cpu().numpy().reshape(-1, 1)
            row_ind, col_ind = linear_sum_assignment(C_cpu)

            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices

# ------------------------------------------------------------------------------------------------
# 5. 主模型 (MVGFormerHand)
# ------------------------------------------------------------------------------------------------

class MVGFormerHand(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1. Backbone
        self.backbone = Backbone(name='resnet50', pretrained=True)

        # 2. Decoder
        self.decoder = MVGDecoder(cfg)

        # 3. Learnable Query Embeddings
        # num_queries = num_instance * num_joints
        # 即使是单手，我们也保留 num_instance 维度以便扩展
        self.num_instance = cfg.DECODER.num_instance
        self.num_joints = cfg.DECODER.num_keypoints  # 21

        # Joint Embedding (21 个关节点的语义)
        self.joint_embed = nn.Embedding(self.num_joints, cfg.DECODER.d_model)
        # Instance Embedding (区分不同的手/候选)
        self.instance_embed = nn.Embedding(self.num_instance, cfg.DECODER.d_model)

        # 4. Reference Points 初始化
        # 可以是随机初始化，或者初始化在空间中心
        # 这里使用 Learnable Linear Layer 从 Query 生成初始位置
        self.ref_point_head = nn.Linear(cfg.DECODER.d_model, 3)

        self.matcher = HungarianMatcher(cost_class=2.0, cost_pose=5.0)

    def forward(self, views, meta):
        """
        Args:
            views: List of tensors [(B, 3, H, W), ...] V 个
            meta: List of dicts (len B) 包含相机参数
        Returns:
            out: {'pred_logits': ..., 'pred_poses': ...}
            loss_dict: 计算好的 loss
        """
        # 1. 准备图像特征
        B = views[0].shape[0]
        V = len(views)
        views_tensor = torch.stack(views, dim=1)  # (B, V, C, H, W)
        views_flat = views_tensor.flatten(0, 1)  # (B*V, C, H, W)
        features_flat = self.backbone(views_flat)
        features = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in features_flat]

        # 2. 准备 Queries
        joint_embed = self.joint_embed.weight.unsqueeze(0)  # (1, 21, C)
        inst_embed = self.instance_embed.weight.unsqueeze(1)  # (N_inst, 1, C)
        query_embed = (joint_embed + inst_embed).flatten(0, 1)  # (N_inst*21, C)
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(query_embed)

        # 3. 初始化 Reference Points
        reference_points = self.ref_point_head(query_embed).sigmoid()
        space_size = torch.tensor(self.cfg.MULTI_PERSON.SPACE_SIZE).to(reference_points.device)
        space_center = torch.tensor(self.cfg.MULTI_PERSON.SPACE_CENTER).to(reference_points.device)
        reference_points = (reference_points - 0.5) * space_size + space_center

        # 4. Decoder Forward
        cameras_stacked = {
            'camera_R': torch.stack([m['camera_R'] for m in meta]),
            'camera_T': torch.stack([m['camera_T'] for m in meta]),
            'camera_Intri': torch.stack([m['camera_Intri'] for m in meta])
        }
        img_size = (views[0].shape[2], views[0].shape[3])
        all_logits, all_poses = self.decoder(tgt, query_embed, reference_points, features, cameras_stacked, img_size)

        outputs = {
            'pred_logits': all_logits[-1],
            'pred_poses': {'outputs_coord': all_poses[-1]},
            'aux_outputs': [
                {'pred_logits': all_logits[i], 'pred_poses': {'outputs_coord': all_poses[i]}}
                for i in range(len(all_logits) - 1)
            ]
        }

        # 5. 损失计算
        loss_dict = {}

        # 确保处于训练模式且有 GT
        if self.training and meta is not None:
            # -----------------------------------------------------------------
            # 1. 数据准备
            # -----------------------------------------------------------------

            # 获取 GT: (B, 21, 3)
            device = outputs['pred_logits'].device
            gt_poses = torch.stack([m['joints_3d'] for m in meta])
            if gt_poses.dim() == 4: gt_poses = gt_poses.squeeze(1)  # (B, 21, 3)
            gt_poses = gt_poses.to(device)  # 确保在 GPU

            # 获取预测并 Reshape
            # 原始 pred_poses 通常是 (B, N_inst*21, 3)
            # 我们需要把它变成 (B, N_inst, 21, 3) 以便按“只手”进行匹配
            pred_pose_all = outputs['pred_poses']['outputs_coord']
            pred_pose_reshaped = pred_pose_all.view(B, self.num_instance, self.num_joints, 3)

            # 获取分类 Logits 并处理成 Instance 粒度
            # 原始 logits: (B, N_inst*21, 1) -> (B, N_inst, 1)
            # 取 21 个关节的平均分作为这只手的得分
            pred_logits_inst = outputs['pred_logits'].view(B, self.num_instance, self.num_joints, 1).mean(dim=2)
            pred_probs_inst = pred_logits_inst.sigmoid()  # 转概率用于 Matcher

            # -----------------------------------------------------------------
            # 2. 执行匈牙利匹配
            # -----------------------------------------------------------------

            # 获取匹配索引 [(src_idx, tgt_idx), ...]
            # src_idx: 选中的预测手索引, tgt_idx: 对应的 GT 手索引(通常是0)
            match_indices = self.matcher(pred_probs_inst, pred_pose_reshaped, gt_poses)

            # -----------------------------------------------------------------
            # 3. 提取最优预测结果 (用于计算 Loss)
            # -----------------------------------------------------------------

            selected_pred_list = []
            target_classes = torch.zeros(B, self.num_instance).to(device)  # 初始化分类标签全 0

            for b in range(B):
                src_idx, tgt_idx = match_indices[b]
                best_idx = src_idx.item()  # 匹配到的最佳预测索引

                # A. 提取用于回归的 Pose
                selected_pred_list.append(pred_pose_reshaped[b, best_idx])

                # B. 设置分类标签: 只有匹配到的这个是正样本 (1)
                target_classes[b, best_idx] = 1.0

            # 堆叠得到最优预测 (B, 21, 3)
            selected_pred = torch.stack(selected_pred_list)

            # -----------------------------------------------------------------
            # 4. 计算具体 Loss
            # -----------------------------------------------------------------

            # --- Loss A: 3D L1 Loss ---
            loss_dict['loss_pose_perjoint'] = F.l1_loss(selected_pred, gt_poses)

            # --- Loss B: 2D Projection Loss (带安全锁) ---
            # 1. 提取相机参数
            Ks = cameras_stacked['camera_Intri']  # (B, V, 3, 3)
            Rs = cameras_stacked['camera_R']
            Ts = cameras_stacked['camera_T']
            H, W = img_size

            # 2. 预测点 World -> Camera
            # (B, 21, 3) -> (B, V, 21, 3)
            pred_3d_exp = selected_pred.unsqueeze(1).expand(-1, V, -1, -1)
            pred_cam = torch.matmul(pred_3d_exp, Rs.transpose(-1, -2)) + Ts.unsqueeze(2)

            # 3. 深度安全锁 (Z-Clamp)
            pred_z = pred_cam[..., 2].clamp(min=0.1)  # 最小 10cm，防止除零
            pred_x = pred_cam[..., 0]
            pred_y = pred_cam[..., 1]

            # 4. 投影计算
            fx = Ks[..., 0, 0].unsqueeze(2)
            fy = Ks[..., 1, 1].unsqueeze(2)
            cx = Ks[..., 0, 2].unsqueeze(2)
            cy = Ks[..., 1, 2].unsqueeze(2)

            pred_u = pred_x * fx / pred_z + cx
            pred_v = pred_y * fy / pred_z + cy

            # 5. 坐标截断 (Pixel-Clamp) 防止梯度爆炸
            pred_u = pred_u.clamp(min=-W, max=2 * W)
            pred_v = pred_v.clamp(min=-H, max=2 * H)
            pred_uv = torch.stack([pred_u, pred_v], dim=-1)

            # 6. 生成 GT 的 2D 投影 (作为 Target)
            # 重新投影 GT 3D -> 2D
            gt_3d_exp = gt_poses.unsqueeze(1).expand(-1, V, -1, -1)  # (B, V, 21, 3)
            gt_cam = torch.matmul(gt_3d_exp, Rs.transpose(-1, -2)) + Ts.unsqueeze(2)
            # 深度保护
            gt_z = gt_cam[..., 2].clamp(min=0.01)
            # [修改点] 直接使用 fx, cx (形状 B,V,1)，不要 squeeze
            # gt_cam[..., 0] 是 (B, V, 21)
            # fx 是 (B, V, 1)
            # 结果是 (B, V, 21) -> 符合广播规则
            gt_u_target = gt_cam[..., 0] * fx / gt_z + cx
            gt_v_target = gt_cam[..., 1] * fy / gt_z + cy
            gt_uv = torch.stack([gt_u_target, gt_v_target], dim=-1)

            # 计算 2D Loss
            loss_dict['loss_pose_perprojection_2d'] = F.l1_loss(pred_uv, gt_uv)

            # --- Loss C: Classification Loss ---
            # pred_logits_inst.squeeze(-1) shape: (B, N_inst)
            # target_classes shape: (B, N_inst)
            loss_dict['loss_ce'] = F.binary_cross_entropy_with_logits(
                pred_logits_inst.squeeze(-1),
                target_classes
            )

        return outputs, loss_dict


# ------------------------------------------------------------------------------------------------
# 6. 工厂函数 (用于 train.py 调用)
# ------------------------------------------------------------------------------------------------

def get_mvp(cfg, is_train=True):
    model = MVGFormerHand(cfg)
    return model