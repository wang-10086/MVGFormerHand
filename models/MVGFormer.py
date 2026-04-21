import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------------------------------------
# 1. 基础组件与几何工具 (Geometry Utils)
# ------------------------------------------------------------------------------------------------

class MLP(nn.Module):
    """简单的多层感知机 (Multi-Layer Perceptron)"""

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
    将 3D 点投影到 2D 图像平面并归一化到 [-1, 1] 区间，供 grid_sample 使用。

    Args:
        points_3d (Tensor): (B, N_query, 3) 绝对世界坐标
        cameras (dict): 包含 'camera_R'(B,V,3,3), 'camera_T'(B,V,3), 'camera_Intri'(B,V,3,3)
        img_size (tuple): (H, W) 图像尺寸

    Returns:
        grid (Tensor): (B, V, N_query, 1, 2) 归一化采样坐标
        in_view (Tensor): (B, V, N_query, 1) 点是否在视场内的 Mask
    """
    Rs = cameras['camera_R']
    Ts = cameras['camera_T']
    Ks = cameras['camera_Intri']
    V = Rs.shape[1]

    # 1. 扩展 3D 点: (B, 1, N_q, 3) -> (B, V, N_q, 3)
    pts = points_3d.unsqueeze(1).expand(-1, V, -1, -1)

    # 2. World -> Camera: P_cam = P_world @ R^T + T
    pts_cam = torch.matmul(pts, Rs.transpose(-1, -2)) + Ts.unsqueeze(2)

    # 3. Camera -> Pixel: u = fx*X/Z + cx
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

    # 5. 生成视场 Mask
    in_view = (u_norm > -1.0) & (u_norm < 1.0) & (v_norm > -1.0) & (v_norm < 1.0) & (pts_cam[..., 2] > 0)

    grid = torch.stack((u_norm, v_norm), dim=-1).unsqueeze(-2)  # (B, V, N_q, 1, 2)

    return grid, in_view.unsqueeze(-1)


# ------------------------------------------------------------------------------------------------
# 2. 投影注意力模块 (Projective Attention)
# ------------------------------------------------------------------------------------------------

class ProjectiveAttention(nn.Module):
    """
    基于几何投影的注意力机制：将 3D Query 投影到多视角特征图上采样并融合。
    """

    def __init__(self, d_model, n_heads, n_levels, n_views, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, reference_points_3d, views_feats, cameras, img_size):
        """
        Args:
            query (Tensor): (B, N_query, C)
            reference_points_3d (Tensor): (B, N_query, 3)
            views_feats (list[Tensor]): 多尺度特征图列表, 每个元素 Shape (B, V, C, H, W)
            cameras (dict): 相机参数
            img_size (tuple): (H, W)
        """
        B, N_q, C = query.shape

        # 1. 计算投影采样网格
        grid_base, mask = batch_project_points(reference_points_3d, cameras, img_size)

        all_sampled_feats = []

        # 2. 多尺度特征采样
        for lvl, feats in enumerate(views_feats):
            # feats: (B, V, C, H, W)
            V = feats.shape[1]
            # Flatten Batch & View: (B*V, C, H, W)
            feats_reshaped = feats.flatten(0, 1)

            # Flatten Grid: (B*V, N_q, 1, 2)
            grid_lvl = grid_base.flatten(0, 1)

            sampled = F.grid_sample(feats_reshaped, grid_lvl, align_corners=False, padding_mode='zeros')
            # (B*V, C, N_q, 1) -> (B, V, C, N_q) -> (B, V, N_q, C)
            sampled = sampled.view(B, V, C, N_q).permute(0, 1, 3, 2)
            all_sampled_feats.append(sampled)

        # 3. 简单的特征平均融合 (此处可扩展为 Attention 融合)
        multi_scale_feat = torch.stack(all_sampled_feats, dim=0).mean(0)  # (B, V, N_q, C)

        # 4. 视角融合 (View Fusion)
        valid_mask = mask.float()  # (B, V, N_q, 1)
        multi_scale_feat = multi_scale_feat * valid_mask

        # 对有效视角取平均
        fused_feat = multi_scale_feat.sum(dim=1) / (valid_mask.sum(dim=1).clamp(min=1.0))  # (B, N_q, C)

        # Residual & Norm
        # [Note] 此处的 query 对应解码器层输入的 tgt (内容特征向量)，之所以命名为query是旨在保持注意力机制的签名一致性。
        # 鉴于几何投影已替代了传统的 Q-K 空间寻址，该 query 目前主要作为残差聚合的基底，用于累积新采样的视觉特征。
        # 扩展方向：在局部遮挡场景下，可利用 query 的高阶语义计算动态视角置信度权重。
        output = query + self.dropout(self.output_proj(fused_feat))
        output = self.norm(output)

        return output

class VanillaCrossAttention(nn.Module):
    """
    加强版标准交叉注意力基线。
    改进点：
      1. 将 reference_points_3d 编码为 query 的位置嵌入（不做投影，只提供 3D 空间感知）
      2. 对 KV 特征图做空间池化，大幅缩短序列长度
      3. 保留视角/尺度/空间三层位置编码
    仍然不使用任何相机参数做几何投影。
    """

    def __init__(self, d_model, n_heads, n_levels, n_views, dropout=0.1, pool_factor=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_views = n_views
        self.pool_factor = pool_factor

        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # [改进1] 3D 坐标 → query 位置嵌入
        self.query_pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # [保留] KV 端位置编码
        self.view_embed = nn.Embedding(n_views, d_model)
        self.level_embed = nn.Embedding(n_levels, d_model)

    def _build_2d_sincos_pos(self, H, W, C, device):
        assert C % 4 == 0
        half_C = C // 2
        ys = torch.arange(H, device=device).unsqueeze(1).expand(H, W).reshape(-1).float()
        xs = torch.arange(W, device=device).unsqueeze(0).expand(H, W).reshape(-1).float()
        dim_t = torch.arange(half_C // 2, device=device).float()
        dim_t = 10000.0 ** (2 * dim_t / half_C)
        pe_x = torch.stack([torch.sin(xs[:, None] / dim_t),
                            torch.cos(xs[:, None] / dim_t)], dim=-1).reshape(-1, half_C)
        pe_y = torch.stack([torch.sin(ys[:, None] / dim_t),
                            torch.cos(ys[:, None] / dim_t)], dim=-1).reshape(-1, half_C)
        return torch.cat([pe_x, pe_y], dim=-1)

    def forward(self, query, reference_points_3d, views_feats, cameras, img_size):
        B, N_q, C = query.shape

        # [改进1] 给 query 注入 3D 空间位置感知
        query_pos_3d = self.query_pos_mlp(reference_points_3d)  # (B, N_q, C)
        query = query + query_pos_3d

        # [改进2] 构建紧凑 KV 序列
        kv_list = []
        for lvl, feats in enumerate(views_feats):
            _B, V, _C, H_l, W_l = feats.shape

            # 空间池化: 缩短 KV 长度
            if H_l > self.pool_factor:
                feats_pooled = F.avg_pool2d(
                    feats.flatten(0, 1),          # (B*V, C, H, W)
                    kernel_size=self.pool_factor,
                    stride=self.pool_factor
                )                                  # (B*V, C, H', W')
                feats_pooled = feats_pooled.view(B, V, C,
                    feats_pooled.shape[2], feats_pooled.shape[3])
            else:
                feats_pooled = feats

            H_p, W_p = feats_pooled.shape[3], feats_pooled.shape[4]
            feat_flat = feats_pooled.permute(0, 1, 3, 4, 2).reshape(B, V, H_p * W_p, C)

            # KV 端位置编码
            spatial_pe = self._build_2d_sincos_pos(H_p, W_p, C, feats.device)
            feat_flat = feat_flat + spatial_pe.unsqueeze(0).unsqueeze(0)
            feat_flat = feat_flat + self.view_embed.weight[:V].unsqueeze(0).unsqueeze(2)
            feat_flat = feat_flat + self.level_embed.weight[lvl]

            kv_list.append(feat_flat.reshape(B, V * H_p * W_p, C))

        memory = torch.cat(kv_list, dim=1)

        # 标准交叉注意力
        out, _ = self.cross_attn(query, memory, memory)

        output = query + self.dropout(self.output_proj(out))
        output = self.norm(output)
        return output

# class VanillaCrossAttention(nn.Module):
#     """
#     带位置编码的标准交叉注意力基线。
#     将多视角多尺度特征 flatten 为 KV 序列，加上视角/空间/尺度三层位置编码后，
#     query 通过 nn.MultiheadAttention 做标准 dot-product attention。
#     不使用任何几何投影信息。
#     """
#
#     def __init__(self, d_model, n_heads, n_levels, n_views, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.n_levels = n_levels
#         self.n_views = n_views
#
#         self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
#         self.output_proj = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(d_model)
#
#         # ---- 位置编码 ----
#         # 1) 视角编码: 区分来自哪个相机
#         self.view_embed = nn.Embedding(n_views, d_model)
#         # 2) 尺度编码: 区分来自哪一层特征图
#         self.level_embed = nn.Embedding(n_levels, d_model)
#         # 3) 2D 空间编码: 区分特征图上的空间位置 (正弦编码, 不可学习)
#         #    在 forward 中按实际 H, W 动态生成
#
#     def _build_2d_sincos_pos(self, H, W, C, device):
#         """生成 2D 正弦余弦位置编码 (H*W, C)"""
#         assert C % 4 == 0, "d_model must be divisible by 4 for 2D sincos PE"
#         half_C = C // 2
#         ys = torch.arange(H, device=device).unsqueeze(1).expand(H, W).reshape(-1).float()
#         xs = torch.arange(W, device=device).unsqueeze(0).expand(H, W).reshape(-1).float()
#
#         dim_t = torch.arange(half_C // 2, device=device).float()
#         dim_t = 10000.0 ** (2 * dim_t / half_C)
#
#         pe_x = torch.stack([torch.sin(xs[:, None] / dim_t), torch.cos(xs[:, None] / dim_t)], dim=-1).reshape(-1, half_C)
#         pe_y = torch.stack([torch.sin(ys[:, None] / dim_t), torch.cos(ys[:, None] / dim_t)], dim=-1).reshape(-1, half_C)
#
#         return torch.cat([pe_x, pe_y], dim=-1)  # (H*W, C)
#
#     def forward(self, query, reference_points_3d, views_feats, cameras, img_size):
#         """
#         与 ProjectiveAttention 保持完全相同的函数签名，
#         但忽略 reference_points_3d、cameras、img_size。
#         """
#         B, N_q, C = query.shape
#
#         kv_list = []
#         for lvl, feats in enumerate(views_feats):
#             # feats: (B, V, C, H_l, W_l)
#             _B, V, _C, H_l, W_l = feats.shape
#
#             # 展平空间维: (B, V, C, H, W) -> (B, V, H*W, C)
#             feat_flat = feats.permute(0, 1, 3, 4, 2).reshape(B, V, H_l * W_l, C)
#
#             # 加位置编码
#             spatial_pe = self._build_2d_sincos_pos(H_l, W_l, C, feats.device)  # (H*W, C)
#             feat_flat = feat_flat + spatial_pe.unsqueeze(0).unsqueeze(0)        # 空间位置
#             feat_flat = feat_flat + self.view_embed.weight[:V].unsqueeze(0).unsqueeze(2)  # 视角
#             feat_flat = feat_flat + self.level_embed.weight[lvl]                # 尺度
#
#             # (B, V, H*W, C) -> (B, V*H*W, C)
#             kv_list.append(feat_flat.reshape(B, V * H_l * W_l, C))
#
#         memory = torch.cat(kv_list, dim=1)  # (B, total_tokens, C)
#
#         # 标准交叉注意力
#         out, _ = self.cross_attn(query, memory, memory)
#
#         # 残差 + LayerNorm
#         output = query + self.dropout(self.output_proj(out))
#         output = self.norm(output)
#
#         return output

# ------------------------------------------------------------------------------------------------
# 3. 解码器 (Transformer Decoder)
# ------------------------------------------------------------------------------------------------

class MVGDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 占位符，将在 Decoder 初始化时被 ProjectiveAttention 替换
        self.proj_attn = None

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, reference_points, src_feats, cameras, img_size):
        # Self Attention
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross Attention (Projective)
        tgt2 = self.proj_attn(tgt, reference_points, src_feats, cameras, img_size)
        tgt = tgt2

        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


class MVGDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.DECODER.d_model
        self.n_layers = cfg.DECODER.num_decoder_layers

        self.layers = nn.ModuleList([
            MVGDecoderLayer(d_model=self.d_model) for _ in range(self.n_layers)
        ])

        for layer in self.layers:
            layer.proj_attn = ProjectiveAttention(
                d_model=self.d_model,
                n_heads=cfg.DECODER.nhead,
                n_levels=len(cfg.DECODER.use_feat_level),
                n_views=cfg.DATASET.CAMERA_NUM
            )

        self.pose_embed = MLP(self.d_model, self.d_model, 3, 3)  # 预测坐标偏移量
        self.class_embed = nn.Linear(self.d_model, 1)  # 预测置信度

    def forward(self, tgt, query_pos, reference_points, src_feats, cameras, img_size):
        output = tgt
        all_pred_poses = []
        all_pred_logits = []

        for layer in self.layers:
            output = layer(output, query_pos, reference_points, src_feats, cameras, img_size)

            # 迭代更新 3D 坐标
            delta_pose = self.pose_embed(output)
            reference_points = reference_points + delta_pose

            logits = self.class_embed(output)

            all_pred_poses.append(reference_points)
            all_pred_logits.append(logits)

        return torch.stack(all_pred_logits), torch.stack(all_pred_poses)


# ------------------------------------------------------------------------------------------------
# 4. 骨干网络 (Backbone)
# ------------------------------------------------------------------------------------------------

class Backbone(nn.Module):
    def __init__(self, name='resnet50', pretrained=True):
        super().__init__()
        if name == 'resnet50':
            backbone = torchvision.models.resnet50(pretrained=pretrained)
            in_channels = [512, 1024, 2048]
        else:
            backbone = torchvision.models.resnet18(pretrained=pretrained)
            in_channels = [128, 256, 512]

        self.body = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1,
                          backbone.layer2),
            backbone.layer3,
            backbone.layer4
        ])

        # 通道对齐层 (统一映射到 256 维)
        self.input_proj = nn.ModuleList([
            nn.Conv2d(c, 256, kernel_size=1) for c in in_channels
        ])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.body):
            x = layer(x)
            features.append(self.input_proj[i](x))
        return features


# ------------------------------------------------------------------------------------------------
# 5. 匈牙利匹配器 (Hungarian Matcher)
# ------------------------------------------------------------------------------------------------

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_pose=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_pose = cost_pose

    @torch.no_grad()
    def forward(self, pred_logits, pred_poses, gt_poses):
        bs, num_inst = pred_logits.shape[:2]
        indices = []
        for b in range(bs):
            p_prob = pred_logits[b]
            p_pose = pred_poses[b]
            t_pose = gt_poses[b].unsqueeze(0)

            # Cost Matrix
            C_class = -p_prob
            C_pose = torch.abs(p_pose - t_pose).sum(dim=(1, 2)).unsqueeze(-1)
            C = self.cost_class * C_class + self.cost_pose * C_pose

            C_cpu = C.squeeze(-1).cpu().numpy().reshape(-1, 1)
            row_ind, col_ind = linear_sum_assignment(C_cpu)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))
        return indices


# ------------------------------------------------------------------------------------------------
# 6. 主模型 (MVGFormerHand)
# ------------------------------------------------------------------------------------------------

class MVGFormerHand(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 1. Backbone
        self.backbone = Backbone(name='resnet50', pretrained=True)

        # 2. Decoder
        self.decoder = MVGDecoder(cfg)

        # 3. Learnable Queries
        self.num_instance = cfg.DECODER.num_instance
        self.num_joints = cfg.DECODER.num_keypoints
        self.joint_embed = nn.Embedding(self.num_joints, cfg.DECODER.d_model)
        self.instance_embed = nn.Embedding(self.num_instance, cfg.DECODER.d_model)

        # 4. Init Reference Points Head
        self.ref_point_head = nn.Linear(cfg.DECODER.d_model, 3)

        # 5. Matcher
        self.matcher = HungarianMatcher(
            cost_class=cfg.DECODER.cost_class,
            cost_pose=cfg.DECODER.cost_pose
        )

    def forward(self, views, meta):
        """
        前向传播函数。

        Args:
            views (list[Tensor]): 图像列表, 长度为 V, 每个元素形状 (B, 3, H, W)
            meta (list[dict]): 元数据列表, 长度为 B, 包含相机参数和 GT

        Returns:
            outputs (dict): 包含 'final_pred_poses' (B, 21, 3) 等
            loss_dict (dict): 训练模式下包含各类 Loss
        """
        B = views[0].shape[0]
        V = len(views)

        # 1. Backbone Feature Extraction
        views_tensor = torch.stack(views, dim=1)  # (B, V, 3, H, W)
        views_flat = views_tensor.flatten(0, 1)  # (B*V, 3, H, W)
        features_flat = self.backbone(views_flat)
        # Reshape features back to (B, V, C, H, W)
        features = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in features_flat]

        # 2. Prepare Queries
        joint_embed = self.joint_embed.weight.unsqueeze(0)
        inst_embed = self.instance_embed.weight.unsqueeze(1)
        query_embed = (joint_embed + inst_embed).flatten(0, 1)  # (N_inst * N_joints, C)
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(query_embed)

        # 3. Initialize Reference Points
        reference_points = self.ref_point_head(query_embed).sigmoid()
        space_size = torch.tensor(self.cfg.MULTI_PERSON.SPACE_SIZE).to(reference_points.device)
        space_center = torch.tensor(self.cfg.MULTI_PERSON.SPACE_CENTER).to(reference_points.device)
        reference_points = (reference_points - 0.5) * space_size + space_center

        # 4. Transformer Decoding
        cameras_stacked = {
            'camera_R': torch.stack([m['camera_R'] for m in meta]),
            'camera_T': torch.stack([m['camera_T'] for m in meta]),
            'camera_Intri': torch.stack([m['camera_Intri'] for m in meta])
        }
        img_size = (views[0].shape[2], views[0].shape[3])

        all_logits, all_poses = self.decoder(tgt, query_embed, reference_points, features, cameras_stacked, img_size)

        outputs = {
            'pred_logits': all_logits[-1],
            'pred_poses': {'outputs_coord': all_poses[-1]}
        }

        # 5. Post-Processing & Loss Calculation
        device = outputs['pred_logits'].device
        pred_pose_final = outputs['pred_poses']['outputs_coord']
        pred_pose_reshaped = pred_pose_final.view(B, self.num_instance, self.num_joints, 3)

        # Logits -> Probabilities
        pred_logits_inst = outputs['pred_logits'].view(B, self.num_instance, self.num_joints, 1).mean(dim=2)
        pred_probs_inst = pred_logits_inst.sigmoid()

        loss_dict = {}
        final_selection = []

        # --- 分支 A: 存在 Meta 信息 (训练/有标签验证) ---
        if meta is not None:
            # Prepare GT
            gt_poses = torch.stack([m['joints_3d'] for m in meta])
            if gt_poses.dim() == 4: gt_poses = gt_poses.squeeze(1)  # (B, 21, 3)
            gt_poses = gt_poses.to(device)

            # Hungarian Matching
            match_indices = self.matcher(pred_probs_inst, pred_pose_reshaped, gt_poses)

            target_classes = torch.zeros(B, self.num_instance).to(device)
            for b in range(B):
                src_idx, tgt_idx = match_indices[b]
                best_idx = src_idx.item()
                final_selection.append(pred_pose_reshaped[b, best_idx])
                target_classes[b, best_idx] = 1.0

            final_pred_poses = torch.stack(final_selection)

            # Compute Loss (Only in Training Mode)
            if self.training:
                # 3D L1 Loss
                loss_dict['loss_pose_perjoint'] = F.l1_loss(final_pred_poses,
                                                            gt_poses) * self.cfg.DECODER.loss_pose_perjoint

                # Classification Loss
                loss_dict['loss_ce'] = F.binary_cross_entropy_with_logits(
                    pred_logits_inst.squeeze(-1), target_classes
                ) * self.cfg.DECODER.loss_weight_loss_ce

                # 2D Reprojection Loss
                Ks = cameras_stacked['camera_Intri']
                Rs = cameras_stacked['camera_R']
                Ts = cameras_stacked['camera_T']
                H, W = img_size

                # Project Prediction
                pred_3d_exp = final_pred_poses.unsqueeze(1).expand(-1, V, -1, -1)
                pred_cam = torch.matmul(pred_3d_exp, Rs.transpose(-1, -2)) + Ts.unsqueeze(2)
                pred_z = pred_cam[..., 2].clamp(min=0.01)
                pred_u = pred_cam[..., 0] * Ks[..., 0, 0].unsqueeze(2) / pred_z + Ks[..., 0, 2].unsqueeze(2)
                pred_v = pred_cam[..., 1] * Ks[..., 1, 1].unsqueeze(2) / pred_z + Ks[..., 1, 2].unsqueeze(2)
                pred_uv = torch.stack([pred_u, pred_v], dim=-1)

                # Project GT for Supervision
                gt_3d_exp = gt_poses.unsqueeze(1).expand(-1, V, -1, -1)
                gt_cam = torch.matmul(gt_3d_exp, Rs.transpose(-1, -2)) + Ts.unsqueeze(2)
                gt_z_safe = gt_cam[..., 2].clamp(min=0.01)
                gt_u_target = gt_cam[..., 0] * Ks[..., 0, 0].unsqueeze(2) / gt_z_safe + Ks[..., 0, 2].unsqueeze(2)
                gt_v_target = gt_cam[..., 1] * Ks[..., 1, 1].unsqueeze(2) / gt_z_safe + Ks[..., 1, 2].unsqueeze(2)
                gt_uv = torch.stack([gt_u_target, gt_v_target], dim=-1)

                # Valid Mask (Depth > 5cm & Inside Image)
                mask_z = gt_cam[..., 2] > 0.05
                mask_box = (gt_u_target >= 0) & (gt_u_target < W) & (gt_v_target >= 0) & (gt_v_target < H)
                valid_mask = (mask_z & mask_box).unsqueeze(-1).float()

                masked_loss = F.l1_loss(pred_uv, gt_uv, reduction='none') * valid_mask
                loss_dict['loss_pose_perprojection_2d'] = masked_loss.sum() / (
                            valid_mask.sum() + 1e-6) * self.cfg.DECODER.loss_pose_perprojection_2d

        # --- 分支 B: 无 Meta 信息 (纯推理模式) ---
        else:
            for b in range(B):
                best_idx = torch.argmax(pred_probs_inst[b]).item()
                final_selection.append(pred_pose_reshaped[b, best_idx])
            final_pred_poses = torch.stack(final_selection)

        outputs['final_pred_poses'] = final_pred_poses
        return outputs, loss_dict


def get_mvgformer(cfg, is_train=True):
    return MVGFormerHand(cfg)