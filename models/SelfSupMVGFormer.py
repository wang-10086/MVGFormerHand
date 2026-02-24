import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment

# 复用原有的基础组件
from models.MVGFormer import MLP, batch_project_points, ProjectiveAttention, MVGDecoderLayer, Backbone


# ------------------------------------------------------------------------------------------------
# 1. 2D 匈牙利匹配器 (完全不依赖 3D GT)
# ------------------------------------------------------------------------------------------------

class HungarianMatcher2D(nn.Module):
    """
    自监督/弱监督版本的匹配器。
    将预测的 3D 坐标投影到多视角 2D 图像上，计算 2D 重投影误差的 Cost。
    """

    def __init__(self, cost_class=2.0, cost_pose=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_pose = cost_pose

    @torch.no_grad()
    def forward(self, pred_logits, pred_poses_3d, gt_poses_2d, cameras, vis_mask):
        B, N_inst = pred_logits.shape[:2]
        V = gt_poses_2d.shape[1]
        indices = []

        Rs = cameras['camera_R']
        Ts = cameras['camera_T']
        Ks = cameras['camera_Intri']

        for b in range(B):
            p_prob = pred_logits[b]
            p_pose_3d = pred_poses_3d[b]

            p_3d_exp = p_pose_3d.unsqueeze(0).expand(V, -1, -1, -1)

            R_b = Rs[b].unsqueeze(1)
            T_b = Ts[b].unsqueeze(1).unsqueeze(2)

            p_cam = torch.matmul(p_3d_exp, R_b.transpose(-1, -2)) + T_b

            z_safe = p_cam[..., 2].clamp(min=0.05)
            K_b = Ks[b].unsqueeze(1).unsqueeze(2)

            u = p_cam[..., 0] * K_b[..., 0, 0] / z_safe + K_b[..., 0, 2]
            v = p_cam[..., 1] * K_b[..., 1, 1] / z_safe + K_b[..., 1, 2]
            pred_uv = torch.stack([u, v], dim=-1)

            target_uv = gt_poses_2d[b].unsqueeze(1)
            mask_b = vis_mask[b].unsqueeze(1).unsqueeze(-1)

            dist_2d = torch.norm(pred_uv - target_uv, p=1, dim=-1) * mask_b.squeeze(-1)
            cost_pose_2d = dist_2d.sum(dim=(0, 2)) / (mask_b.sum() + 1e-6)

            C_class = -p_prob.squeeze(-1)
            C = self.cost_class * C_class + self.cost_pose * cost_pose_2d

            C_cpu = C.cpu().numpy().reshape(-1, 1)
            row_ind, col_ind = linear_sum_assignment(C_cpu)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices


# ------------------------------------------------------------------------------------------------
# 2. 自监督/解码器
# ------------------------------------------------------------------------------------------------
class SelfSupMVGDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.DECODER.d_model
        self.n_layers = cfg.DECODER.num_decoder_layers

        self.layers = nn.ModuleList([
            MVGDecoderLayer(d_model=self.d_model) for _ in range(self.n_layers)
        ])
        for layer in self.layers:
            layer.proj_attn = ProjectiveAttention(
                d_model=self.d_model, n_heads=cfg.DECODER.nhead,
                n_levels=len(cfg.DECODER.use_feat_level), n_views=cfg.DATASET.CAMERA_NUM
            )
        self.pose_embed = MLP(self.d_model, self.d_model, 3, 3)
        self.class_embed = nn.Linear(self.d_model, 1)

    def forward(self, tgt, query_pos, reference_points, src_feats, cameras, img_size):
        output = tgt
        all_pred_poses, all_pred_logits = [], []
        for layer in self.layers:
            output = layer(output, query_pos, reference_points, src_feats, cameras, img_size)
            delta_pose = self.pose_embed(output)
            reference_points = reference_points + delta_pose
            logits = self.class_embed(output)
            all_pred_poses.append(reference_points)
            all_pred_logits.append(logits)
        return torch.stack(all_pred_logits), torch.stack(all_pred_poses)


# ------------------------------------------------------------------------------------------------
# 3. 主模型 (SelfSupMVGFormerHand)
# ------------------------------------------------------------------------------------------------

class SelfSupMVGFormerHand(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(name='resnet50', pretrained=True)
        self.decoder = SelfSupMVGDecoder(cfg)

        self.num_instance = cfg.DECODER.num_instance
        self.num_joints = cfg.DECODER.num_keypoints
        self.joint_embed = nn.Embedding(self.num_joints, cfg.DECODER.d_model)
        self.instance_embed = nn.Embedding(self.num_instance, cfg.DECODER.d_model)
        self.ref_point_head = nn.Linear(cfg.DECODER.d_model, 3)

        self.matcher = HungarianMatcher2D(
            cost_class=cfg.DECODER.cost_class,
            cost_pose=cfg.DECODER.cost_pose
        )

        self.hand_skeleton = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]

    def forward(self, views, meta):
        B = views[0].shape[0]
        V = len(views)

        # 1. 骨干网络
        views_tensor = torch.stack(views, dim=1)
        views_flat = views_tensor.flatten(0, 1)
        features_flat = self.backbone(views_flat)
        features = [f.view(B, V, f.shape[1], f.shape[2], f.shape[3]) for f in features_flat]

        # 2. Query 准备
        joint_embed = self.joint_embed.weight.unsqueeze(0)
        inst_embed = self.instance_embed.weight.unsqueeze(1)
        query_embed = (joint_embed + inst_embed).flatten(0, 1).unsqueeze(0).expand(B, -1, -1)
        tgt = torch.zeros_like(query_embed)

        # 3. 初始化 3D 锚点
        reference_points = self.ref_point_head(query_embed).sigmoid()
        space_size = torch.tensor(self.cfg.MULTI_PERSON.SPACE_SIZE).to(reference_points.device)
        space_center = torch.tensor(self.cfg.MULTI_PERSON.SPACE_CENTER).to(reference_points.device)
        reference_points = (reference_points - 0.5) * space_size + space_center

        # 4. 解码器
        cameras_stacked = {
            'camera_R': torch.stack([m['camera_R'] for m in meta]),
            'camera_T': torch.stack([m['camera_T'] for m in meta]),
            'camera_Intri': torch.stack([m['camera_Intri'] for m in meta])
        }
        img_size = (views[0].shape[2], views[0].shape[3])
        H, W = img_size

        all_logits, all_poses = self.decoder(tgt, query_embed, reference_points, features, cameras_stacked, img_size)

        outputs = {
            'pred_logits': all_logits[-1],
            'pred_poses': {'outputs_coord': all_poses[-1]}
        }

        # ---------------------------------------------------------------------
        # 后处理与 Loss
        # ---------------------------------------------------------------------
        device = outputs['pred_logits'].device
        pred_pose_final = outputs['pred_poses']['outputs_coord']
        pred_pose_reshaped = pred_pose_final.view(B, self.num_instance, self.num_joints, 3)
        pred_logits_inst = outputs['pred_logits'].view(B, self.num_instance, self.num_joints, 1).mean(dim=2)
        pred_probs_inst = pred_logits_inst.sigmoid()

        loss_dict = {}
        final_selection = []

        if meta is not None:
            gt_3d = torch.stack([m['joints_3d'] for m in meta])
            if gt_3d.dim() == 4: gt_3d = gt_3d.squeeze(1)
            gt_3d = gt_3d.to(device)

            gt_3d_exp = gt_3d.unsqueeze(1).expand(-1, V, -1, -1)
            Ks, Rs, Ts = cameras_stacked['camera_Intri'], cameras_stacked['camera_R'], cameras_stacked['camera_T']
            gt_cam = torch.matmul(gt_3d_exp, Rs.transpose(-1, -2)) + Ts.unsqueeze(2)
            gt_z_safe = gt_cam[..., 2].clamp(min=0.01)
            gt_u = gt_cam[..., 0] * Ks[..., 0, 0].unsqueeze(2) / gt_z_safe + Ks[..., 0, 2].unsqueeze(2)
            gt_v = gt_cam[..., 1] * Ks[..., 1, 1].unsqueeze(2) / gt_z_safe + Ks[..., 1, 2].unsqueeze(2)
            gt_poses_2d = torch.stack([gt_u, gt_v], dim=-1)

            # --- Mask 生成 ---
            mask_z = gt_cam[..., 2] > 0.05
            mask_box = (gt_u > 0) & (gt_u < W) & (gt_v > 0) & (gt_v < H)
            # 计算 Batch 级别的 3D 跨度
            span = gt_3d.max(dim=1)[0] - gt_3d.min(dim=1)[0]  # (B, 3)
            valid_batch_mask = span.sum(dim=-1) > 0.01  # (B,) 真正的有效样本掩码
            # 将 Batch 级别的有效性广播到 (B, V, 21)，用于屏蔽重投影
            mask_valid_gt = valid_batch_mask.unsqueeze(1).unsqueeze(2).expand(-1, V, self.num_joints)
            # 三者同时满足，才是真正需要算 2D 重投影的合法点
            vis_mask = (mask_z & mask_box & mask_valid_gt).float()

            match_indices = self.matcher(pred_probs_inst, pred_pose_reshaped, gt_poses_2d, cameras_stacked, vis_mask)

            target_classes = torch.zeros(B, self.num_instance).to(device)
            for b in range(B):
                src_idx, tgt_idx = match_indices[b]
                best_idx = src_idx.item()
                final_selection.append(pred_pose_reshaped[b, best_idx])

                # 只有真实的、跨度大于 1cm 的手，才被标记为正样本
                if valid_batch_mask[b]:
                    target_classes[b, best_idx] = 1.0

            final_pred_poses = torch.stack(final_selection)

            if self.training:
                # [A] 2D 重投影 Loss
                pred_3d_exp = final_pred_poses.unsqueeze(1).expand(-1, V, -1, -1)
                pred_cam = torch.matmul(pred_3d_exp, Rs.transpose(-1, -2)) + Ts.unsqueeze(2)
                pred_z = pred_cam[..., 2].clamp(min=0.01)
                pred_u = pred_cam[..., 0] * Ks[..., 0, 0].unsqueeze(2) / pred_z + Ks[..., 0, 2].unsqueeze(2)
                pred_v = pred_cam[..., 1] * Ks[..., 1, 1].unsqueeze(2) / pred_z + Ks[..., 1, 2].unsqueeze(2)
                pred_uv = torch.stack([pred_u, pred_v], dim=-1)

                masked_loss = F.l1_loss(pred_uv, gt_poses_2d, reduction='none') * vis_mask.unsqueeze(-1)
                loss_dict['loss_reprojection_2d'] = masked_loss.sum() / (vis_mask.sum() + 1e-6)

                # [B] 置信度 Loss
                loss_dict['loss_ce'] = F.binary_cross_entropy_with_logits(
                    pred_logits_inst.squeeze(-1), target_classes
                )

                # [C] 骨骼结构正则化 Loss (防坍缩)
                # 传入 valid_batch_mask 防止没有手的样本干扰训练
                loss_dict['loss_bone_prior'] = self._compute_bone_prior_loss(final_pred_poses, valid_batch_mask)

        else:
            for b in range(B):
                best_idx = torch.argmax(pred_probs_inst[b]).item()
                final_selection.append(pred_pose_reshaped[b, best_idx])
            final_pred_poses = torch.stack(final_selection)

        outputs['final_pred_poses'] = final_pred_poses
        return outputs, loss_dict

    def _compute_bone_prior_loss(self, poses_3d, valid_batch_mask):
        """
        [修改] L1 骨骼先验防坍缩，同时跳过空样本
        """
        loss = 0.0
        # 只取有手的样本计算
        valid_poses = poses_3d[valid_batch_mask]

        if len(valid_poses) == 0:
            return torch.tensor(0.0, device=poses_3d.device)

        for s, e in self.hand_skeleton:
            bone_len = torch.norm(valid_poses[:, s, :] - valid_poses[:, e, :], dim=-1)
            # 采用 L1 约束 (F.relu 且不平方)
            too_long = F.relu(bone_len - 0.045)
            too_short = F.relu(0.02 - bone_len)
            loss += (too_long.mean() + too_short.mean())

        # 保持大权重，推开重叠点
        return loss * 100


def get_self_sup_mvgformer(cfg, is_train=True):
    return SelfSupMVGFormerHand(cfg)