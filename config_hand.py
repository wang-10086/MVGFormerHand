# config_hand.py
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# 1. 空间与几何 (Hand-Specific)
# -----------------------------------------------------------------------------
_C.MULTI_PERSON = CN()
# 定义 3D 搜索空间的物理尺寸 (单位: 米)
# 假设手部活动范围是一个 0.8m 的立方体
_C.MULTI_PERSON.SPACE_SIZE = [0.8, 0.8, 0.8]
# 空间的中心坐标 (单位: 米)，根据你的世界坐标系原点调整
# 如果 DexYCB 的世界坐标原点在桌面上，这里设为 [0.0, 0.0, 0.2] 可能比较合适
_C.MULTI_PERSON.SPACE_CENTER = [0.0, 0.0, 0.0]
_C.MULTI_PERSON.INITIAL_CUBE_SIZE = [64, 64, 64] # 空间网格划分密度
_C.MULTI_PERSON.MAX_PEOPLE_NUM = 1 # 只有一只手

# -----------------------------------------------------------------------------
# 2. 网络结构 (MVGFormer Core)
# -----------------------------------------------------------------------------
_C.NETWORK = CN()
_C.NETWORK.IMAGE_SIZE = [256, 256] # 输入图像尺寸
_C.NETWORK.NUM_JOINTS = 21         # 手部 21 个关键点

_C.BACKBONE_MODEL = 'pose_resnet'
_C.POSE_RESNET = CN()
_C.POSE_RESNET.NUM_LAYERS = 50     # 使用 ResNet50 提取特征

# -----------------------------------------------------------------------------
# 3. Transformer 解码器配置
# -----------------------------------------------------------------------------
_C.DECODER = CN()
_C.DECODER.d_model = 256
_C.DECODER.nhead = 8
_C.DECODER.dim_feedforward = 1024
_C.DECODER.dropout = 0.1
_C.DECODER.activation = 'relu'
_C.DECODER.num_feature_levels = 3   # 使用 ResNet 的 Layer 2,3,4 多尺度特征
_C.DECODER.dec_n_points = 4         # Deformable Attention 采样点数
_C.DECODER.num_decoder_layers = 4   # 迭代层数
_C.DECODER.return_intermediate_dec = True
_C.DECODER.num_instance = 10        # Query 数量：设为 10 个足矣，最后取置信度最高的
_C.DECODER.num_keypoints = 21       # 对应 21 个关节 Query
_C.DECODER.with_pose_refine = True
_C.DECODER.aux_loss = True          # 使用辅助 Loss
_C.DECODER.pred_class_fuse = 'mean'
_C.DECODER.pred_conf_threshold = 0.5
_C.DECODER.match_coord_est = 'abs'  # 预测绝对坐标
_C.DECODER.match_coord_gt = 'abs'   # GT 也是绝对坐标
_C.DECODER.fuse_view_feats = 'cat_proj'
_C.DECODER.projattn_posembed_mode = 'use_rayconv' # 启用 Ray Convolution 增强几何感知
_C.DECODER.query_embed_type = 'person_joint'      # Query 构成方式
_C.DECODER.query_adaptation = True                # 启用 Query 自适应
_C.DECODER.use_feat_level = [0, 1, 2]
_C.DECODER.share_layer_weights = False
_C.DECODER.pose_embed_layer = 3
_C.DECODER.loss_joint_type = 'l1'
# -----------------------------------------------------------------------------
# 4. 训练与损失函数
# -----------------------------------------------------------------------------
_C.DECODER.loss_pose_perjoint = 10.0            # 3D L1 Loss 权重
_C.DECODER.loss_pose_perprojection_2d = 5.0     # 2D 投影 Loss 权重 (对手部很重要)
_C.DECODER.loss_weight_loss_ce = 1.0            # 分类 Loss (判断是不是手)
_C.DECODER.loss_weight_init = 1.0
_C.DECODER.use_loss_pose_perbone = False
_C.DECODER.use_loss_pose_perprojection = False
_C.DECODER.use_loss_pose_perprojection_2d = True
_C.DECODER.loss_pose_normalize = False          # 直接回归物理坐标(米)

_C.DECODER.match_method = 'hungarian'           # 匈牙利匹配
_C.DECODER.match_method_value = 5
_C.DECODER.cost_class = 2.0
_C.DECODER.cost_pose = 5.0

# -----------------------------------------------------------------------------
# 5. 数据集参数
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.CAMERA_NUM = 8      # 你的 DexYCB 是 8 视角
_C.DATASET.ROOTIDX = 0         # 根关节索引 (Wrist)
_C.DATASET.TEST_DATASET = 'dexycb'