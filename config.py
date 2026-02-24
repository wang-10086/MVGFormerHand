import os
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# 1. 模型与架构配置 (Model & Architecture)
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'self_sup_mvgformer'  # 可选: 'mvgformer', 'lat', 'lvt', 'self_sup_mvgformer'
_C.MODEL.BACKBONE = 'resnet50'

# -----------------------------------------------------------------------------
# 2. 网络基础配置 (Network)
# -----------------------------------------------------------------------------
_C.NETWORK = CN()
_C.NETWORK.IMAGE_SIZE = [256, 256]  # [H, W]

# -----------------------------------------------------------------------------
# 3. 数据集配置 (Dataset)
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = 'dexycb'  # 'dexycb' or 'driverhoi'
_C.DATASET.SPLIT_STRATEGY = 'random'  # 'subject' or 'random'
_C.DATASET.ROOT_DEXYCB = '/home/wk/wk/wk/datasets/DexYCB'
_C.DATASET.ROOT_DRIVERHOI = '/home/wk/wk/wk/datasets/DriverHOI3D'
_C.DATASET.CAMERA_NUM = 8  # 使用的视角数量

# -----------------------------------------------------------------------------
# 4. 训练配置 (Train)
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.NUM_WORKERS = 0  # 建议根据CPU核心数调整
_C.TRAIN.LR = 1e-4
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.EPOCHS = 100
_C.TRAIN.LR_STEP_SIZE = 20
_C.TRAIN.LR_GAMMA = 0.5
_C.TRAIN.RESUME_PATH = ''  # 断点续训的模型路径

# -----------------------------------------------------------------------------
# 5. 测试与可视化配置 (Test)
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1
_C.TEST.TEST_CKPT = 'checkpoints/20260130_172533_lvt_dexycb/best_model.pth' # 测试用权重路径
_C.TEST.VIZ = True           # 是否保存可视化结果
_C.TEST.VIZ_FREQ = 10        # 可视化频率 (每N个Batch)
_C.TEST.VIZ_DIR = './test_results/viz_output'

# -----------------------------------------------------------------------------
# 6. 空间与几何配置 (Geometry & Volumetric)
# -----------------------------------------------------------------------------
_C.MULTI_PERSON = CN()
# 3D 空间定义 (单位: 米)
_C.MULTI_PERSON.SPACE_SIZE = [4.0, 4.0, 4.0]
_C.MULTI_PERSON.SPACE_CENTER = [0.0, 0.0, 0.0]
# 体素分辨率 (LVT模型使用)
_C.MULTI_PERSON.VOL_SIZE = 64

# -----------------------------------------------------------------------------
# 7. 解码器与Loss配置 (Decoder & Loss)
# -----------------------------------------------------------------------------
_C.DECODER = CN()
_C.DECODER.d_model = 256
_C.DECODER.nhead = 8
_C.DECODER.dim_feedforward = 1024
_C.DECODER.dropout = 0.1
_C.DECODER.num_decoder_layers = 4
_C.DECODER.num_instance = 100 # Query 数量
_C.DECODER.num_keypoints = 21
_C.DECODER.use_feat_level = [0, 1, 2] # 使用 Backbone 的哪些层级特征

# Loss 权重分配
_C.DECODER.loss_pose_perjoint = 10.0         # 3D L1 Loss
_C.DECODER.loss_pose_perprojection_2d = 1.0  # 2D Reprojection Loss
_C.DECODER.loss_weight_loss_ce = 1.0         # Classification Loss
_C.DECODER.cost_class = 2.0                  # Hungarian Matcher Class Cost
_C.DECODER.cost_pose = 5.0                   # Hungarian Matcher Pose Cost
_C.DECODER.loss_bone_prior = 100.0

# 导出全局配置对象
cfg = _C