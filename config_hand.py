# config_hand.py
from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = 'lat'  # 可选: 'mvgformer', 'lat', 'lvt', 'cvf', 'epitrans'

# -----------------------------------------------------------------------------
# 1. 数据集配置 (核心修改)
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = 'dexycb'  # 可选: 'dexycb' 或 'driverhoi'
# _C.DATASET.NAME = 'driverhoi'

# 'subject': 按被试划分 (Train: Subject 0-8, Test: Subject 9) -> 验证泛化性
# 'random':  按比例随机划分 (Train: 90%所有数据, Test: 10%所有数据) -> 验证拟合能力/扩大训练数据
_C.DATASET.SPLIT_STRATEGY = 'random'

# 数据集路径配置
_C.DATASET.ROOT_DEXYCB = '/home/wk/wk/wk/datasets/DexYCB'
_C.DATASET.ROOT_DRIVERHOI = '/home/wk/wk/wk/datasets/DriverHOI3D'

# 相机视角数
_C.DATASET.CAMERA_NUM = 8

# -----------------------------------------------------------------------------
# 2. 空间与几何 (Hand-Specific)
# -----------------------------------------------------------------------------
_C.MULTI_PERSON = CN()
# 3D 搜索空间尺寸 (单位: 米)
_C.MULTI_PERSON.SPACE_SIZE = [4.0, 4.0, 4.0]
# 空间中心 (单位: 米)
_C.MULTI_PERSON.SPACE_CENTER = [0.0, 0.0, 0.0]

# -----------------------------------------------------------------------------
# 3. Transformer 解码器配置
# -----------------------------------------------------------------------------
_C.DECODER = CN()
_C.DECODER.d_model = 256
_C.DECODER.nhead = 8
_C.DECODER.dim_feedforward = 1024
_C.DECODER.dropout = 0.1
_C.DECODER.num_decoder_layers = 4   # 迭代层数
_C.DECODER.num_instance = 100         # Query 数量 (单手设为1)
_C.DECODER.num_keypoints = 21       # 21 个关节 Query
# 使用 ResNet 的哪些层特征 (0=Layer2, 1=Layer3, 2=Layer4)
_C.DECODER.use_feat_level = [0, 1, 2]

# -----------------------------------------------------------------------------
# 4. 匹配代价 (用于匈牙利匹配)
# -----------------------------------------------------------------------------
_C.DECODER.cost_class = 2.0
_C.DECODER.cost_pose = 5.0

# -----------------------------------------------------------------------------
# 5. 训练 Loss 权重
# -----------------------------------------------------------------------------
_C.DECODER.loss_pose_perjoint = 10.0            # 3D L1 Loss
_C.DECODER.loss_pose_perprojection_2d = 1.0     # 2D 投影 Loss
_C.DECODER.loss_weight_loss_ce = 1.0            # 分类 Loss