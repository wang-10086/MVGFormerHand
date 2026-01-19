import torch


class Config:
    # 数据集配置
    datasets = 'dex-ycb'  # mhp or dex-ycb
    root_dir = '/home/wk/wk/wk/datasets/DexYCB'   # mhp datasets: '/root/wk/datasets/mhp', dex-ycb dataset: '/root/wk/datasets/DexYCB'
    save_dir = './checkpoints'

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MANO模型配置
    mano_model_path = './manopth/mano/models'
    use_mano = True

    # 模型架构配置
    model = 'hamuco'  # baseline or hamuco or mvhandfusion or resnetdlt or resnetdltfusion
    backbone = 'resnet50'  # resnet18 or stacked_hourglass
    num_view = 8

    # 手部模型配置
    root_idx = 0
    hand_root_mode = 0  # 0: use the wrist joint as root, 1: use the palm center as root
    num_joints = 21
    bbox_3d_size = 0.4

    # 数据处理配置
    data_normalization = False
    input_img_shape = (256, 256)

    # 训练参数
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 150
    num_worker = 0
    resume_checkpoint = None
    description = 'use resnetdlt model'

    # 测试
    test_checkpoint = '/root/wk/code/Multi-view_pose_estimation/checkpoints/2025-06-07_15-54-31/best_model.pth'

    random_mask = False


cfg = Config()
