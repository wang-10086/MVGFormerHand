# check_world_consistency.py
import torch
import numpy as np
import warnings

# 忽略警告以便看清输出
warnings.filterwarnings("ignore")

from config_hand import _C as cfg
from datasets.DexYCB import DEXYCBDatasets
from adapter import collate_dexycb_to_mvgformer


def check_world_coord_consistency():
    print(">>> 1. Loading Dataset...")
    # 确保路径正确
    root_dir = '/home/wk/wk/wk/datasets/DexYCB'
    dataset = DEXYCBDatasets(root_dir=root_dir, split='train')

    # 获取第 0 个样本
    inputs, targets, meta = dataset[0]

    # 提取 Dataset 中的原始世界坐标
    # targets['world_coord'] shape: (V, 21, 3)
    raw_world_coord = targets['world_coord']

    print(f"\n[Dataset] World Coords Shape: {raw_world_coord.shape}")
    # 打印第一个关节（通常是手腕）的坐标
    print(f"[Dataset] View 0, Wrist Coords: {raw_world_coord[0, 0]}")
    print(f"[Dataset] All Joints Mean: {np.mean(raw_world_coord[0], axis=0)}")

    # -------------------------------------------------------------

    print("\n>>> 2. Running Adapter...")
    # 构造 Batch
    batch_inputs = {
        'img': torch.tensor(inputs['img']).unsqueeze(0),
        'intrinsic': torch.tensor(inputs['intrinsic']).unsqueeze(0),
        'extrinsic': torch.tensor(inputs['extrinsic']).unsqueeze(0)
    }
    batch_targets = {
        'world_coord': torch.tensor(targets['world_coord']).unsqueeze(0)
    }

    device = torch.device('cpu')
    views, meta_list = collate_dexycb_to_mvgformer(batch_inputs, batch_targets, device)

    # 提取 Adapter 输出的世界坐标
    # 根据 adapter.py 逻辑: 'joints_3d': gt_3d[b, 0:1, ...] -> Shape (1, 21, 3)
    adapter_world_coord = meta_list[0]['joints_3d'].numpy()

    print(f"\n[Adapter] Output Shape: {adapter_world_coord.shape}")
    print(f"[Adapter] Wrist Coords: {adapter_world_coord[0, 0]}")

    # -------------------------------------------------------------

    print("\n>>> 3. Comparing...")
    # 对比 Dataset 的第 0 个视角 和 Adapter 的输出
    # raw_world_coord[0] vs adapter_world_coord[0]
    diff = np.abs(raw_world_coord[0] - adapter_world_coord[0])
    max_diff = np.max(diff)

    print(f"Max Difference: {max_diff}")

    if max_diff < 1e-5:
        print("✅ PASS: 世界坐标完全一致！")
        print("   结论：Adapter 没有修改坐标数值。")
    else:
        print("❌ FAIL: 坐标不一致！Adapter 可能切片切错了视角，或者修改了数值。")

    # -------------------------------------------------------------
    # 额外检查：数值是否合理？
    # -------------------------------------------------------------
    mean_val = np.mean(adapter_world_coord[0], axis=0)
    print(f"\n[Sanity Check] 坐标均值: {mean_val}")
    if np.abs(mean_val).max() > 100:
        print("⚠️ 警告: 坐标数值极大 (>100)！单位可能是毫米(mm)？")
        print("   如果模型预期是米(m)，这会导致 Loss 爆炸和投影飞出天际。")
    else:
        print("ℹ️ 提示: 坐标数值在合理范围 (米级)。")


if __name__ == "__main__":
    check_world_coord_consistency()