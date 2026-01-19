# train_hand.py
import ssl

# 全局取消证书验证，防止 urllib 报错
ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import warnings
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # --- [新增] 导入 TensorBoard ---

# 1. 导入配置和适配器
from config_hand import _C as cfg
from adapter import collate_dexycb_to_mvgformer

# 2. 导入你的数据加载器
from datasets.DexYCB import DEXYCBDatasets

# 3. 导入 MVGFormer 模型
from model import get_mvp


def main():
    # --- 忽略所有警告 ---
    warnings.filterwarnings("ignore")

    # --- 设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 路径配置
    DEXYCB_ROOT = '/home/wk/wk/wk/datasets/DexYCB'

    # --- [新增] TensorBoard 设置 ---
    # 日志将保存在 logs/tb_logs 目录下
    log_dir = os.path.join("logs", "tb_logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging to: {log_dir}")

    # --- 1. 初始化数据 ---
    print("Initializing DexYCB Dataset...")
    dataset = DEXYCBDatasets(root_dir=DEXYCB_ROOT, split='train')

    # num_workers=0 方便调试，如果要加速加载可以设为 4 或 8
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

    # --- 2. 初始化 MVGFormer 模型 ---
    print("Building MVGFormer Hand Model...")
    model = get_mvp(cfg, is_train=True).to(device)

    # --- 3. 优化器 ---
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # --- 4. 训练循环 ---
    print("Start Training...")
    model.train()

    num_epochs = 100
    for epoch in range(num_epochs):
        # --- [修改] 初始化 Epoch 累计变量 ---
        total_loss_epoch = 0.0
        total_l1_epoch = 0.0  # 累计 L1 Loss
        total_proj_epoch = 0.0  # 累计 Projection Loss

        # 使用 tqdm 包装 dataloader
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

        for i, (batch_inputs, batch_targets, _) in enumerate(pbar):
            # A. 数据适配
            views, meta = collate_dexycb_to_mvgformer(batch_inputs, batch_targets, device)

            # B. 前向传播
            out, loss_dict = model(views, meta)

            # C. Loss 计算
            loss_total = torch.tensor(0.0).to(device)

            # 提取具体的子 Loss (用于计算 total loss)
            l1_loss = loss_dict.get('loss_pose_perjoint', torch.tensor(0.0).to(device))
            proj_loss = loss_dict.get('loss_pose_perprojection_2d', torch.tensor(0.0).to(device))
            ce_loss = loss_dict.get('loss_ce', torch.tensor(0.0).to(device))

            # 加权求和
            if 'loss_pose_perjoint' in loss_dict:
                loss_total += l1_loss * cfg.DECODER.loss_pose_perjoint

            if 'loss_pose_perprojection_2d' in loss_dict:
                loss_total += proj_loss * cfg.DECODER.loss_pose_perprojection_2d

            if 'loss_ce' in loss_dict:
                loss_total += ce_loss * cfg.DECODER.loss_weight_loss_ce

            # 辅助 Loss (Auxiliary Loss)
            for k, v in loss_dict.items():
                if '_0' in k or '_1' in k or '_2' in k:
                    base_key = k.rsplit('_', 1)[0]
                    if base_key == 'loss_pose_perjoint':
                        loss_total += v * cfg.DECODER.loss_pose_perjoint
                    elif base_key == 'loss_pose_perprojection_2d':
                        loss_total += v * cfg.DECODER.loss_pose_perprojection_2d

            # D. 反向传播
            optimizer.zero_grad()
            loss_total.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

            optimizer.step()

            # --- [修改] 累计 Loss ---
            # 使用 .item() 获取 python float 数值，防止显存泄漏
            total_loss_epoch += loss_total.item()
            total_l1_epoch += l1_loss.item()  # 记录原始 L1 (未加权)
            total_proj_epoch += proj_loss.item()  # 记录原始 Proj (未加权)

            # 更新进度条
            pbar.set_postfix({
                "Total": f"{loss_total.item():.4f}",
                "L1": f"{l1_loss.item():.4f}",
                "Proj": f"{proj_loss.item():.4f}"
            })

        # --- [新增] Epoch 结束处理与 TensorBoard 写入 ---
        avg_total = total_loss_epoch / len(dataloader)
        avg_l1 = total_l1_epoch / len(dataloader)
        avg_proj = total_proj_epoch / len(dataloader)

        print(f"Epoch {epoch + 1} Finished. Avg Loss: {avg_total:.4f}")

        # 写入 TensorBoard (按 Epoch 记录)
        writer.add_scalar('Train/Total_Loss', avg_total, epoch + 1)
        writer.add_scalar('Train/L1_Loss', avg_l1, epoch + 1)
        writer.add_scalar('Train/Proj_Loss', avg_proj, epoch + 1)

        # 保存模型
        if (epoch + 1) % 5 == 0:
            save_path = f"mvgformer_hand_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    # 关闭 writer
    writer.close()


if __name__ == "__main__":
    main()