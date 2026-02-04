# train.py
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import warnings
import time
import datetime
import csv
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# -----------------------------------------------------------------------------
# 1. 导入配置与工具 (Imports)
# -----------------------------------------------------------------------------
# 使用新的统一配置
from config import cfg
from adapter import collate_dexycb_to_mvgformer

# 数据集
from datasets.DexYCB import DEXYCBDatasets
from datasets.DriverHOI import DriverHOIDatasets

# 模型导入 (从 models 文件夹)
try:
    from models.MVGFormer import get_mvgformer
    from models.LAT import get_lat_model
    from models.LVT import get_lvt_model
except ImportError as e:
    print(f"[Error] Failed to import models: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# 2. 实验管理器 (Experiment Manager)
# -----------------------------------------------------------------------------
class ExperimentManager:
    """管理实验文件夹创建、日志记录、TensorBoard 和 CSV 统计"""

    def __init__(self, base_dir="checkpoints"):
        self.base_dir = base_dir
        resume_path = cfg.TRAIN.RESUME_PATH
        self.resuming = False

        if resume_path and os.path.exists(resume_path):
            self.exp_dir = os.path.dirname(resume_path)
            self.resuming = True
            print(f"[ExperimentManager] Resuming in: {self.exp_dir}")
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{timestamp}_{cfg.MODEL.NAME}_{cfg.DATASET.NAME}"
            self.exp_dir = os.path.join(base_dir, exp_name)
            os.makedirs(self.exp_dir, exist_ok=True)
            print(f"[ExperimentManager] Created: {self.exp_dir}")

        self.log_file = os.path.join(self.exp_dir, "output.out")
        self.logger = self._setup_logger()
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tb_logs"))
        self.csv_file = os.path.join(self.exp_dir, "training_stats.csv")
        self.csv_headers = ["Epoch", "Train_Total_Loss", "Val_MPJPE", "Time_Elapsed"]

        if not self.resuming or not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as f:
                csv.writer(f).writerow(self.csv_headers)

        self.logger.info(f"Experiment Config: {cfg.MODEL.NAME} on {cfg.DATASET.NAME}")

    def _setup_logger(self):
        logger = logging.getLogger("Trainer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            mode = 'a' if self.resuming else 'w'
            fh = logging.FileHandler(self.log_file, mode=mode)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(fh)
            logger.addHandler(ch)
        return logger

    def log(self, msg):
        self.logger.info(msg)

    def log_csv(self, row_data):
        with open(self.csv_file, mode='a', newline='') as f:
            csv.writer(f).writerow(row_data)

    def save_checkpoint(self, model, optimizer, scheduler, epoch, metric, is_best=False):
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        state = {
            'epoch': epoch,
            'state_dict': model_state,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'mpjpe': metric,
            'config': cfg,  # 保存完整配置以便复现
        }
        torch.save(state, os.path.join(self.exp_dir, "last_model.pth"))
        if is_best:
            torch.save(state, os.path.join(self.exp_dir, "best_model.pth"))
            self.log(f"New Best Model Saved! MPJPE: {metric:.2f} mm")


# -----------------------------------------------------------------------------
# 3. 辅助函数 (Utils)
# -----------------------------------------------------------------------------
def build_model(cfg, device):
    """根据配置名称构建对应模型"""
    model_name = cfg.MODEL.NAME.lower()
    print(f"Building Model: {model_name} ...")

    if model_name == 'mvgformer':
        model = get_mvgformer(cfg, is_train=True)
    elif model_name == 'lat':
        model = get_lat_model(cfg)
    elif model_name == 'lvt':
        model = get_lvt_model(cfg)
    else:
        raise ValueError(f"Unknown Model Name: {model_name}")

    return model.to(device)


def compute_mpjpe(pred_poses, gt_poses):
    """计算 MPJPE (Mean Per Joint Position Error), 单位 mm"""
    error = torch.norm(pred_poses - gt_poses, dim=-1)  # (B, 21)
    return error.mean().item() * 1000.0


def validate(model, dataloader, device):
    """验证循环"""
    model.eval()
    total_mpjpe = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_inputs, batch_targets, _ in tqdm(dataloader, desc="Validating", leave=False):
            views, meta = collate_dexycb_to_mvgformer(batch_inputs, batch_targets, device)
            outputs, _ = model(views, meta)

            # 统一输出接口
            if 'final_pred_poses' in outputs:
                pred_poses = outputs['final_pred_poses']
            else:
                pred_poses = outputs['pred_poses']['outputs_coord']

            if pred_poses.dim() == 4: pred_poses = pred_poses.squeeze(1)

            gt_poses = torch.stack([m['joints_3d'] for m in meta])
            if gt_poses.dim() == 4: gt_poses = gt_poses.squeeze(1)
            gt_poses = gt_poses.to(device)

            total_mpjpe += compute_mpjpe(pred_poses, gt_poses)
            num_batches += 1

    return total_mpjpe / num_batches if num_batches > 0 else 0.0


# -----------------------------------------------------------------------------
# 4. 主程序 (Main)
# -----------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    exp_manager = ExperimentManager()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_manager.log(f"Using device: {device}")

    # 1. 数据集加载
    dataset_name = cfg.DATASET.NAME.lower()
    root_dir = cfg.DATASET.ROOT_DEXYCB if dataset_name == 'dexycb' else cfg.DATASET.ROOT_DRIVERHOI
    DatasetClass = DEXYCBDatasets if dataset_name == 'dexycb' else DriverHOIDatasets

    train_dataset = DatasetClass(root_dir=root_dir, split='train', split_strategy=cfg.DATASET.SPLIT_STRATEGY)
    val_dataset = DatasetClass(root_dir=root_dir, split='test', split_strategy=cfg.DATASET.SPLIT_STRATEGY)

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=False)

    # 2. 构建模型与优化器
    model = build_model(cfg, device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.LR_STEP_SIZE, gamma=cfg.TRAIN.LR_GAMMA)

    # 3. 恢复权重
    start_epoch = 0
    best_mpjpe = float('inf')
    if cfg.TRAIN.RESUME_PATH and os.path.exists(cfg.TRAIN.RESUME_PATH):
        ckpt = torch.load(cfg.TRAIN.RESUME_PATH, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']
        best_mpjpe = ckpt.get('mpjpe', float('inf'))
        exp_manager.log(f"Resumed from Epoch {start_epoch}. Best MPJPE: {best_mpjpe:.2f}")

    # 4. 训练循环
    exp_manager.log(">>> Start Training <<<")
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        start_time = time.time()
        model.train()
        loss_meters = defaultdict(float)
        total_loss_meter = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.TRAIN.EPOCHS}", unit="batch")

        for batch_inputs, batch_targets, _ in pbar:
            views, meta = collate_dexycb_to_mvgformer(batch_inputs, batch_targets, device)
            _, loss_dict = model(views, meta)

            loss_total = sum(loss_dict.values())

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_loss_meter += loss_total.item()

            # 更新进度条显示
            postfix_str = f"Total: {loss_total.item():.2f}"
            for k, v in loss_dict.items():
                val = v.item()
                loss_meters[k] += val
                postfix_str += f" | {k.replace('loss_', '')}: {val:.4f}"
            pbar.set_postfix_str(postfix_str)

        # Epoch 结束处理
        avg_total_loss = total_loss_meter / len(train_loader)
        avg_sub_losses = {k: v / len(train_loader) for k, v in loss_meters.items()}

        val_mpjpe = validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - start_time

        # 记录与保存
        exp_manager.log(
            f"Epoch {epoch + 1} | Time: {elapsed:.0f}s | Val MPJPE: {val_mpjpe:.2f} | Loss: {avg_total_loss:.4f}")
        exp_manager.writer.add_scalar('Train/Total_Loss', avg_total_loss, epoch)
        exp_manager.writer.add_scalar('Val/MPJPE', val_mpjpe, epoch)
        for k, v in avg_sub_losses.items():
            exp_manager.writer.add_scalar(f'Train/{k}', v, epoch)

        exp_manager.log_csv([epoch + 1, avg_total_loss, val_mpjpe, elapsed])

        is_best = val_mpjpe < best_mpjpe
        if is_best: best_mpjpe = val_mpjpe
        exp_manager.save_checkpoint(model, optimizer, scheduler, epoch + 1, val_mpjpe, is_best)

    exp_manager.log("Training Finished.")
    exp_manager.writer.close()


if __name__ == "__main__":
    main()