# train.py
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import sys
import warnings
import time
import datetime
import csv
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config_hand import _C as cfg
from adapter import collate_dexycb_to_mvgformer
# 引入两个数据集类
from datasets.DexYCB import DEXYCBDatasets
from datasets.DriverHOI import DriverHOIDatasets
from model import get_mvp


class ExperimentManager:
    def __init__(self, base_dir="checkpoints"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 将数据集名称加入实验文件夹，方便区分
        exp_name = f"{timestamp}_{cfg.DATASET.NAME}"
        self.exp_dir = os.path.join(base_dir, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tb_logs"))
        self.log_file = os.path.join(self.exp_dir, "output.out")
        self.logger = self._setup_logger()

        self.csv_file = os.path.join(self.exp_dir, "training_stats.csv")
        self.csv_headers = ["Epoch", "Train_Loss", "Train_L1", "Train_Proj", "Val_MPJPE", "Time_Elapsed"]
        with open(self.csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)

        self.logger.info(f"Experiment started! Artifacts saved in: {self.exp_dir}")
        self.logger.info(f"Current Dataset Config: {cfg.DATASET.NAME}")

    def _setup_logger(self):
        logger = logging.getLogger("Trainer")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            fh = logging.FileHandler(self.log_file)
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
            writer = csv.writer(f)
            writer.writerow(row_data)

    def save_checkpoint(self, model, epoch, metric, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'mpjpe': metric,
            'config': cfg,  # 保存配置以便复现
        }
        torch.save(state, os.path.join(self.exp_dir, "last_model.pth"))
        if is_best:
            torch.save(state, os.path.join(self.exp_dir, "best_model.pth"))
            self.log(f"New Best Model! MPJPE: {metric:.2f} mm")


def compute_mpjpe(pred_poses, gt_poses):
    error = torch.norm(pred_poses - gt_poses, dim=-1)
    return error.mean().item() * 1000.0


def validate(model, dataloader, device):
    model.eval()
    total_mpjpe = 0.0
    num_batches = 0

    if hasattr(model, 'matcher'):
        matcher = model.matcher
    elif hasattr(model.module, 'matcher'):
        matcher = model.module.matcher
    else:
        matcher = None

    with torch.no_grad():
        for batch_inputs, batch_targets, _ in tqdm(dataloader, desc="Validating", leave=False):
            views, meta = collate_dexycb_to_mvgformer(batch_inputs, batch_targets, device)
            outputs, _ = model(views, meta)

            pred_all = outputs['pred_poses']['outputs_coord']
            B = pred_all.shape[0]
            pred_poses_reshaped = pred_all.view(B, cfg.DECODER.num_instance, 21, 3)
            pred_logits = outputs['pred_logits']
            pred_probs = pred_logits.view(B, cfg.DECODER.num_instance, 21, 1).mean(dim=2).sigmoid()
            gt_poses = torch.stack([m['joints_3d'] for m in meta]).squeeze(1).to(device)

            batch_error = 0.0
            if matcher is not None:
                indices = matcher(pred_probs, pred_poses_reshaped, gt_poses)
                for b in range(B):
                    src_idx, tgt_idx = indices[b]
                    best_idx = src_idx.item()
                    pred_pose = pred_poses_reshaped[b, best_idx]
                    gt_pose = gt_poses[b]
                    batch_error += compute_mpjpe(pred_pose, gt_pose)
            else:
                for b in range(B):
                    gt_b = gt_poses[b]
                    preds_b = pred_poses_reshaped[b]
                    diff = preds_b - gt_b.unsqueeze(0)
                    errors = torch.norm(diff, dim=-1).mean(dim=-1)
                    min_error_m = errors.min().item()
                    batch_error += (min_error_m * 1000.0)

            total_mpjpe += (batch_error / B)
            num_batches += 1

    return total_mpjpe / num_batches if num_batches > 0 else 0.0


def main():
    warnings.filterwarnings("ignore")

    exp_manager = ExperimentManager()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_manager.log(f"Using device: {device}")

    # -----------------------------------------------------------
    # [核心修改] 根据 config 动态加载数据集
    # -----------------------------------------------------------
    dataset_name = cfg.DATASET.NAME.lower()
    split_strategy = cfg.DATASET.SPLIT_STRATEGY  # 获取策略
    exp_manager.log(f"Initializing Dataset: {dataset_name} ...")

    if dataset_name == 'dexycb':
        root_dir = cfg.DATASET.ROOT_DEXYCB
        train_dataset = DEXYCBDatasets(root_dir=root_dir, split='train', split_strategy=split_strategy)
        val_dataset = DEXYCBDatasets(root_dir=root_dir, split='test', split_strategy=split_strategy)

    elif dataset_name == 'driverhoi':
        root_dir = cfg.DATASET.ROOT_DRIVERHOI
        train_dataset = DriverHOIDatasets(root_dir=root_dir, split='train', split_strategy=split_strategy)
        val_dataset = DriverHOIDatasets(root_dir=root_dir, split='test', split_strategy=split_strategy)

    else:
        raise ValueError(f"Unknown dataset name in config: {dataset_name}")

    exp_manager.log(f"Train samples: {len(train_dataset)}")
    exp_manager.log(f"Val   samples: {len(val_dataset)}")

    # -----------------------------------------------------------

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=False)

    exp_manager.log("Building Model...")
    model = get_mvp(cfg, is_train=True).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    exp_manager.log(">>> Start Training <<<")
    best_mpjpe = float('inf')
    num_epochs = 100

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()

        total_loss = 0.0
        total_l1 = 0.0
        total_proj = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit="batch")

        for i, (batch_inputs, batch_targets, _) in enumerate(pbar):
            views, meta = collate_dexycb_to_mvgformer(batch_inputs, batch_targets, device)
            out, loss_dict = model(views, meta)

            loss_total = torch.tensor(0.0).to(device)
            l1_loss = loss_dict.get('loss_pose_perjoint', torch.tensor(0.0).to(device))
            proj_loss = loss_dict.get('loss_pose_perprojection_2d', torch.tensor(0.0).to(device))
            ce_loss = loss_dict.get('loss_ce', torch.tensor(0.0).to(device))

            if 'loss_pose_perjoint' in loss_dict:
                loss_total += l1_loss * cfg.DECODER.loss_pose_perjoint
            if 'loss_pose_perprojection_2d' in loss_dict:
                loss_total += proj_loss * cfg.DECODER.loss_pose_perprojection_2d
            if 'loss_ce' in loss_dict:
                loss_total += ce_loss * cfg.DECODER.loss_weight_loss_ce

            for k, v in loss_dict.items():
                if '_0' in k or '_1' in k:
                    base = k.rsplit('_', 1)[0]
                    if base == 'loss_pose_perjoint':
                        loss_total += v * cfg.DECODER.loss_pose_perjoint
                    elif base == 'loss_pose_perprojection_2d':
                        loss_total += v * cfg.DECODER.loss_pose_perprojection_2d

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            total_loss += loss_total.item()
            total_l1 += l1_loss.item()
            total_proj += proj_loss.item()

            pbar.set_postfix({"L1": f"{l1_loss.item():.4f}", "Proj": f"{proj_loss.item():.2f}"})

        avg_loss = total_loss / len(train_loader)
        avg_l1 = total_l1 / len(train_loader)
        avg_proj = total_proj / len(train_loader)

        val_mpjpe = validate(model, val_loader, device)
        scheduler.step()

        elapsed = time.time() - start_time
        log_str = (f"Epoch {epoch + 1} | "
                   f"Loss: {avg_loss:.4f} (L1: {avg_l1:.4f}, Proj: {avg_proj:.2f}) | "
                   f"Val MPJPE: {val_mpjpe:.2f} mm | "
                   f"Time: {elapsed:.0f}s")
        exp_manager.log(log_str)

        exp_manager.writer.add_scalar('Train/Loss', avg_loss, epoch)
        exp_manager.writer.add_scalar('Train/L1', avg_l1, epoch)
        exp_manager.writer.add_scalar('Train/Proj', avg_proj, epoch)
        exp_manager.writer.add_scalar('Val/MPJPE_mm', val_mpjpe, epoch)
        exp_manager.log_csv([epoch + 1, avg_loss, avg_l1, avg_proj, val_mpjpe, elapsed])

        is_best = val_mpjpe < best_mpjpe
        if is_best:
            best_mpjpe = val_mpjpe
        exp_manager.save_checkpoint(model, epoch + 1, val_mpjpe, is_best=is_best)

    exp_manager.log("Training Finished.")
    exp_manager.writer.close()


if __name__ == "__main__":
    main()