#!/bin/bash
#SBATCH --job-name=mvg_hand       # 作业名称
#SBATCH --partition=GPU           # 分区名称
#SBATCH --gres=gpu:rtx6000ada:1      # 申请资源
#SBATCH --cpus-per-task=8         # 申请CPU核数: 建议8核以配合DataLoader的多线程
#SBATCH --mem=32G                 # 申请内存
#SBATCH --output=logs/train_%j.out # 标准输出日志 (需先创建logs文件夹: mkdir logs)
#SBATCH --time=48:00:00

# --- 调试信息 (参考手册示例) ---
echo "作业运行节点: $(hostname)"
echo "开始时间: $(date)"
echo "使用的Python: $(which python)"

# --- 执行训练 ---
# -u 参数让 python 实时输出日志，不要缓存，方便查看 logs
python -u train.py

echo "结束时间: $(date)"