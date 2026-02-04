#!/bin/bash
#SBATCH --job-name=mvg_test       # 作业名称
#SBATCH --partition=GPU           # 分区名称
#SBATCH --gres=gpu:rtx5090:1   # 申请资源
#SBATCH --cpus-per-task=8         # 申请CPU核数
#SBATCH --mem=32G                 # 申请内存
#SBATCH --output=logs/test_%j.out # 标准输出日志 (改为 logs/test_%j.out)
#SBATCH --time=04:00:00           # 申请时间

# --- 调试信息 ---
echo "作业运行节点: $(hostname)"
echo "开始时间: $(date)"
echo "使用的Python: $(which python)"

# --- 执行测试 ---
# -u 参数让 python 实时输出日志
python -u test.py --checkpoint "$CHECKPOINT_PATH"

echo "结束时间: $(date)"