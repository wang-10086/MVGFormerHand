import torch

ckpt = torch.load('/root/autodl-tmp/MVGFormerHandCheckpoints/20260225_203338_self_sup_mvgformer_dexycb/best_model.pth', map_location='cuda')

print(f"Epoch: {ckpt['epoch']}")
print(f"MPJPE: {ckpt['mpjpe']:.2f} mm")

# 学习率
for i, pg in enumerate(ckpt['optimizer']['param_groups']):
    print(f"Param group {i}: lr={pg['lr']}, weight_decay={pg['weight_decay']}")

# Scheduler 状态
print(f"Scheduler: {ckpt['scheduler']}")

# 保存时的完整配置
print(f"Config:\n{ckpt['config']}")