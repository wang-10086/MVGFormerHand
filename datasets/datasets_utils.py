import yaml
import numpy as np
import matplotlib.pyplot as plt


def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.FullLoader)

def xyz2uvd(xyz, K):
    """
    将相机坐标系下的3D点投影到像素坐标系(uv)和深度(d)。
    """
    fx, fy, fu, fv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    # 避免除以0的潜在风险
    z = xyz[:, 2]
    eps = 1e-10
    z_safe = np.where(z == 0, eps, z)

    uvd[:, 0] = (xyz[:, 0] * fx / z_safe + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / z_safe + fv)
    uvd[:, 2] = z
    return uvd


def camera2world(cam_joints_3d, extrinsic):
    """
    将相机坐标系下的三维关节点坐标转换到世界坐标系。
    注意：假设输入的 extrinsic 是从 世界坐标系 -> 相机坐标系 的变换矩阵 [R|t]。
    变换公式: X_world = R^T * (X_cam - t)
    """
    R = extrinsic[:, :3]
    t = extrinsic[:, 3]

    # 利用广播机制替代循环，提高效率
    # (N, 3) - (3,) -> (N, 3)
    # (N, 3) @ (3, 3) -> (N, 3)
    world_joints_3d = np.dot(cam_joints_3d - t, R)  # R.T @ (x-t) 等价于 (x-t) @ R

    return world_joints_3d


def get_camera_color_intrinsics(yaml_path):
    """读取相机内参"""
    with open(yaml_path, 'r') as file:
        content = yaml.load(file, Loader=yaml.FullLoader)

    if 'color' in content:
        c = content['color']
        required = ['fx', 'fy', 'ppx', 'ppy']
        if all(k in c for k in required):
            return {
                'fx': c['fx'], 'fy': c['fy'],
                'ppx': c['ppx'], 'ppy': c['ppy'],
                'extrinsics': content.get('extrinsics')
            }
    raise ValueError(f"YAML文件 {yaml_path} 中缺少必要的 color 参数")


def construct_projection_matrix(intrinsic_data, extrinsic_data):
    """
    构造投影矩阵 P, 内参矩阵 K, 和修正后的外参矩阵。

    Args:
        extrinsic_data: 原始外参数据。
        注意：DexYCB 数据集提供的外参通常是 Camera -> World，
        而标准渲染管线通常需要 World -> Camera。这里进行了相应的求逆转换。
    """
    # 1. 构造内参 K
    K = np.array([
        [intrinsic_data['fx'], 0, intrinsic_data['ppx']],
        [0, intrinsic_data['fy'], intrinsic_data['ppy']],
        [0, 0, 1]
    ])

    # 2. 处理外参
    # 原始数据 reshape
    extrinsics_matrix = np.array(extrinsic_data).reshape(3, 4)
    R_cam2world = extrinsics_matrix[:, :3]
    t_cam2world = extrinsics_matrix[:, 3].reshape(3, 1)

    # 计算 World -> Camera 的变换
    R_world2cam = np.linalg.inv(R_cam2world)
    t_world2cam = -np.dot(R_world2cam, t_cam2world)

    extrinsics_matrix_new = np.hstack((R_world2cam, t_world2cam))

    # 3. 计算投影矩阵 P = K @ [R|t]
    # 注意：你原始代码中使用了 identity_extrinsics_matrix，这意味着 P = K。
    # 如果你的目的是获取通过外参变换后的投影，应该使用 extrinsics_matrix_new。
    # 如果你的模型输入已经是在相机坐标系下的，那么 P=K 是对的。
    # 这里为了通用性，我保留了你的 "P = K" 逻辑，但请确认这是否符合你的预期。

    P = np.dot(K, extrinsics_matrix_new) # 标准做法

    # identity_extrinsics_matrix = np.eye(4)[:3, :]
    # P = np.dot(K, identity_extrinsics_matrix)  # 原始代码逻辑: 仅内参投影

    return P, K, extrinsics_matrix_new


def visualize_3d_joints(joints_3d, title='3D Hand Joints'):
    """使用 Matplotlib 可视化 3D 骨架"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    ax.scatter(xs, ys, zs, c='r', marker='o')

    # 绘制骨骼连接
    bone_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20)  # Little
    ]

    for start, end in bone_pairs:
        ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()