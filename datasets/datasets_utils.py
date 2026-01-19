import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from manopth.manolayer import ManoLayer
from PIL import Image, ImageDraw


def xyz2uvd(xyz, K):
    fx, fy, fu, fv = K[0, 0], K[0, 0], K[0, 2], K[1, 2]
    uvd = np.zeros_like(xyz, np.float32)
    uvd[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
    uvd[:, 1] = (xyz[:, 1] * fy / xyz[:, 2] + fv)
    uvd[:, 2] = xyz[:, 2]
    return uvd


def project_3d_to_uvd(joints_3d, K):
    """
    将三维手部关节点投影到二维平面上，并保留深度信息

    参数:
        joints_3d: numpy数组，形状为(21, 3)，表示21个手部关节点的3D坐标 (X, Y, Z)
        K: numpy数组，形状为(3, 3)，相机内参矩阵

    返回:
        joints_uvd: numpy数组，形状为(21, 3)，表示投影后的坐标 (u, v, d)，
                   其中d是深度值（通常是原始的Z值）
    """
    # 确保输入数据形状正确
    assert joints_3d.shape == (21, 3), f"3D关节点数组形状应为(21, 3)，但得到{joints_3d.shape}"
    assert K.shape == (3, 3), f"内参矩阵形状应为(3, 3)，但得到{K.shape}"

    # 获取深度值（Z坐标）
    depth = joints_3d[:, 2].copy()

    # 处理Z为0的情况
    Z = depth.copy()
    # 将Z为0的位置设为一个很小的非零值，避免除以零
    Z[Z == 0] = 1e-10

    # 创建齐次坐标 (X/Z, Y/Z, 1)
    joints_3d_homogeneous = np.column_stack([
        joints_3d[:, 0] / Z,  # X/Z
        joints_3d[:, 1] / Z,  # Y/Z
        np.ones(joints_3d.shape[0])  # 1
    ])

    # 使用内参矩阵进行投影
    joints_2d_homogeneous = np.dot(joints_3d_homogeneous, K.T)  # (21, 3)

    # 提取u, v坐标
    u = joints_2d_homogeneous[:, 0]
    v = joints_2d_homogeneous[:, 1]

    # 组合u, v和深度值d
    joints_uvd = np.column_stack([u, v, depth])

    return joints_uvd


def camera2world(cam_joints_3d, extrinsic):
    """
    将相机坐标系下的三维关节点坐标转换到世界坐标系

    参数:
        cam_joints_3d: numpy数组，形状为(21, 3)，表示21个手部关节点在相机坐标系下的3D坐标
        extrinsic: numpy数组，形状为(3, 4)，相机外参矩阵 [R|t]，其中R是3x3旋转矩阵，t是3x1平移向量

    返回:
        world_joints_3d: numpy数组，形状为(21, 3)，表示转换到世界坐标系下的3D坐标
    """
    # 提取旋转矩阵R和平移向量t
    R = extrinsic[:, :3]  # 3x3旋转矩阵
    t = extrinsic[:, 3]  # 3x1平移向量

    # 初始化世界坐标系下的关节点坐标
    world_joints_3d = np.zeros_like(cam_joints_3d)

    # 逐个关节点进行坐标转换
    for i in range(cam_joints_3d.shape[0]):
        # 相机坐标系到世界坐标系的转换: X_world = R^(-1) * (X_camera - t)
        # 注意：这里使用R的逆矩阵，因为extrinsic是从世界坐标系到相机坐标系的变换
        # 对于正交旋转矩阵，其逆等于其转置
        world_joints_3d[i] = np.dot(R.T, cam_joints_3d[i] - t)

    return world_joints_3d
def get_camera_color_intrinsics(yaml_path):
    """
    从YAML文件中读取相机的color内参参数，包括fx, fy, ppx, ppy。

    Args:
        yaml_path (str): YAML文件的路径。

    Returns:
        dict: 包含fx, fy, ppx, ppy的字典。
    """

    # 定义处理元组的方法
    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))

    # 注册自定义的处理方法
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.FullLoader)

    # 读取YAML文件
    with open(yaml_path, 'r') as file:
        content = yaml.load(file, Loader=yaml.FullLoader)

    # 提取color下的fx, fy, ppx, ppy
    if 'color' in content:
        fx = content['color'].get('fx')
        fy = content['color'].get('fy')
        ppx = content['color'].get('ppx')
        ppy = content['color'].get('ppy')
        if fx is not None and fy is not None and ppx is not None and ppy is not None:
            return {'fx': fx, 'fy': fy, 'ppx': ppx, 'ppy': ppy, 'extrinsics': content['extrinsics']}
        else:
            raise ValueError("YAML文件中缺少color的某些参数")
    else:
        raise ValueError("YAML文件中没有找到color信息")

def construct_projection_matrix(intrinsic_data, extrinsic_data):
    """
    根据相机的内参和外参数据构造投影矩阵。

    Args:
        intrinsic_data (dict): 包含fx、fy、ppx、ppy的内参字典。
        extrinsic_data (tuple): 大小为1x12的外参元组。

    Returns:
        P: 3x4的相机投影矩阵。
        K: 3x3的相机内参矩阵。
        extrinsics_matrix: 3x4的相机外参矩阵。
    """
    # 构造内参矩阵K
    fx = intrinsic_data['fx']
    fy = intrinsic_data['fy']
    ppx = intrinsic_data['ppx']
    ppy = intrinsic_data['ppy']
    K = np.array([
        [fx, 0, ppx],
        [0, fy, ppy],
        [0, 0, 1]
    ])

    # 构造外参矩阵[R|T]
    extrinsics_matrix = np.array(extrinsic_data).reshape(3, 4)

    R = extrinsics_matrix[:, :3]  # 3x3 旋转矩阵
    t = extrinsics_matrix[:, 3].reshape(3, 1)  # 3x1 平移向量

    R_prime = np.linalg.inv(R)
    t_prime = -np.dot(R_prime, t)

    # 因为这SB数据集的外参矩阵搞错了，是从相机到世界坐标系的变换而不是一般意义上的世界坐标系到相机坐标系的变换，所以需要重新修正
    extrinsics_matrix_new = np.hstack((R_prime, t_prime))

    # 构造单位外参矩阵 [R|T]
    identity_extrinsics_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])  # 3x4 的单位矩阵

    # 计算投影矩阵P = K * [R|T]
    # P = np.dot(K, extrinsics_matrix)
    P = np.dot(K, identity_extrinsics_matrix)

    return P, K, extrinsics_matrix_new

def generate_3d_joints(pose_m, mano_model_path, side='right'):
    # 检查 pose_m 的类型和形状
    if not isinstance(pose_m, np.ndarray) or pose_m.shape != (1, 51):
        raise ValueError(
            "pose_m 应该是形状为 (1, 51) 的 numpy 数组，但收到的是: {} with shape {}".format(type(pose_m),
                                                                                            getattr(pose_m, 'shape',
                                                                                                    None)))

    # 初始化 ManoLayer
    mano_layer = ManoLayer(mano_root=mano_model_path, side=side, use_pca=True, ncomps=45, flat_hand_mean=True)

    # 提取姿态参数和平移向量
    pose_params = torch.tensor(pose_m[0, :48]).unsqueeze(0)  # 前 48 个元素为姿态参数
    shape_params = torch.zeros(1, 10)  # 默认形状参数
    t_local_to_camera = torch.tensor(pose_m[0, 48:51]).unsqueeze(0)  # 最后 3 个元素为平移向量

    # 使用 MANO 层生成手部关节点
    _, joints_camera = mano_layer(pose_params, shape_params, t_local_to_camera)

    # 将手部关节点从 tensor 转为 numpy 数组
    joints_camera = joints_camera.squeeze(0).detach().cpu().numpy()

    return joints_camera

def visualize_3d_joints(joints_3d):
    """
    可视化三维关节点坐标。

    Args:
        joints_3d (numpy.ndarray): 形状为 (21, 3) 的数组，表示 21 个关节点的三维坐标。
    """
    # 检查输入数据的形状
    if joints_3d.shape != (21, 3):
        raise ValueError("输入的关节点数组形状应为 (21, 3)")

    # 创建 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取坐标
    xs = joints_3d[:, 0]
    ys = joints_3d[:, 1]
    zs = joints_3d[:, 2]

    # 绘制散点图
    ax.scatter(xs, ys, zs, c='r', marker='o', label='Joints')

    # 在每个关节点旁标注序号
    for i, (x, y, z) in enumerate(joints_3d):
        ax.text(x, y, z, f"{i}", color='red')

    # 绘制关节点的连线（如果有骨骼结构，可以在此添加）
    # 示例骨骼连接（需要根据实际情况调整）
    bone_pairs = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 手指1
        (0, 5), (5, 6), (6, 7), (7, 8),  # 手指2
        (0, 9), (9, 10), (10, 11), (11, 12),  # 手指3
        (0, 13), (13, 14), (14, 15), (15, 16),  # 手指4
        (0, 17), (17, 18), (18, 19), (19, 20)  # 手指5
    ]

    for start, end in bone_pairs:
        ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'b')

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Hand Joints Visualization')

    # 显示图例和图形
    ax.legend()
    plt.show()

def visualize_2d_joints_on_image(image_path, joints_2d, radius=2, joint_color=(255, 0, 0), line_color=(0, 255, 0)):
    """
    在RGB图像上绘制二维关节点及其连接。

    Args:
        image_path (str): RGB图片的路径。
        joints_2d (numpy.ndarray): 二维关节点坐标，形状为 (21, 2)。
        radius (int, optional): 关节点的半径，默认为 5。
        joint_color (tuple, optional): 关节点的颜色，默认为红色 (255, 0, 0)。
        line_color (tuple, optional): 连接线的颜色，默认为绿色 (0, 255, 0)。

    Returns:
        PIL.Image: 绘制了关节点及其连接的图像。
    """
    # 打开图片
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 手指1
        (0, 5), (5, 6), (6, 7), (7, 8),  # 手指2
        (0, 9), (9, 10), (10, 11), (11, 12),  # 手指3
        (0, 13), (13, 14), (14, 15), (15, 16),  # 手指4
        (0, 17), (17, 18), (18, 19), (19, 20)  # 手指5
    ]
    # 绘制连接
    for connection in connections:
        joint1, joint2 = connection
        if 0 <= joints_2d[joint1][0] < width and 0 <= joints_2d[joint1][1] < height and \
           0 <= joints_2d[joint2][0] < width and 0 <= joints_2d[joint2][1] < height:
            # 绘制连接线
            draw.line([tuple(joints_2d[joint1]), tuple(joints_2d[joint2])], fill=line_color, width=1)

    # 绘制关节点
    for joint in joints_2d:
        x, y = joint
        if 0 <= x < width and 0 <= y < height:  # 确保关节点在图像范围内
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=joint_color)

    return image
