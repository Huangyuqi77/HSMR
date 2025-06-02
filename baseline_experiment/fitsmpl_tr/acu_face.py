import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import json
import os

def create_acupoint_definitions(pcd_path, acupoint_json_path):
    """
    创建 acupoint_definitions 字典，通过寻找点云 Delaunay 三角剖分中距离每个穴位最近的面片，
    并计算重心坐标。

    Args:
        pcd_path (str): 点云文件路径 (.pcd)。
        acupoint_json_path (str): 包含穴位名称和 3D 坐标的 JSON 文件路径。

    Returns:
        dict: 一个字典，将穴位名称映射到其所在面片索引和重心坐标。
              例如: {"大椎": {"face_index": 1234, "barycentric_coordinates": [0.2, 0.3, 0.5]}, ...}
    """

    # 加载点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    # 构建 Delaunay 三角剖分
    try:
        tri = Delaunay(points)
    except Exception as e:
        print(f"Error creating Delaunay triangulation: {e}")
        return {}

    # 创建 KD 树，用于快速查找最近点
    kdtree = cKDTree(points)

    # 从 JSON 文件加载穴位坐标
    with open(acupoint_json_path, 'r') as f:
        acupoint_coordinates = json.load(f)

    acupoint_definitions = {}
    for acupoint_name, acupoint_coord in acupoint_coordinates.items():
        # 寻找包含穴位的四面体（如果存在）
        simplex = tri.find_simplex(acupoint_coord)

        if simplex == -1:
            # 穴位不在任何四面体内，寻找最近的顶点和包含该顶点的四面体
            distance, point_index = kdtree.query(acupoint_coord)

            # 检查距离阈值
            if distance > 0.1:
                print(f"Warning: {acupoint_name} 距离点云太远，跳过")
                continue

            # 找到包含最近顶点的所有四面体
            containing_simplices = []
            for i in range(len(tri.simplices)):
                if point_index in tri.simplices[i]:
                    containing_simplices.append(i)

            if not containing_simplices:
                print(f"Warning: 无法找到包含 {acupoint_name} 最近顶点的四面体，跳过")
                continue

            # 使用第一个包含最近顶点的四面体
            simplex = containing_simplices[0]  # 使用第一个找到的四面体
            print(f"Warning: {acupoint_name} 不在任何四面体内，使用最近顶点所在的四面体 {simplex}")

        # 获取四面体的顶点索引
        v0_index, v1_index, v2_index, v3_index = tri.simplices[simplex]

        # Calculate barycentric coordinates using the points directly from the array
        try:
            v0 = points[v0_index]
            v1 = points[v1_index]
            v2 = points[v2_index]
            v3 = points[v3_index]
            barycentric_coords = calculate_barycentric_coordinates_tetrahedron(acupoint_coord, np.array([v0, v1, v2, v3]))

            if np.any(barycentric_coords < 0) or np.sum(barycentric_coords) == 0:
                print(f"Warning: Calculated barycentric coordinates are invalid for {acupoint_name}, skipping")
                continue

            acupoint_definitions[acupoint_name] = {
                "face_index": simplex,
                "barycentric_coordinates": barycentric_coords.tolist()
            }
        except np.linalg.LinAlgError as e:
            print(f"Warning: Error calculating barycentric coordinates for {acupoint_name}: {e}, skipping")
            continue
        except Exception as e:
            print(f"Warning: Error calculating barycentric coordinates for {acupoint_name}: {e}, skipping")

    return acupoint_definitions


def calculate_barycentric_coordinates_tetrahedron(point, tetrahedron_vertices):
    """
    Calculates the barycentric coordinates of a point inside a tetrahedron.

    Args:
        point (numpy.ndarray): The 3D coordinates of the point.
        tetrahedron_vertices (numpy.ndarray): A numpy array of shape (4, 3) containing the 3D coordinates of the tetrahedron vertices.

    Returns:
        numpy.ndarray: A numpy array of shape (4,) containing the barycentric coordinates.
    """
    v0, v1, v2, v3 = tetrahedron_vertices

    try:
        # Solve the linear system
        A = np.array([v1 - v0, v2 - v0, v3 - v0]).T
        b = point - v0
        x = np.linalg.solve(A, b)
        barycentric_coords = np.concatenate(([1 - np.sum(x)], x))
        return barycentric_coords
    except np.linalg.LinAlgError as e:
        # Matrix is singular or some other linear algebra error occurred
        print(f"Linear algebra error: {e}")
        # Return default barycentric coordinates
        return np.array([0.25, 0.25, 0.25, 0.25])
    except Exception as e:
        print(f"Other error: {e}")
        return np.array([0, 0, 0, 0])

# 示例用法:
pcd_path = "/home/kemove/test/acupoints_process/model_data/tongren/male_aligned.pcd"
acupoint_json_path = "/home/kemove/dataset/git/HSMR/baseline_experiment/data/tongren/tongren_acu84.json"

acupoint_definitions = create_acupoint_definitions(pcd_path, acupoint_json_path)
print(acupoint_definitions)

# 将 acupoint_definitions 保存到当前目录中的 JSON 文件
output_file = "acupoint_definitions.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(acupoint_definitions, f, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)

print(f"Acupoint definitions saved to {output_file}")