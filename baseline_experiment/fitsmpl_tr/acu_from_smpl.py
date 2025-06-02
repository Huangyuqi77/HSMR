import numpy as np
import json
import os
import open3d as o3d

def generate_acupoints_from_smpl(smpl_model_path, acupoint_definitions_path, output_path):
    """
    根据 SMPL 模型和 acupoint_definitions.json 文件生成穴位点坐标。

    Args:
        smpl_model_path (str): SMPL 模型文件路径 (.obj 或 .ply)。
        acupoint_definitions_path (str): 包含穴位定义和重心坐标的 JSON 文件路径。
        output_path (str): 输出生成的穴位点坐标的 JSON 文件路径。
    """
    # 加载 SMPL 模型
    smpl_mesh = o3d.io.read_triangle_mesh(smpl_model_path)
    smpl_vertices = np.asarray(smpl_mesh.vertices)
    smpl_faces = np.asarray(smpl_mesh.triangles)

    # 加载 acupoint_definitions.json 文件
    with open(acupoint_definitions_path, "r", encoding="utf-8") as f:
        acupoint_definitions = json.load(f)
    
    # 存储生成的穴位点
    acupoint_coordinates = {}

    for label, definition in acupoint_definitions.items():
        face_index = definition["face_index"]
        barycentric_coords = definition["barycentric_coordinates"]

        # 获取面片顶点
        v0_index, v1_index, v2_index = smpl_faces[face_index]
        v0 = smpl_vertices[v0_index]
        v1 = smpl_vertices[v1_index]
        v2 = smpl_vertices[v2_index]

        # 使用重心坐标计算穴位点
        acupoint_coord = (
            barycentric_coords[0] * v0 +
            barycentric_coords[1] * v1 +
            barycentric_coords[2] * v2
        )
        acupoint_coordinates[label] = acupoint_coord.tolist()

    # 保存生成的穴位点到文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(acupoint_coordinates, f, indent=2, ensure_ascii=False)

    print(f"穴位点坐标已保存到 {output_path}")


# 示例用法
smpl_model_path = "/path/to/your/smpl_model.obj"  # 替换为你的 SMPL 模型文件路径
acupoint_definitions_path = "acupoint_definitions.json"  # 替换为 acupoint_definitions 文件路径
output_path = "generated_acupoints.json"  # 输出文件路径

generate_acupoints_from_smpl(smpl_model_path, acupoint_definitions_path, output_path)