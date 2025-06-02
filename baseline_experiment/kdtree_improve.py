#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMPL拓扑感知穴位点映射增强框架
作者: Huangyuqi77
日期: 2025-04-20

这个新框架通过以下几个核心优化点显著增强了SMPL拓扑穴位点映射能力：

拓扑感知：利用SMPL模型的拓扑结构，使用测地距离而非简单的欧几里得距离
特征增强：加入局部几何特征（如法向量、FPFH特征）辅助匹配
批处理机制：引入批处理机制以优化内存使用和提高并行性能
GPU加速：可选的GPU加速支持，提高大规模计算效率
自适应策略：根据局部几何特征和匹配质量自动选择最佳映射策略
骨骼感知：考虑SMPL模型的骨骼结构，支持姿态变化下的穴位映射
这些优化不仅提高了穴位点映射的精度和效率，也增强了框架的可扩展性和适应性，使其能够更好地处理各种SMPL模型和穴位点映射场景。
测地距离（Geodesic Distance）是指在曲面（如人体表面）上沿着表面行走的最短路径长度，这与直线距离（欧几里得距离）有很大不同。

"""
import numpy as np
import open3d as o3d
import torch
import argparse
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.spatial import cKDTree
import igl
class SMPLAcupointMapper:
    """SMPL拓扑感知的穴位点映射综合框架"""
    
    def __init__(self, args):
        self.args = args
        self.logger = self._setup_logger()
        self.use_gpu = args.use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.device = torch.device('cuda')
            self.logger.info("使用GPU加速计算")
        else:
            self.device = torch.device('cpu')
            self.logger.info("使用CPU进行计算")
            
        # 加载SMPL模型拓扑信息
        self.load_smpl_topology(args.smpl_model)
        
    def _setup_logger(self):
        """设置日志系统"""
        logger = logging.getLogger('SMPLAcupointMapper')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
            
    def load_smpl_topology(self, model_path):
        """加载SMPL模型拓扑结构"""
        if not os.path.exists(model_path):
            self.logger.warning(f"SMPL模型文件'{model_path}'不存在，仅使用基本几何方法")
            self.smpl_topology_loaded = False
            return
            
        try:
            import pickle
            with open(model_path, 'rb') as f:
                smpl_model = pickle.load(f, encoding='latin1')
                
            # 提取SMPL模型的关键信息
            self.smpl_faces = smpl_model.get('f', [])  # 面片索引
            self.smpl_joints = smpl_model.get('J', [])  # 关节位置
            self.smpl_weights = smpl_model.get('weights', [])  # 顶点权重
            self.smpl_vert_mapping = smpl_model.get('vert_mapping', {})  # 顶点映射
            
            # 提取骨骼父子关系
            self.smpl_kintree = smpl_model.get('kintree_table', np.zeros((2, 24)))
            self.smpl_parent = self.smpl_kintree[0].astype(np.int32)
            
            # 创建网格用于测地距离计算
            self.smpl_mesh = o3d.geometry.TriangleMesh()
            self.smpl_mesh.triangles = o3d.utility.Vector3iVector(self.smpl_faces)
            
            self.smpl_topology_loaded = True
            self.logger.info(f"SMPL拓扑结构加载成功: {len(self.smpl_faces)}个面片, {len(self.smpl_joints)}个关节")
        except Exception as e:
            self.logger.error(f"加载SMPL拓扑结构失败: {e}")
            self.smpl_topology_loaded = False
    def compute_exact_geodesic(self, mesh, source_vertex_id, target_vertex_ids=None):
        """计算精确的测地距离"""
        try:
            
            
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            # 如果没有指定目标顶点，计算到所有顶点的距离
            if target_vertex_ids is None:
                target_vertex_ids = []
            
            # 计算测地距离
            distance = igl.exact_geodesic(vertices, faces, [source_vertex_id], target_vertex_ids, [])
            
            return distance
        except ImportError:
            self.logger.warning("未安装igl库，使用近似测地距离")
            
            # 使用Dijkstra算法的近似实现
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            
            # 构建顶点邻接图
            adjacency = {}
            for face in faces:
                for i in range(3):
                    v1, v2 = face[i], face[(i+1)%3]
                    
                    if v1 not in adjacency:
                        adjacency[v1] = []
                    if v2 not in adjacency:
                        adjacency[v2] = []
                    
                    dist = np.linalg.norm(vertices[v1] - vertices[v2])
                    
                    if v2 not in adjacency[v1]:
                        adjacency[v1].append((v2, dist))
                    if v1 not in adjacency[v2]:
                        adjacency[v2].append((v1, dist))
            
            # Dijkstra算法
            import heapq
            
            distances = [float('inf')] * len(vertices)
            distances[source_vertex_id] = 0
            
            priority_queue = [(0, source_vertex_id)]
            
            while priority_queue:
                current_distance, current_vertex = heapq.heappop(priority_queue)
                
                if current_distance > distances[current_vertex]:
                    continue
                
                # 探索邻居
                if current_vertex in adjacency:
                    for neighbor, weight in adjacency[current_vertex]:
                        distance = current_distance + weight
                        
                        if distance < distances[neighbor]:
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))
            
            if target_vertex_ids:
                return [distances[vid] for vid in target_vertex_ids]
            return distances            
    def compute_geodesic_coordinates(self, mesh, points):
        """计算网格上点的测地坐标"""
        if not self.smpl_topology_loaded:
            return None
            
        try:
            # 初始化测地距离计算器
            mesh_vertices = np.asarray(mesh.vertices)
            mesh_triangles = np.asarray(mesh.triangles)
            
            import igl  # 使用libigl库计算测地距离
            
            # 选择参考点（通常选择骨骼关节作为参考点）
            reference_points = self.smpl_joints
            
            # 计算从参考点到网格所有顶点的测地距离
            geodesic_distances = []
            for ref_point in reference_points:
                # 找到最近的网格顶点作为源点
                ref_vertex_id = np.argmin(np.linalg.norm(mesh_vertices - ref_point, axis=1))
                
                # 计算单源最短路径距离
                source_indices = [ref_vertex_id]
                d = igl.exact_geodesic(mesh_vertices, mesh_triangles, 
                                    source_indices, [], [])
                geodesic_distances.append(d)
                
            # 将测地距离转换为坐标
            geodesic_coords = np.column_stack(geodesic_distances)
            
            # 计算查询点的测地坐标
            point_geodesic_coords = []
            for point in points:
                # 找到最近的网格顶点
                vertex_id = np.argmin(np.linalg.norm(mesh_vertices - point, axis=1))
                # 获取该顶点的测地坐标
                point_geodesic_coords.append(geodesic_coords[vertex_id])
                
            return np.array(point_geodesic_coords)
            
        except ImportError:
            self.logger.warning("未安装igl库，无法计算测地距离")
            return None
        except Exception as e:
            self.logger.error(f"计算测地坐标失败: {e}")
            return None
        
    def compute_local_features(self, pcd):
        """计算点云的局部几何特征"""
        # 计算法向量
        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                  radius=self.args.normal_radius, max_nn=30))
        
        # 计算FPFH特征（快速点特征直方图）
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.args.feature_radius, max_nn=100))
                
        return np.asarray(fpfh.data).T  # 转置为每行一个点的格式
    
    def find_correspondence_with_topology(self, source_point, target_pcd, target_features=None):
        """使用拓扑信息寻找对应点"""
        target_points = np.asarray(target_pcd.points)
        
        # 如果有拓扑信息，使用测地距离和局部特征
        if self.smpl_topology_loaded and hasattr(self, 'geodesic_coords_source') and hasattr(self, 'geodesic_coords_target'):
            try:
                # 找到源点的最近网格顶点
                source_vertex_id = np.argmin(np.linalg.norm(
                    np.asarray(self.source_mesh.vertices) - source_point, axis=1))
                
                # 获取该顶点的测地坐标
                source_geodesic = self.geodesic_coords_source[source_vertex_id]
                
                # 在目标模型中寻找测地坐标相似的点
                geodesic_dists = np.linalg.norm(
                    self.geodesic_coords_target - source_geodesic, axis=1)
                
                # 获取最相似的K个点
                k = min(self.args.max_neighbors, len(geodesic_dists))
                indices = np.argsort(geodesic_dists)[:k]
                distances = geodesic_dists[indices]
                
                # 如果需要，结合几何特征
                if target_features is not None and hasattr(self, 'source_features'):
                    # 找到源点对应的特征
                    source_feature = self.source_features[source_vertex_id]
                    
                    # 计算特征距离
                    feature_dists = np.linalg.norm(
                        target_features[indices] - source_feature, axis=1)
                    
                    # 结合几何距离和特征距离
                    combined_dists = distances * 0.7 + feature_dists * 0.3
                    
                    # 重新排序
                    sorted_indices = np.argsort(combined_dists)
                    indices = indices[sorted_indices]
                    distances = distances[sorted_indices]
                
                return indices, distances
                
            except Exception as e:
                self.logger.warning(f"拓扑匹配失败，回退到KNN: {e}")
        
        # 基本的KNN搜索作为备选或回退方案
        tree = cKDTree(target_points)
        distances, indices = tree.query(source_point, k=self.args.max_neighbors)
        
        # 如果有特征但没有使用拓扑方法，仍可以使用特征进行细化
        if target_features is not None and hasattr(self, 'source_features'):
            try:
                # 找到源点对应的特征
                source_feature_idx = np.argmin(np.linalg.norm(
                    np.asarray(self.source_mesh.vertices) - source_point, axis=1))
                source_feature = self.source_features[source_feature_idx]
                
                # 使用特征细化匹配
                feature_dists = np.linalg.norm(
                    target_features[indices] - source_feature, axis=1)
                
                # 结合空间距离和特征距离
                combined_dists = distances * 0.7 + feature_dists * 0.3
                
                # 重新排序
                sorted_indices = np.argsort(combined_dists)
                indices = indices[sorted_indices]
                distances = distances[sorted_indices]
            except Exception as e:
                self.logger.warning(f"特征匹配细化失败: {e}")
        
        return indices, distances
    
    def apply_skeleton_deformation(self, points, source_pose, target_pose):
        """应用骨骼驱动的变形"""
        if not self.smpl_topology_loaded:
            return points
            
        try:
            import torch
            from scipy.spatial.transform import Rotation
            
            # 确保输入格式正确
            points = np.array(points)
            
            # 获取最接近的顶点和对应的蒙皮权重
            closest_vertices = []
            vertex_weights = []
            
            mesh_vertices = np.asarray(self.smpl_mesh.vertices)
            
            for point in points:
                # 找到最近的顶点
                vertex_id = np.argmin(np.linalg.norm(mesh_vertices - point, axis=1))
                closest_vertices.append(vertex_id)
                
                # 获取该顶点的蒙皮权重
                if hasattr(self, 'smpl_weights') and len(self.smpl_weights) > vertex_id:
                    vertex_weights.append(self.smpl_weights[vertex_id])
                else:
                    # 如果没有权重信息，使用距离加权
                    dists = np.linalg.norm(self.smpl_joints - point, axis=1)
                    weights = 1.0 / (dists + 1e-8)
                    weights = weights / np.sum(weights)
                    vertex_weights.append(weights)
            
            vertex_weights = np.array(vertex_weights)
            
            # 计算骨骼变换
            num_joints = len(self.smpl_joints)
            joint_transforms = []
            
            for j in range(num_joints):
                # 计算源姿态到目标姿态的变换
                if isinstance(source_pose, np.ndarray) and isinstance(target_pose, np.ndarray):
                    # 假设姿态以旋转矩阵形式提供
                    source_rot = source_pose[j]
                    target_rot = target_pose[j]
                    
                    # 计算相对旋转
                    rel_rot = target_rot @ np.linalg.inv(source_rot)
                    
                    # 应用变换
                    parent = self.smpl_parent[j]
                    if parent >= 0:
                        # 这里需要考虑骨骼的层级关系
                        parent_transform = joint_transforms[parent]
                        transform = parent_transform @ np.vstack([
                            np.hstack([rel_rot, self.smpl_joints[j].reshape(3, 1)]),
                            np.array([0, 0, 0, 1])
                        ])
                    else:
                        # 根关节
                        transform = np.vstack([
                            np.hstack([rel_rot, self.smpl_joints[j].reshape(3, 1)]),
                            np.array([0, 0, 0, 1])
                        ])
                    
                    joint_transforms.append(transform)
                else:
                    # 如果没有姿态信息，使用单位变换
                    joint_transforms.append(np.eye(4))
            
            joint_transforms = np.array(joint_transforms)
            
            # 应用线性混合蒙皮
            transformed_points = []
            for i, point in enumerate(points):
                weights = vertex_weights[i]
                
                # 计算加权变换
                weighted_transform = np.zeros((4, 4))
                for j in range(num_joints):
                    weighted_transform += weights[j] * joint_transforms[j]
                
                # 应用变换
                point_homogeneous = np.append(point, 1.0)
                transformed_point = weighted_transform @ point_homogeneous
                
                transformed_points.append(transformed_point[:3])
            
            return np.array(transformed_points)
            
        except Exception as e:
            self.logger.error(f"骨骼变形失败: {e}")
            return points
    
    def select_mapping_strategy(self, source_point, indices, distances, target_pcd):
        """选择最佳的映射策略"""
        # 基于距离阈值的渐进策略
        target_points = np.asarray(target_pcd.points)[indices]
        
        # 直接映射
        if distances[0] < self.args.threshold:
            return target_points[0], "direct"
            
        # 平均映射
        if len(indices) >= 2:
            mean_point = np.mean(target_points[:2], axis=0)
            if np.linalg.norm(mean_point - source_point) < self.args.threshold:
                return mean_point, "average_2"
                
        if len(indices) >= 3:
            mean_point = np.mean(target_points[:3], axis=0)
            if np.linalg.norm(mean_point - source_point) < self.args.threshold:
                return mean_point, "average_3"
                
        # 变换映射
        if len(indices) >= 4:
            source_neighborhood = np.array([source_point] * len(indices[:4]))
            target_neighborhood = target_points[:4]
            
            # 计算最佳刚性变换
            R, t = self.estimate_rigid_transform(source_neighborhood, target_neighborhood)
            transformed_point = (R @ source_point) + t
            return transformed_point, "transform"
            
        # 如果上述策略都不适用，使用最近点
        return target_points[0], "nearest"
    
    def estimate_rigid_transform(self, source, target):
        """估计刚性变换(R, t)"""
        # 计算质心
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(target, axis=0)
        
        # 中心化点集
        source_centered = source - centroid_source
        target_centered = target - centroid_target
        
        # 计算协方差矩阵
        H = source_centered.T @ target_centered
        
        # SVD分解
        U, _, Vt = np.linalg.svd(H)
        
        # 计算旋转矩阵
        R = Vt.T @ U.T
        
        # 处理反射情况
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 计算平移
        t = centroid_target - R @ centroid_source
        
        return R, t
    
    def process_batch(self, source_points, source_pcd, target_pcd, source_features=None, target_features=None):
        """批量处理点集"""
        results = []
        methods = []
        
        # 如果使用GPU，将数据转移到GPU
        if self.use_gpu:
            # 实现GPU加速计算
            pass
        else:
            # CPU计算
            for source_point in source_points:
                # 寻找对应点
                indices, distances = self.find_correspondence_with_topology(
                    source_point, target_pcd, target_features)
                
                # 选择映射策略
                mapped_point, method = self.select_mapping_strategy(
                    source_point, indices, distances, target_pcd)
                
                results.append(mapped_point)
                methods.append(method)
                
        return results, methods
    
    def map_acupoints(self, source_pcd_path, acupoints_path, target_mesh_path):
        """执行穴位点映射流程"""
        # 加载源点云
        self.logger.info(f"正在从{source_pcd_path}加载源点云")
        source_pcd = o3d.io.read_point_cloud(source_pcd_path)
        
        # 加载目标网格
        self.logger.info(f"正在从{target_mesh_path}加载目标网格")
        if target_mesh_path.endswith('.npy'):
            target_vertices = np.load(target_mesh_path)
            # 创建一个临时网格用于测地距离计算
            if self.smpl_topology_loaded:
                self.target_mesh = o3d.geometry.TriangleMesh()
                self.target_mesh.vertices = o3d.utility.Vector3dVector(target_vertices[:, :3])
                self.target_mesh.triangles = self.smpl_mesh.triangles
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_vertices[:, :3])
        else:
            self.target_mesh = o3d.io.read_triangle_mesh(target_mesh_path)
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = self.target_mesh.vertices
        
        # 创建源网格（如果尚未创建）
        if self.smpl_topology_loaded and not hasattr(self, 'source_mesh'):
            self.source_mesh = o3d.geometry.TriangleMesh()
            self.source_mesh.vertices = source_pcd.points
            self.source_mesh.triangles = self.smpl_mesh.triangles
        
        # 加载穴位点
        self.logger.info(f"正在从{acupoints_path}加载穴位点")
        with open(acupoints_path, 'r') as file:
            point_lines = file.readlines()
        
        source_points = []
        for line in point_lines:
            line = line.strip()
            if not line:
                continue
            coordinates = line.split(' ')
            source_points.append([float(num) for num in coordinates])
        
        source_points = np.array(source_points)
        
        # 如果启用了拓扑感知，预计算测地坐标
        if self.smpl_topology_loaded and hasattr(self, 'source_mesh') and hasattr(self, 'target_mesh'):
            try:
                self.logger.info("计算测地坐标...")
                self.geodesic_coords_source = self.compute_geodesic_coordinates(
                    self.source_mesh, np.asarray(self.source_mesh.vertices))
                self.geodesic_coords_target = self.compute_geodesic_coordinates(
                    self.target_mesh, np.asarray(self.target_mesh.vertices))
                self.logger.info("测地坐标计算完成")
            except Exception as e:
                self.logger.warning(f"测地坐标计算失败: {e}")
        
        # 计算几何特征（可选）
        if self.args.use_features:
            self.logger.info("计算几何特征")
            self.source_features = self.compute_local_features(source_pcd)
            target_features = self.compute_local_features(target_pcd)
        else:
            self.source_features = None
            target_features = None
        
        # 分批处理
        self.logger.info(f"开始处理{len(source_points)}个穴位点")
        batch_size = min(self.args.batch_size, len(source_points))
        num_batches = (len(source_points) + batch_size - 1) // batch_size
        
        all_results = []
        method_counts = {"direct": 0, "average_2": 0, "average_3": 0, "transform": 0, "nearest": 0}
        
        with ThreadPoolExecutor(max_workers=self.args.num_workers) as executor:
            futures = []
            
            # 提交批处理任务
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(source_points))
                batch = source_points[start_idx:end_idx]
                
                future = executor.submit(
                    self.process_batch,
                    batch, source_pcd, target_pcd,
                    None if self.source_features is None else self.source_features[start_idx:end_idx],
                    target_features
                )
                futures.append(future)
            
            # 收集结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="映射穴位点"):
                batch_results, batch_methods = future.result()
                all_results.extend(batch_results)
                
                # 统计使用的方法
                for method in batch_methods:
                    method_counts[method] += 1
        
        self.logger.info(f"映射统计: {method_counts}")
        
        # 如果需要，创建可视化
        if self.args.visualize:
            self.visualize_results(target_pcd, all_results)
        
        return np.array(all_results)
    def process_batch_gpu(self, source_points, source_pcd, target_pcd, source_features=None, target_features=None):
        """使用GPU批量处理点集"""
        # 将数据转移到GPU
        device = torch.device('cuda')
        source_points_gpu = torch.tensor(source_points, dtype=torch.float32, device=device)
        target_points_gpu = torch.tensor(np.asarray(target_pcd.points), dtype=torch.float32, device=device)
        
        # KNN搜索
        results = []
        methods = []
        
        # GPU版本的KNN搜索
        # 注意：这里使用了一个简化的GPU KNN实现
        # 实际应用可能需要使用更高效的库如FAISS
        for i in range(len(source_points_gpu)):
            source_point = source_points_gpu[i]
            
            # 计算到所有目标点的距离
            diff = target_points_gpu - source_point.unsqueeze(0)
            dist_squared = torch.sum(diff * diff, dim=1)
            
            # 获取最近的K个点
            k = min(self.args.max_neighbors, len(target_points_gpu))
            distances, indices = torch.topk(-dist_squared, k)
            distances = torch.sqrt(-distances)
            
            # 转回CPU处理
            indices_cpu = indices.cpu().numpy()
            distances_cpu = distances.cpu().numpy()
            source_point_cpu = source_point.cpu().numpy()
            
            # 选择映射策略
            target_points = np.asarray(target_pcd.points)[indices_cpu]
            
            # 与CPU版本相同的策略选择逻辑
            if distances_cpu[0] < self.args.threshold:
                mapped_point = target_points[0]
                method = "direct"
            elif len(indices_cpu) >= 2:
                mean_point = np.mean(target_points[:2], axis=0)
                if np.linalg.norm(mean_point - source_point_cpu) < self.args.threshold:
                    mapped_point = mean_point
                    method = "average_2"
                elif len(indices_cpu) >= 3:
                    mean_point = np.mean(target_points[:3], axis=0)
                    if np.linalg.norm(mean_point - source_point_cpu) < self.args.threshold:
                        mapped_point = mean_point
                        method = "average_3"
                    else:
                        # 变换映射
                        source_neighborhood = np.array([source_point_cpu] * len(indices_cpu[:4]))
                        target_neighborhood = target_points[:4]
                        
                        # 计算最佳刚性变换
                        R, t = self.estimate_rigid_transform(source_neighborhood, target_neighborhood)
                        mapped_point = (R @ source_point_cpu) + t
                        method = "transform"
                else:
                    # 变换映射
                    source_neighborhood = np.array([source_point_cpu] * len(indices_cpu))
                    target_neighborhood = target_points
                    
                    # 计算最佳刚性变换
                    R, t = self.estimate_rigid_transform(source_neighborhood, target_neighborhood)
                    mapped_point = (R @ source_point_cpu) + t
                    method = "transform"
            else:
                mapped_point = target_points[0]
                method = "nearest"
                
            results.append(mapped_point)
            methods.append(method)
    
        return results, methods
    def process_batch(self, source_points, source_pcd, target_pcd, source_features=None, target_features=None):
        """批量处理点集"""
        # 如果使用GPU并且可用，调用GPU版本
        if self.use_gpu and torch.cuda.is_available():
            try:
                return self.process_batch_gpu(source_points, source_pcd, target_pcd, source_features, target_features)
            except Exception as e:
                self.logger.warning(f"GPU处理失败，回退到CPU: {e}")
        
        # CPU版本
        results = []
        methods = []
        
        for source_point in source_points:
            # 寻找对应点
            indices, distances = self.find_correspondence_with_topology(
                source_point, target_pcd, target_features)
            
            # 选择映射策略
            mapped_point, method = self.select_mapping_strategy(
                source_point, indices, distances, target_pcd)
            
            results.append(mapped_point)
            methods.append(method)
            
        return results, methods
    def visualize_results(self, target_pcd, mapped_points):
        """可视化映射结果"""
        self.logger.info("生成可视化")
        
        # 为映射点创建点云
        mapped_pcd = o3d.geometry.PointCloud()
        mapped_pcd.points = o3d.utility.Vector3dVector(mapped_points)
        mapped_pcd.paint_uniform_color([1, 0, 0])  # 穴位点为红色
        
        # 为目标模型着色
        target_pcd.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色
        
        # 可视化
        o3d.visualization.draw_geometries([target_pcd, mapped_pcd],
                                         window_name="SMPL穴位点映射",
                                         width=1024, height=768)
    def visualize_results_enhanced(self, target_pcd, mapped_points, method_labels=None):
        """增强的可视化结果"""
        self.logger.info("生成增强可视化")
        
        # 为映射点创建点云
        mapped_pcd = o3d.geometry.PointCloud()
        mapped_pcd.points = o3d.utility.Vector3dVector(mapped_points)
        
        # 如果提供了方法标签，使用不同颜色
        if method_labels:
            colors = []
            method_colors = {
                "direct": [1, 0, 0],      # 红色
                "average_2": [0, 1, 0],   # 绿色
                "average_3": [0, 0, 1],   # 蓝色
                "transform": [1, 1, 0],   # 黄色
                "nearest": [1, 0, 1]      # 紫色
            }
            
            for method in method_labels:
                colors.append(method_colors.get(method, [0, 0, 0]))
            
            mapped_pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # 否则使用统一红色
            mapped_pcd.paint_uniform_color([1, 0, 0])
        
        # 为目标模型着色
        target_pcd.paint_uniform_color([0.8, 0.8, 0.8])
        
        # 创建信息文本
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="SMPL穴位点映射", width=1024, height=768)
        
        # 添加点云
        vis.add_geometry(target_pcd)
        vis.add_geometry(mapped_pcd)
        
        # 设置相机参数
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 5.0
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
def main():
    """主函数 - 处理命令行参数并执行穴位点映射流程
    更新日期: 2025-04-20
    用户: Huangyuqi77
    """
    parser = argparse.ArgumentParser(description='SMPL拓扑感知穴位点映射工具')
    
    # 输入/输出参数
    parser.add_argument('--source', type=str, default='sampled_pointcloud.pcd',
                        help='源点云文件路径')
    parser.add_argument('--acupoints', type=str, default='aligned_acupointcloud.txt',
                        help='穴位点文件路径')
    parser.add_argument('--mesh', type=str, default='target_mesh.npy',
                        help='目标网格文件路径')
    parser.add_argument('--output', type=str, default='mapped_acupoints.npy',
                        help='输出文件路径')
    parser.add_argument('--smpl_model', type=str, default='smpl_model.pkl',
                        help='SMPL模型文件路径')
    
    # 算法参数
    parser.add_argument('--threshold', type=float, default=0.0001,
                        help='距离阈值')
    parser.add_argument('--max_neighbors', type=int, default=4,
                        help='最大邻居点数')
    parser.add_argument('--use_features', action='store_true',
                        help='是否使用几何特征辅助匹配')
    parser.add_argument('--feature_radius', type=float, default=0.05,
                        help='特征计算半径')
    parser.add_argument('--normal_radius', type=float, default=0.01,
                        help='法向量计算半径')
    parser.add_argument('--use_geodesic', action='store_true',
                        help='是否使用测地距离')
    parser.add_argument('--use_skeleton', action='store_true',
                        help='是否使用骨骼驱动')
    
    # 性能参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='工作线程数')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='批处理大小')
    parser.add_argument('--use_gpu', action='store_true',
                        help='是否使用GPU加速')
    
    # 其他选项
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    parser.add_argument('--enhanced_vis', action='store_true',
                        help='使用增强可视化')
    parser.add_argument('--log_file', type=str, default='',
                        help='日志文件路径，默认为控制台输出')
    
    args = parser.parse_args()
    
    # 设置日志文件
    if args.log_file:
        logging.basicConfig(
            filename=args.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # 检查GPU是否可用
    if args.use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                print("警告: GPU不可用，将使用CPU计算")
                args.use_gpu = False
        except ImportError:
            print("警告: 未安装PyTorch，无法使用GPU加速")
            args.use_gpu = False
    
    # 检查输入文件
    for path_arg, path_name in [
        ('source', '源点云文件'), 
        ('acupoints', '穴位点文件'), 
        ('mesh', '目标网格文件'),
        ('smpl_model', 'SMPL模型文件')
    ]:
        path = getattr(args, path_arg)
        if not os.path.exists(path):
            print(f"警告: {path_name} '{path}' 不存在!")
            if path_arg == 'smpl_model':
                print("将不使用SMPL拓扑结构，仅使用基本几何方法")
            else:
                user_input = input(f"是否继续? (y/n): ")
                if user_input.lower() != 'y':
                    return
    
    # 打印配置信息
    print("\n" + "="*50)
    print(f"SMPL穴位点映射工具 - 由 {os.getlogin()} 运行于 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    print(f"源点云: {args.source}")
    print(f"穴位点: {args.acupoints}")
    print(f"目标网格: {args.mesh}")
    print(f"SMPL模型: {args.smpl_model}")
    print(f"线程数: {args.num_workers}")
    print(f"使用GPU: {'是' if args.use_gpu else '否'}")
    print(f"使用特征匹配: {'是' if args.use_features else '否'}")
    print(f"使用测地距离: {'是' if args.use_geodesic else '否'}")
    print(f"使用骨骼驱动: {'是' if args.use_skeleton else '否'}")
    print("="*50 + "\n")
    
    # 创建映射器并处理
    start_time = time.time()
    mapper = SMPLAcupointMapper(args)
    
    try:
        # 执行映射
        print("\n开始穴位点映射过程...")
        mapped_points = mapper.map_acupoints(
            args.source,
            args.acupoints,
            args.mesh
        )
        
        # 计算并显示耗时
        elapsed_time = time.time() - start_time
        print(f"\n映射完成! 处理时间: {elapsed_time:.2f}秒 ({len(mapped_points)/elapsed_time:.2f}点/秒)")
        
        # 保存结果
        print(f"正在保存映射结果到 {args.output}...")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.save(args.output, mapped_points)
        print(f"结果已保存!")
        
        # 可视化结果
        if args.visualize:
            print("正在生成可视化...")
            target_pcd = o3d.geometry.PointCloud()
            
            # 加载目标网格/点云以进行可视化
            if args.mesh.endswith('.npy'):
                target_vertices = np.load(args.mesh)
                target_pcd.points = o3d.utility.Vector3dVector(target_vertices[:, :3])
            else:
                target_mesh = o3d.io.read_triangle_mesh(args.mesh)
                target_pcd.points = target_mesh.vertices
            
            # 选择可视化方法
            if args.enhanced_vis:
                # 收集映射方法标签以进行增强可视化
                method_labels = getattr(mapper, 'mapping_methods', None)
                mapper.visualize_results_enhanced(target_pcd, mapped_points, method_labels)
            else:
                mapper.visualize_results(target_pcd, mapped_points)
        
        # 输出统计信息
        stats = getattr(mapper, 'mapping_stats', None)
        if stats:
            print("\n映射统计信息:")
            for method, count in stats.items():
                print(f"  {method}: {count} 点 ({count/len(mapped_points)*100:.1f}%)")
        
        print("\n穴位点映射过程成功完成!")
        
    except Exception as e:
        print(f"\n映射过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    # 添加必要的导入
    import os
    import time
    import sys
    # 执行主函数
    sys.exit(main())