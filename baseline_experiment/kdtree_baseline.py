#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版SMPL拓扑穴位点映射框架 - 直接运行版本
作者: Huangyuqi77
日期: 2025-04-20
"""

import numpy as np
import open3d as o3d
import argparse
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class AcupointMapper:
    """在SMPL拓扑模型间映射穴位点"""
    
    def __init__(self, args):
        """
        使用命令行参数初始化
        
        Args:
            args: 解析后的命令行参数
        """
        self.threshold = args.threshold
        self.max_neighbors = args.max_neighbors
        self.visualize = args.visualize
        self.num_workers = args.num_workers
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """设置日志记录"""
        logger = logging.getLogger('AcupointMapper')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def compute_distance(self, p1, p2):
        """
        计算两个3D点之间的欧几里得距离
        
        Args:
            p1, p2 (numpy.ndarray): 3D点坐标
            
        Returns:
            float: 欧几里得距离
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def estimate_rigid_transform(self, source, target):
        """
        估计将源点映射到目标点的刚性变换(R, t)
        
        Args:
            source (numpy.ndarray): 源点集 (Nx3)
            target (numpy.ndarray): 目标点集 (Nx3)
            
        Returns:
            tuple: (R, t) 旋转矩阵和平移向量
        """
        assert len(source) == len(target)
        
        # 计算质心
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(target, axis=0)
        
        # 中心化点集
        source_centered = source - centroid_source
        target_centered = target - centroid_target
        
        # 计算交叉协方差矩阵
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
    
    def process_single_point(self, target_point, source_pcd, target_pcd, pcd_tree):
        """
        处理单个穴位点映射
        
        Args:
            target_point (numpy.ndarray): 目标穴位点坐标
            source_pcd (open3d.geometry.PointCloud): 源点云
            target_pcd (open3d.geometry.PointCloud): 目标点云
            pcd_tree (open3d.geometry.KDTreeFlann): 源点云的KD树
            
        Returns:
            tuple: (映射点, 映射方法)
        """
        # 尝试1点映射
        k, idx, _ = pcd_tree.search_knn_vector_3d(target_point, 1)
        source_point = np.asarray(source_pcd.points)[idx][0]
        dist = self.compute_distance(source_point, target_point)
        
        if dist < self.threshold:
            return np.asarray(target_pcd.points)[idx][0], "direct"
            
        # 尝试2点映射
        k, idx, _ = pcd_tree.search_knn_vector_3d(target_point, 2)
        source_points = np.asarray(source_pcd.points)[idx]
        mean_point = np.mean(source_points, axis=0)
        dist = self.compute_distance(mean_point, target_point)
        
        if dist < self.threshold:
            target_points = np.asarray(target_pcd.points)[idx]
            return np.mean(target_points, axis=0), "average_2"
            
        # 尝试3点映射
        k, idx, _ = pcd_tree.search_knn_vector_3d(target_point, 3)
        source_points = np.asarray(source_pcd.points)[idx]
        mean_point = np.mean(source_points, axis=0)
        dist = self.compute_distance(mean_point, target_point)
        
        if dist < self.threshold:
            target_points = np.asarray(target_pcd.points)[idx]
            return np.mean(target_points, axis=0), "average_3"
        
        # 使用变换映射
        k, idx, _ = pcd_tree.search_knn_vector_3d(target_point, 4)
        source_points = np.asarray(source_pcd.points)[idx]
        target_points = np.asarray(target_pcd.points)[idx]
        
        # 估计变换
        R, t = self.estimate_rigid_transform(source_points, target_points)
        transformed_point = (R @ np.array(target_point)) + t
        
        return transformed_point, "transform"
    
    def map_acupoints(self, source_pcd_path, acupoints_path, target_mesh):
        """
        将穴位点从源模型映射到目标模型
        
        Args:
            source_pcd_path (str): 源点云路径
            acupoints_path (str): 穴位点文件路径
            target_mesh (numpy.ndarray): 目标网格顶点
            
        Returns:
            numpy.ndarray: 映射后的穴位点
        """
        self.logger.info(f"正在从{source_pcd_path}加载源点云")
        source_pcd = o3d.io.read_point_cloud(source_pcd_path)
        
        # 从网格创建目标点云
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_mesh[:, :3])
        
        # 为源点云创建KD树
        self.logger.info("正在为源点云构建KD树")
        pcd_tree = o3d.geometry.KDTreeFlann(source_pcd)
        
        # 加载穴位点
        self.logger.info(f"正在从{acupoints_path}加载穴位点")
        with open(acupoints_path, 'r') as file:
            point_xyz = file.readlines()
        
        # 解析穴位点
        target_points = []
        for line in point_xyz:
            line = line.strip()
            if not line:
                continue
            dataline = line.split(' ')
            target_points.append([float(num) for num in dataline])
        
        # 使用并行执行处理穴位点
        self.logger.info(f"正在处理{len(target_points)}个穴位点")
        mapped_points = []
        method_counts = {"direct": 0, "average_2": 0, "average_3": 0, "transform": 0}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for target_point in target_points:
                future = executor.submit(
                    self.process_single_point,
                    target_point, source_pcd, target_pcd, pcd_tree
                )
                futures.append(future)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="映射点"):
                point, method = future.result()
                mapped_points.append(point)
                method_counts[method] += 1
        
        self.logger.info(f"映射统计: {method_counts}")
        
        # 如果需要，创建可视化
        if self.visualize:
            self._visualize_results(target_pcd, mapped_points)
        
        return np.array(mapped_points)
    
    def _visualize_results(self, target_pcd, mapped_points):
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
                                         window_name="穴位点映射",
                                         width=1024, height=768)

def main():
    """命令行执行的主函数"""
    parser = argparse.ArgumentParser(description='SMPL穴位点映射工具')
    parser.add_argument('--source', type=str, default='sampled_pointcloud.pcd', 
                        help='源点云文件路径（默认：当前目录下的sampled_pointcloud.pcd）')
    parser.add_argument('--acupoints', type=str, default='aligned_acupointcloud.txt',
                        help='穴位点文件路径（默认：当前目录下的aligned_acupointcloud.txt）')
    parser.add_argument('--mesh', type=str, default='target_mesh.npy',
                        help='目标网格文件路径（默认：当前目录下的target_mesh.npy）')
    parser.add_argument('--output', type=str, default='mapped_acupoints.npy',
                        help='输出文件路径（默认：当前目录下的mapped_acupoints.npy）')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果（默认：否）')
    parser.add_argument('--threshold', type=float, default=0.0001,
                        help='距离阈值（默认：0.0001）')
    parser.add_argument('--max_neighbors', type=int, default=4,
                        help='最大邻居数（默认：4）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='工作线程数（默认：4）')
    args = parser.parse_args()
    
    # 检查源文件是否存在
    for path_arg, path_name in [('source', '源点云文件'), ('acupoints', '穴位点文件'), ('mesh', '目标网格文件')]:
        path = getattr(args, path_arg)
        if not os.path.exists(path):
            print(f"警告: {path_name} '{path}' 不存在。")
            user_input = input(f"是否继续? (y/n): ")
            if user_input.lower() != 'y':
                return
    
    # 创建映射器并处理
    mapper = AcupointMapper(args)
    
    # 加载目标网格
    try:
        target_mesh = np.load(args.mesh)
    except Exception as e:
        print(f"加载目标网格文件失败: {e}")
        print("使用随机测试网格数据...")
        # 创建一个随机测试网格
        target_mesh = np.random.rand(6890, 3)  # SMPL通常有6890个顶点
    
    # 映射穴位点
    try:
        mapped_points = mapper.map_acupoints(
            args.source,
            args.acupoints,
            target_mesh
        )
        
        # 保存结果
        np.save(args.output, mapped_points)
        print(f"映射穴位点已保存到 {args.output}")
    except Exception as e:
        print(f"映射过程中出错: {e}")

if __name__ == "__main__":
    main()