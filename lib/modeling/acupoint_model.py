import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from lib.modeling.heads import SMPLHead, SKELHead, AcupointHead
from lib.body_models.smpl_wrapper import SMPLWrapper as SMPL
from lib.body_models.skel_wrapper import SKELWrapper as SKEL
from lib.modeling.backbone import ViTBackbone

class AcupointModel(nn.Module):
    def __init__(
        self,
        acupoint_definitions_path: str,
        num_acupoints: int = 84,
        smpl_model_path: str = None,
        skel_model_path: str = None,
        pretrained_vit_path: str = None,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.num_acupoints = num_acupoints
        
        # Load acupoint definitions
        with open(acupoint_definitions_path, 'r') as f:
            self.acupoint_definitions = json.load(f)
        
        # Initialize SMPL and SKEL models
        self.smpl_model = SMPL(smpl_model_path).to(device)
        self.skel_model = SKEL(skel_model_path).to(device)
        
        # Initialize ViT backbone
        self.vit_backbone = ViTBackbone(
            pretrained_path=pretrained_vit_path,
            device=device
        ).to(device)
        
        # Initialize prediction heads
        self.smpl_head = SMPLHead().to(device)
        self.skel_head = SKELHead().to(device)
        self.acupoint_head = AcupointHead(num_acupoints).to(device)
        
        # Initialize loss weights
        self.loss_weights = {
            'acupoint_2d': 0.1,  # Will be dynamically adjusted
            'keypoint_2d': 1.0,
            'shape_prior': 0.1,
            'pose_prior': 0.1,
            'skeleton_consistency': 0.1
        }
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[Dict] = None
    ) -> Dict:
        """
        Forward pass of the model
        
        Args:
            images: Input images [B, 3, H, W] in range [-1, 1]
            targets: Optional ground truth targets
            
        Returns:
            Dictionary containing predictions and losses
        """
        # Ensure input is on correct device
        images = images.to(self.device)
        
        # Extract features using ViT backbone
        features = self.vit_backbone(images)
        
        # Predict SMPL parameters
        smpl_params = self.smpl_head(features)
        
        # Predict SKEL parameters
        skel_params = self.skel_head(features)
        
        # Generate SMPL mesh
        smpl_output = self.smpl_model(**smpl_params)
        vertices = smpl_output.vertices
        
        # Calculate 3D acupoint coordinates using SMPL mesh
        acupoints_3d = self._calculate_acupoints_3d(vertices)
        
        # Project to 2D
        acupoints_2d = self._project_to_2d(acupoints_3d, smpl_params['camera'])
        
        # Calculate losses if targets are provided
        losses = {}
        if targets is not None:
            # Move targets to device
            targets = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in targets.items()}
            
            losses = self._calculate_losses(
                acupoints_2d=acupoints_2d,
                targets=targets,
                smpl_params=smpl_params,
                skel_params=skel_params
            )
        
        return {
            'acupoints_3d': acupoints_3d,
            'acupoints_2d': acupoints_2d,
            'smpl_params': smpl_params,
            'skel_params': skel_params,
            'losses': losses
        }
    
    def _calculate_acupoints_3d(self, vertices: torch.Tensor) -> torch.Tensor:
        """Calculate 3D coordinates of acupoints using barycentric coordinates"""
        batch_size = vertices.shape[0]
        acupoints_3d = torch.zeros(batch_size, self.num_acupoints, 3, device=self.device)
        
        for i in range(self.num_acupoints):
            face_idx = self.acupoint_definitions[str(i)]['face_index']
            bary_coords = torch.tensor(
                self.acupoint_definitions[str(i)]['barycentric_coordinates'],
                device=self.device
            )
            
            # Get vertices of the face
            face_vertices = vertices[:, face_idx]
            
            # Calculate acupoint position using barycentric coordinates
            acupoints_3d[:, i] = torch.sum(
                face_vertices * bary_coords.view(1, 4, 1),
                dim=1
            )
        
        return acupoints_3d
    
    def _project_to_2d(
        self,
        points_3d: torch.Tensor,
        camera_params: Dict
    ) -> torch.Tensor:
        """Project 3D points to 2D using camera parameters"""
        # Implement perspective projection
        scale = camera_params['scale']
        translation = camera_params['translation']
        
        # Project points
        points_2d = points_3d[:, :, :2] * scale.view(-1, 1, 1) + translation.view(-1, 1, 2)
        return points_2d
    
    def _calculate_losses(
        self,
        acupoints_2d: torch.Tensor,
        targets: Dict,
        smpl_params: Dict,
        skel_params: Dict
    ) -> Dict:
        """Calculate all losses"""
        losses = {}
        
        # 2D acupoint loss
        losses['acupoint_2d'] = F.mse_loss(
            acupoints_2d,
            targets['acupoints']
        )
        
        # Shape prior loss
        losses['shape_prior'] = torch.mean(smpl_params['betas'] ** 2)
        
        # Pose prior loss
        losses['pose_prior'] = self._calculate_pose_prior(smpl_params['pose'])
        
        # Skeleton consistency loss
        losses['skeleton_consistency'] = self._calculate_skeleton_consistency(
            smpl_params,
            skel_params
        )
        
        # Apply loss weights
        total_loss = sum(
            self.loss_weights[k] * v
            for k, v in losses.items()
        )
        
        losses['total'] = total_loss
        return losses
    
    def _calculate_pose_prior(self, pose: torch.Tensor) -> torch.Tensor:
        """Calculate pose prior loss using joint angle limits"""
        return torch.mean(torch.clamp(pose, -np.pi, np.pi) ** 2)
    
    def _calculate_skeleton_consistency(
        self,
        smpl_params: Dict,
        skel_params: Dict
    ) -> torch.Tensor:
        """Calculate consistency loss between SMPL and SKEL models"""
        smpl_joints = self.smpl_model(**smpl_params).joints
        skel_joints = self.skel_model(**skel_params).joints
        
        return F.mse_loss(smpl_joints, skel_joints)
    
    def acpify(
        self,
        smpl_params: Dict,
        targets_2d: torch.Tensor,
        num_iterations: int = 100
    ) -> Dict:
        """
        Optimize SMPL parameters to align acupoint projections with ground truth
        
        Args:
            smpl_params: Initial SMPL parameters
            targets_2d: Ground truth 2D acupoint coordinates
            num_iterations: Number of optimization iterations
            
        Returns:
            Optimized SMPL parameters
        """
        # Convert parameters to optimizable tensors
        params = {
            k: nn.Parameter(v.clone()) for k, v in smpl_params.items()
        }
        
        # Create optimizer
        optimizer = torch.optim.Adam(params.values(), lr=0.01)
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            smpl_output = self.smpl_model(**{k: v for k, v in params.items()})
            vertices = smpl_output.vertices
            
            # Calculate acupoints and project
            acupoints_3d = self._calculate_acupoints_3d(vertices)
            acupoints_2d = self._project_to_2d(acupoints_3d, params['camera'])
            
            # Calculate losses
            loss_2d = F.mse_loss(acupoints_2d, targets_2d)
            loss_shape = torch.mean(params['betas'] ** 2)
            loss_pose = self._calculate_pose_prior(params['pose'])
            
            # Total loss
            loss = loss_2d + 0.1 * loss_shape + 0.1 * loss_pose
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        return {k: v.detach() for k, v in params.items()} 