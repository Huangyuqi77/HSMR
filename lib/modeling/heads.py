import torch
import torch.nn as nn
import torch.nn.functional as F

class SMPLHead(nn.Module):
    def __init__(self, num_betas=10, num_pose=72):
        super().__init__()
        self.num_betas = num_betas
        self.num_pose = num_pose
        
        # Shape parameters (betas)
        self.beta_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, num_betas)
        )
        
        # Pose parameters (theta)
        self.pose_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, num_pose)
        )
        
        # Camera parameters
        self.camera_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # scale and translation
        )
    
    def forward(self, features):
        return {
            'betas': self.beta_head(features),
            'pose': self.pose_head(features),
            'camera': {
                'scale': self.camera_head(features)[:, 0],
                'translation': self.camera_head(features)[:, 1:]
            }
        }

class SKELHead(nn.Module):
    def __init__(self, num_joints=24):
        super().__init__()
        self.num_joints = num_joints
        
        # Joint positions
        self.joint_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, num_joints * 3)
        )
        
        # Joint rotations
        self.rotation_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, num_joints * 3)
        )
    
    def forward(self, features):
        joints = self.joint_head(features).view(-1, self.num_joints, 3)
        rotations = self.rotation_head(features).view(-1, self.num_joints, 3)
        
        return {
            'joints': joints,
            'rotations': rotations
        }

class AcupointHead(nn.Module):
    def __init__(self, num_acupoints=84):
        super().__init__()
        self.num_acupoints = num_acupoints
        
        # 3D acupoint coordinates
        self.acupoint_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, num_acupoints * 3)  # 3D coordinates for each acupoint
        )
    
    def forward(self, features):
        acupoints = self.acupoint_head(features).view(-1, self.num_acupoints, 3)
        return acupoints 