import torch
import torch.nn.functional as F
import numpy as np
import json
import cv2
from typing import Tuple, Dict
from pathlib import Path

class AcupointAugmentation:
    def __init__(
        self,
        occlusion_prob: float = 0.3,
        max_occlusion_size: float = 0.2,
        pose_perturb_std: float = 0.1
    ):
        self.occlusion_prob = occlusion_prob
        self.max_occlusion_size = max_occlusion_size
        self.pose_perturb_std = pose_perturb_std
    
    def __call__(
        self,
        images: torch.Tensor,
        keypoints: torch.Tensor,
        acupoints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to images and keypoints
        
        Args:
            images: Input images [B, 3, H, W]
            keypoints: 2D keypoints [B, N, 2]
            acupoints: 2D acupoints [B, M, 2]
            
        Returns:
            Augmented images, keypoints, and acupoints
        """
        # Random occlusion
        if np.random.random() < self.occlusion_prob:
            images = self._apply_occlusion(images)
        
        # Pose perturbation
        keypoints, acupoints = self._perturb_pose(keypoints, acupoints)
        
        return images, keypoints, acupoints
    
    def _apply_occlusion(self, images: torch.Tensor) -> torch.Tensor:
        """Apply random occlusion to images"""
        batch_size, _, height, width = images.shape
        
        # Generate random occlusion boxes
        num_boxes = np.random.randint(1, 4)
        for _ in range(num_boxes):
            # Random box size
            box_h = int(height * np.random.uniform(0.1, self.max_occlusion_size))
            box_w = int(width * np.random.uniform(0.1, self.max_occlusion_size))
            
            # Random position
            top = np.random.randint(0, height - box_h)
            left = np.random.randint(0, width - box_w)
            
            # Apply occlusion
            images[:, :, top:top+box_h, left:left+box_w] = 0
        
        return images
    
    def _perturb_pose(
        self,
        keypoints: torch.Tensor,
        acupoints: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random pose perturbation to keypoints and acupoints"""
        # Generate random perturbation
        perturbation = torch.randn_like(keypoints) * self.pose_perturb_std
        
        # Apply perturbation
        keypoints = keypoints + perturbation
        
        # Scale perturbation for acupoints (usually smaller)
        acupoint_perturbation = perturbation[:, :acupoints.shape[1]] * 0.5
        acupoints = acupoints + acupoint_perturbation
        
        return keypoints, acupoints

class AcupointDataset:
    def __init__(
        self,
        image_paths: list,
        acupoint_paths: list,
        augmentation: AcupointAugmentation = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        self.image_paths = image_paths
        self.acupoint_paths = acupoint_paths
        self.augmentation = augmentation
        self.image_size = image_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict:
        # Load image
        image = self._load_image(self.image_paths[idx])
        
        # Load acupoints
        acupoints = self._load_acupoints(self.acupoint_paths[idx])
        
        # Apply augmentation if specified
        if self.augmentation is not None:
            image, _, acupoints = self.augmentation(
                image.unsqueeze(0),
                torch.zeros(1, 1, 2),  # Dummy keypoints
                acupoints.unsqueeze(0)
            )
            image = image.squeeze(0)
            acupoints = acupoints.squeeze(0)
        
        return {
            'image': image,
            'acupoints': acupoints
        }
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess image"""
        # Read image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # Scale to [-1, 1]
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image
    
    def _load_acupoints(self, path: str) -> torch.Tensor:
        """Load acupoints from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert to tensor
        acupoints = torch.zeros(84, 2)  # 84 acupoints, each with x,y coordinates
        for i in range(84):
            if str(i) in data:
                acupoints[i] = torch.tensor(data[str(i)])
        
        return acupoints 