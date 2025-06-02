import torch
import torch.nn as nn
from typing import Optional
from .networks.backbones.vit import ViT

class ViTBackbone(nn.Module):
    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        image_size: int = 256,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.image_size = image_size
        
        # Initialize ViT model
        self.vit = ViT(
            img_size=image_size,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            frozen_stages=-1,
            ratio=1,
            last_norm=True
        )
        
        # Load pretrained weights if provided
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.vit.load_state_dict(state_dict)
        
        # Move model to device
        self.vit = self.vit.to(device)
        
        # Freeze ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def preprocess_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input images for ViT model
        
        Args:
            x: Input images [B, 3, H, W] in range [-1, 1]
            
        Returns:
            Preprocessed images [B, 3, H, W] in range [0, 1]
        """
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        
        # Resize if needed
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W] in range [-1, 1]
            
        Returns:
            Features [B, 768]
        """
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Preprocess images
        x = self.preprocess_image(x)
        
        # Get ViT output
        with torch.no_grad():
            features = self.vit(x)
        
        # Reshape features to [B, 768]
        features = features.mean(dim=[-2, -1])
        
        return features 