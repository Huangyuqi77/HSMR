import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from lib.modeling.acupoint_model import AcupointModel
from lib.data.augmentation import AcupointDataset, AcupointAugmentation
from lib.platform import entrypoint_with_args

@entrypoint_with_args(exp='hsmr/train_acupoint')
def main(cfg: DictConfig):
    # Set random seed
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed)
    
    # Initialize model
    model = AcupointModel(
        acupoint_definitions_path=cfg.acupoint_definitions_path,
        num_acupoints=cfg.num_acupoints,
        smpl_model_path=cfg.smpl_model_path,
        skel_model_path=cfg.skel_model_path,
        pretrained_vit_path=cfg.pretrained_vit_path,
        device=cfg.device
    )
    
    # Initialize augmentation
    augmentation = AcupointAugmentation(
        occlusion_prob=cfg.augmentation.occlusion_prob,
        max_occlusion_size=cfg.augmentation.max_occlusion_size,
        pose_perturb_std=cfg.augmentation.pose_perturb_std
    )
    
    # Initialize datasets
    train_dataset = AcupointDataset(
        image_paths=cfg.train.image_paths,
        keypoint_paths=cfg.train.keypoint_paths,
        acupoint_paths=cfg.train.acupoint_paths,
        augmentation=augmentation
    )
    
    val_dataset = AcupointDataset(
        image_paths=cfg.val.image_paths,
        keypoint_paths=cfg.val.keypoint_paths,
        acupoint_paths=cfg.val.acupoint_paths,
        augmentation=None
    )
    
    # Initialize dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val.batch_size,
        shuffle=False,
        num_workers=cfg.val.num_workers
    )
    
    # Initialize callbacks
    callbacks = []
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.output_dir, 'checkpoints'),
        filename='acupoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    callbacks.append(checkpoint_callback)
    
    # Initialize logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(cfg.output_dir, 'tb_logs'),
        name=cfg.exp_name,
        version=''
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        callbacks=callbacks,
        **cfg.trainer
    )
    
    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )

if __name__ == '__main__':
    main() 