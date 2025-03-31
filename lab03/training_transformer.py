import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import wandb

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim, self.scheduler = self.configure_optimizers(args)
        self.prepare_training(args)
        self.device = args.device
        
    @staticmethod
    def prepare_training(args):
        os.makedirs(args.checkpoint_path, exist_ok=True)

    def train_one_epoch(self, train_dataloader, epoch, args):
        losses = []
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        self.model.train()
        for batch_idx, (images) in enumerate(pbar):
            x = images.to(args.device)
            logits, z_indices = self.model.forward(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1), ignore_index=self.model.mask_token_id)
            loss.backward()
            losses.append(loss.item())
            if (batch_idx + 1) % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            pbar.set_postfix(loss=loss.item(), lr=self.optim.param_groups[0]['lr'])
            
            # Log batch loss to wandb
            if args.use_wandb and batch_idx % args.wandb_log_interval == 0:
                wandb.log({
                    "batch": batch_idx + epoch * len(train_dataloader),
                    "train_batch_loss": loss.item(),
                    "learning_rate": self.optim.param_groups[0]['lr']
                })
                
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        if self.scheduler is not None:
            self.scheduler.step()
        return avg_loss

    def eval_one_epoch(self, val_dataloader, epoch, args):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, (images) in enumerate(val_dataloader):
                x = images.to(self.device)
                logits, z_indices = self.model.forward(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1), ignore_index=self.model.mask_token_id)
                losses.append(loss.item())
        avg_loss = np.mean(losses)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss
    

    def configure_optimizers(self, args):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        elif args.scheduler == 'none':
            scheduler = None
        return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints_transformer', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=40, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=5, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='Save CKPT per ** epochs(default: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'], help='Learning rate scheduler.')    

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    # wandb arguments
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='NYCU-DL-lab03', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--wandb_log_interval', type=int, default=10, help='Log to wandb every N batches')

    args = parser.parse_args()

    # Initialize wandb
    if args.use_wandb:
        wandb_config = {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "scheduler": args.scheduler,
            "accum_grad": args.accum_grad,
            "device": args.device,
        }
        
        # Initialize wandb run
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=wandb_config
        )
        
        # Add MaskGit config to wandb
        MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
        wandb.config.update(MaskGit_CONFIGS["model_param"])
    else:
        MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_path, exist_ok=True)
    
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:
    best_train = float('inf')
    best_val = float('inf')
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        train_loss = train_transformer.train_one_epoch(train_loader, epoch, args)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch, args)
        
        # Log metrics to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
        
        
        # Save checkpoints
        if epoch % args.save_per_epoch == 0:
            checkpoint = {
                'model': train_transformer.model.state_dict(),
                'optimizer': train_transformer.optim.state_dict(),
                'epoch': epoch,
                'best_train': best_train,
                'best_val': best_val,
                'args': args
            }
            if args.scheduler is not None:
                checkpoint['scheduler'] = train_transformer.scheduler.state_dict()
            
            torch.save(checkpoint, f"{args.checkpoint_path}/checkpoint_epoch_{epoch}.pth")
            
            # Log checkpoint to wandb
            if args.use_wandb:
                wandb.save(f"{args.checkpoint_path}/checkpoint_epoch_{epoch}.pth")

        if train_loss < best_train:
            best_train = train_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"{args.checkpoint_path}/best_train.pth")
            if args.use_wandb:
                wandb.run.summary["best_train_loss"] = best_train
                wandb.save(f"{args.checkpoint_path}/best_train.pth")
                
        if val_loss < best_val:
            best_val = val_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"{args.checkpoint_path}/best_val.pth")
            if args.use_wandb:
                wandb.run.summary["best_val_loss"] = best_val
                wandb.save(f"{args.checkpoint_path}/best_val.pth")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()