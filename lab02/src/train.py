import os
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as T

from tqdm import tqdm
from datetime import datetime

from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score

def train(args):
    # Create a directory for saving checkpoints
    try:
        os.makedirs(args.save_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating save directory: {e}")
        raise
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_transforms = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor()
    ])

    val_transforms = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor()
    ])

    
    # Load the data
    train_dataset = load_dataset(args.data_path, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    val_dataset = load_dataset(args.data_path, mode='valid')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    test_dataset = load_dataset(args.data_path, mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Load the model
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=1)
    elif args.model == 'resnet34_unet':
        model = ResNet34_UNet(n_channels=3, n_classes=1)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    
    # Load the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Load the loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Load the learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=args.learning_rate,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
    else:
        scheduler = None
    
    # Load the checkpoint if provided
    start_epoch = 0
    best_val_dice = 0.0
    if args.checkpoint:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_dice = checkpoint.get('best_val_dice', 0.0)
            print(f"Loaded checkpoint from epoch {start_epoch} with validation Dice score {best_val_dice:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")

    # Lists to store metrics for plotting
    train_losses = []
    train_dice_scores = []
    val_losses = []
    val_dice_scores = []

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        for i, batch in train_pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            batch_dice = dice_score(outputs, masks)
            train_loss += loss.item()
            train_dice += batch_dice
            
            # Update the progress bar directly with current loss and dice score
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{batch_dice:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.4E}")
            
            # Update OneCycleLR scheduler if used
            if args.scheduler == 'onecycle':
                scheduler.step()
        
        # Calculate average metrics
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        
        train_losses.append(train_loss)
        train_dice_scores.append(train_dice)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc="Validation"):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_score(outputs, masks)
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)
        
        # Update ReduceLROnPlateau or CosineAnnealingLR scheduler
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.scheduler == 'cosine':
            scheduler.step()
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_dice': train_dice,
                'val_loss': val_loss,
                'val_dice': val_dice,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
            'best_val_dice': best_val_dice,
            'train_losses': train_losses,
            'train_dice_scores': train_dice_scores,
            'val_losses': val_losses,
            'val_dice_scores': val_dice_scores
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_checkpoint.pth'))
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint['best_val_dice'] = best_val_dice
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"New best model saved with validation Dice score: {best_val_dice:.4f}")
    
    # Final evaluation on test set
    print("\nTraining completed! Evaluating on test set...")
    
    # Load best model for testing
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with validation Dice score: {checkpoint['best_val_dice']:.4f}")
    
    model.eval()
    test_loss = 0
    test_dice = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            test_loss += loss.item()
            test_dice += dice_score(outputs, masks)
    
    test_loss /= len(test_loader)
    test_dice /= len(test_loader)
    
    print(f"Test Loss: {test_loss:.4f}, Test Dice: {test_dice:.4f}")
    
    if args.use_wandb:
        wandb.log({
            'test_loss': test_loss,
            'test_dice': test_dice
        })
    
    # Save final model with test results
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'test_loss': test_loss,
        'test_dice': test_dice,
        'best_val_dice': best_val_dice,
        'train_losses': train_losses,
        'train_dice_scores': train_dice_scores,
        'val_losses': val_losses,
        'val_dice_scores': val_dice_scores
    }
    torch.save(final_checkpoint, os.path.join(args.save_dir, f'{args.model}_final.pth'))
    
    print(f"Final model saved to {os.path.join(args.save_dir, f'{args.model}_final.pth')}")
    print("Training and evaluation completed!")
    return

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--num_workers', '-nw', type=int, default=4, help='number of workers for data loader')
    parser.add_argument('--image_size', type=int, default=256, help='image size')
    parser.add_argument('--model', type=str, default='unet', help='model to train', choices=['unet', 'resnet34_unet'])
    parser.add_argument('--scheduler', type=str, default='none', help='learning rate scheduler', choices=['none', 'plateau', 'cosine', 'onecycle'])
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking')
    parser.add_argument('--project', type=str, default='DLP2024-hw2', help='wandb project name')

    return parser.parse_args() 
 
if __name__ == "__main__":
    args = get_args()    
    if args.use_wandb:
        try:
            wandb.init(project=args.project, config=vars(args))
        except Exception as e:
            print(f"Error initializing wandb: {e}")
            args.use_wandb = False
    
    # Create unique save directory based on model and timestamp
    args.save_dir = os.path.join(args.output_dir, f'{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    
    print(f"Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    print("\nStarting Training...")
    train(args)