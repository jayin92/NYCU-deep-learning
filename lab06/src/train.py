import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision.utils import make_grid, save_image
import argparse
from tqdm import tqdm
import numpy as np

from dataset import get_dataloader
from model import ConditionalUNet, DDPM
from evaluator import evaluation_model

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    
    # Setup wandb
    wandb.init(
        project="DLP-lab-06",
        name=args.run_name,
        config={
            "model_channels": args.model_channels,
            "time_dim": args.time_dim,
            "timesteps": args.timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "beta_schedule": args.beta_schedule,
            "use_adagn": args.use_adagn,
            "num_groups": args.num_groups,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "guidance_scale": args.guidance_scale,
            "use_classifier_guidance": args.use_classifier_guidance
        }
    )
    
    # Load data
    train_loader = get_dataloader(
        json_path=os.path.join(args.data_dir, "train.json"),
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        mode="train",
        shuffle=True
    )
    
    # Initialize model
    model = ConditionalUNet(
        in_channels=3,
        model_channels=args.model_channels,
        out_channels=3,
        num_classes=24,
        time_dim=args.time_dim,
        use_adagn=args.use_adagn,
        num_groups=args.num_groups,
        device=device
    ).to(device)
    
    # Initialize diffusion model
    diffusion = DDPM(
        model=model,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        timesteps=args.timesteps,
        device=device
    ).to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(diffusion.parameters(), lr=args.lr)
    
    # Calculate total number of training steps
    total_steps = args.epochs * len(train_loader)
    
    # Setup OneCycleLR scheduler with warmup and cosine annealing
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.0,  # 5% of training for warmup
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1000,  # min_lr = initial_lr/5
        anneal_strategy='cos'  # cosine annealing
    )
    
    # Load evaluator
    evaluator = evaluation_model()
    
    # Load checkpoint if exists
    start_epoch = 0
    if args.resume and os.path.exists(os.path.join(args.output_dir, "checkpoints", "latest.pth")):
        checkpoint = torch.load(os.path.join(args.output_dir, "checkpoints", "latest.pth"))
        diffusion.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        diffusion.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            noise_pred, noise_target = diffusion(images, labels)
            
            # Calculate loss
            loss = nn.MSELoss()(noise_pred, noise_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()  # Step the scheduler after each batch for OneCycleLR
            
            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log to wandb
            global_step = epoch * len(train_loader) + step
            wandb.log({"train/loss": loss.item()}, step=global_step)
            
            # Generate and log samples periodically
            if step % args.sample_interval == 0:
                diffusion.eval()
                
                # Sample a batch of fixed labels for visualization
                sample_labels = labels[:8].clone()
                
                # Generate samples
                samples = diffusion.sample(
                    labels=sample_labels,
                    batch_size=len(sample_labels),
                    classifier_guidance_scale=args.guidance_scale,
                    classifier=evaluator.resnet18 if args.use_classifier_guidance else None
                )
                
                # Save samples
                grid = make_grid(samples, nrow=4)
                save_image(grid, os.path.join(args.output_dir, "samples", f"epoch_{epoch}_step_{step}.png"))
                wandb.log({"samples": wandb.Image(grid)}, step=global_step)
                
                # Calculate accuracy using evaluator
                with torch.no_grad():
                    # Normalize samples for evaluator
                    norm_samples = samples * 2 - 1
                    accuracy = evaluator.eval(norm_samples.cuda(), sample_labels.cuda())
                    wandb.log({"train/sample_accuracy": accuracy}, step=global_step)
                    print(f"Sample accuracy: {accuracy:.4f}")
                
                diffusion.train()
        
        # Log epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch
        })
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model': diffusion.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': avg_loss,
        }, os.path.join(args.output_dir, "checkpoints", "latest.pth"))
        
        # Save model checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': diffusion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': avg_loss,
            }, os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch+1}.pth"))
        
        print(f"Epoch {epoch+1}/{args.epochs} completed. Average loss: {avg_loss:.6f}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model': diffusion.state_dict(),
    }, os.path.join(args.output_dir, "checkpoints", "final.pth"))
    
    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a conditional DDPM model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./src", help="Path to the data directory")
    parser.add_argument("--image_dir", type=str, default="./iclevr", help="Path to the image directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Path to the output directory")
    
    # Model arguments
    parser.add_argument("--model_channels", type=int, default=64, help="Base channel count for the model")
    parser.add_argument("--time_dim", type=int, default=256, help="Dimension of time embedding")
    parser.add_argument("--use_adagn", action="store_true", help="Use Adaptive Group Normalization")
    parser.add_argument("--num_groups", type=int, default=32, help="Number of groups for Group Normalization")
    
    # Diffusion arguments
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Starting value for beta schedule")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending value for beta schedule")
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "cosine"], help="Beta schedule")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lr_step", type=int, default=50, help="Step size for learning rate scheduler")
    parser.add_argument("--lr_gamma", type=float, default=0.5, help="Gamma for learning rate scheduler")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving model checkpoints")
    parser.add_argument("--sample_interval", type=int, default=500, help="Interval for generating samples")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the wandb run")
    
    # Guidance arguments
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="Scale for classifier guidance")
    parser.add_argument("--use_classifier_guidance", action="store_true", help="Use classifier guidance for sampling")
    
    args = parser.parse_args()
    train(args)
