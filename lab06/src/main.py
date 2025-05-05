import os
import argparse
import torch
from train import train
from test import test

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run training if specified
    if args.train:
        print("Starting training...")
        train_args = argparse.Namespace(
            data_dir=args.data_dir,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            model_channels=args.model_channels,
            time_dim=args.time_dim,
            use_adagn=args.use_adagn,
            num_groups=args.num_groups,
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            run_name=args.run_name,
            resume=args.resume,
            save_interval=args.save_interval,
            sample_interval=args.sample_interval,
            guidance_scale=args.guidance_scale,
            use_classifier_guidance=args.use_classifier_guidance
        )
        train(train_args)
    
    # Run testing if specified
    if args.test:
        print("Starting testing...")
        test_args = argparse.Namespace(
            data_dir=args.data_dir,
            image_dir=args.image_dir,
            output_dir=os.path.join(args.output_dir, "results"),
            checkpoint=args.checkpoint if args.checkpoint else os.path.join(args.output_dir, "checkpoints", "final.pth"),
            model_channels=args.model_channels,
            time_dim=args.time_dim,
            use_adagn=args.use_adagn,
            num_groups=args.num_groups,
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            batch_size=args.test_batch_size,
            guidance_scale=args.guidance_scale,
            use_classifier_guidance=args.use_classifier_guidance
        )
        test(test_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a conditional DDPM model")
    
    # Mode arguments
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Run testing")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./src", help="Path to the data directory")
    parser.add_argument("--image_dir", type=str, default="./iclevr", help="Path to the image directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Path to the output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint for testing")
    
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
    parser.add_argument("--lr", type=float, default=2e-4, help="Maximum learning rate for OneCycleLR")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval for saving model checkpoints")
    parser.add_argument("--sample_interval", type=int, default=500, help="Interval for generating samples")
    
    # Testing arguments
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for testing")
    
    # Guidance arguments
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="Scale for classifier guidance")
    parser.add_argument("--use_classifier_guidance", action="store_true", help="Use classifier guidance for sampling")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--run_name", type=str, default=None, help="Name for the wandb run")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.train and not args.test:
        parser.error("At least one of --train or --test must be specified")
    
    main(args)
