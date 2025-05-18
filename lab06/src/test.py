import os
import json
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import numpy as np
from torchvision.utils import make_grid, save_image

from dataset import get_dataloader
from model import ConditionalUNet, DDPM
from evaluator import evaluation_model

def test(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "new_test"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "grids"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "denoising_process"), exist_ok=True)
    
    # Load test data
    test_loader = get_dataloader(
        json_path=os.path.join(args.data_dir, "test.json"),
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        mode="test",
        shuffle=False
    )
    
    new_test_loader = get_dataloader(
        json_path=os.path.join(args.data_dir, "new_test.json"),
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        mode="test",
        shuffle=False
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
    
    # Load checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        diffusion.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        raise ValueError(f"Checkpoint {args.checkpoint} does not exist")
    
    # Load evaluator
    evaluator = evaluation_model()
    
    # Set model to evaluation mode
    diffusion.eval()
    
    # Load object mapping
    with open(os.path.join(args.data_dir, "objects.json"), 'r') as f:
        object_mapping = json.load(f)
    
    # Load test data
    with open(os.path.join(args.data_dir, "test.json"), 'r') as f:
        test_data = json.load(f)
    
    with open(os.path.join(args.data_dir, "new_test.json"), 'r') as f:
        new_test_data = json.load(f)
    
    # Generate images for test.json
    print("Generating images for test.json...")
    test_images = []
    test_labels = []
    
    for i, (_, label) in enumerate(tqdm(test_loader)):
        label = label.to(device)
        test_labels.append(label)
        
        with torch.no_grad():
            samples = diffusion.sample(
                labels=label,
                batch_size=len(label),
                classifier_guidance_scale=args.guidance_scale,
                classifier=evaluator.resnet18 if args.use_classifier_guidance else None
            )
            
            test_images.append(samples)
            
            # Save individual images
            for j, img in enumerate(samples):
                idx = i * args.batch_size + j
                if idx < len(test_data):
                    save_image(img, os.path.join(args.output_dir, "test", f"{idx}.png"))
    
    # Concatenate all images and labels
    test_images = torch.cat(test_images)
    test_labels = torch.cat(test_labels)
    
    # Create grid of test images
    test_grid = make_grid(test_images[:32], nrow=8, normalize=False)
    save_image(test_grid, os.path.join(args.output_dir, "grids", "test_grid.png"))
    
    # Calculate accuracy for test.json
    norm_test_images = test_images * 2 - 1 # Normalize to [-1, 1] for evaluator
    
    # Resize images to 64x64 for evaluation if they're not already that size
    if norm_test_images.shape[2] != 64 or norm_test_images.shape[3] != 64:
        print(f"Resizing images from {norm_test_images.shape[2]}x{norm_test_images.shape[3]} to 64x64 for evaluation")
        norm_test_images_resized = F.interpolate(norm_test_images[:len(test_data)].cuda(), size=(64, 64), mode='bilinear', align_corners=False)
        test_accuracy = evaluator.eval(norm_test_images_resized, test_labels[:len(test_data)].cuda())
    else:
        test_accuracy = evaluator.eval(norm_test_images[:len(test_data)].cuda(), test_labels[:len(test_data)].cuda())
    print(f"Accuracy on test.json: {test_accuracy:.4f}")
    
    # Generate images for new_test.json
    print("Generating images for new_test.json...")
    new_test_images = []
    new_test_labels = []
    
    for i, (_, label) in enumerate(tqdm(new_test_loader)):
        label = label.to(device)
        new_test_labels.append(label)
        
        with torch.no_grad():
            samples = diffusion.sample(
                labels=label,
                batch_size=len(label),
                classifier_guidance_scale=args.guidance_scale,
                classifier=evaluator.resnet18 if args.use_classifier_guidance else None
            )
            
            new_test_images.append(samples)
            
            # Save individual images
            for j, img in enumerate(samples):
                idx = i * args.batch_size + j
                if idx < len(new_test_data):
                    save_image(img, os.path.join(args.output_dir, "new_test", f"{idx}.png"))
    
    # Concatenate all images and labels
    new_test_images = torch.cat(new_test_images)
    new_test_labels = torch.cat(new_test_labels)
    
    # Create grid of new test images
    new_test_grid = make_grid(new_test_images[:32], nrow=8, normalize=False)
    save_image(new_test_grid, os.path.join(args.output_dir, "grids", "new_test_grid.png"))
    
    # Calculate accuracy for new_test.json
    norm_new_test_images = new_test_images * 2 - 1 # Normalize to [-1, 1] for evaluator
    
    
    # Resize images to 64x64 for evaluation if they're not already that size
    if norm_new_test_images.shape[2] != 64 or norm_new_test_images.shape[3] != 64:
        print(f"Resizing images from {norm_new_test_images.shape[2]}x{norm_new_test_images.shape[3]} to 64x64 for evaluation")
        norm_new_test_images_resized = F.interpolate(norm_new_test_images[:len(new_test_data)].cuda(), size=(64, 64), mode='bilinear', align_corners=False)
        new_test_accuracy = evaluator.eval(norm_new_test_images_resized, new_test_labels[:len(new_test_data)].cuda())
    else:
        new_test_accuracy = evaluator.eval(norm_new_test_images[:len(new_test_data)].cuda(), new_test_labels[:len(new_test_data)].cuda())
    print(f"Accuracy on new_test.json: {new_test_accuracy:.4f}")
    
    # Generate denoising process visualization for ["red sphere", "cyan cylinder", "cyan cube"]
    print("Generating denoising process visualization...")
    
    # Create label for ["red sphere", "cyan cylinder", "cyan cube"]
    special_label = torch.zeros(1, 24, device=device)
    special_label[0, object_mapping["red sphere"]] = 1
    special_label[0, object_mapping["cyan cylinder"]] = 1
    special_label[0, object_mapping["cyan cube"]] = 1
    
    # Visualize denoising process
    timesteps_to_save = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]

    # Start from pure noise
    img = torch.randn(1, 3, 64, 64, device=device)
    process_images = []
    
    # Save initial noisy image
    process_images.append(img.clone())
    
    # Iteratively denoise and save intermediate steps
    for i in reversed(range(0, diffusion.timesteps)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # Predict noise
            predicted_noise = diffusion.model(img, t, special_label)
            
            # Apply classifier guidance if enabled
            if args.use_classifier_guidance and args.guidance_scale > 0:
                with torch.enable_grad():
                    img_in = img.detach().requires_grad_(True)
                    logits = evaluator.resnet18(img_in)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    selected_logprobs = torch.sum(special_label * log_probs, dim=-1)
                    
                    # Compute gradient of log probability with respect to input image
                    grad = torch.autograd.grad(selected_logprobs.sum(), img_in)[0]
                    
                    # Apply classifier guidance
                    predicted_noise = predicted_noise - args.guidance_scale * grad
            
            # Get alpha and beta values for current timestep
            alpha = diffusion.alphas[i]
            alpha_cumprod = diffusion.alphas_cumprod[i]
            beta = diffusion.betas[i]
            
            # No noise for the last step
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)
                
            # Update image using the reverse diffusion process
            img = (1 / torch.sqrt(alpha)) * (
                img - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
            # Save image at specified timesteps
            if i in timesteps_to_save:
                process_images.append(img.clone())
    
    # Normalize images to [0, 1] range
    process_images = [(img.clamp(-1, 1) + 1) / 2 for img in process_images]
    
    # Create grid of denoising process
    process_grid = make_grid(torch.cat(process_images), nrow=len(process_images), normalize=False)
    save_image(process_grid, os.path.join(args.output_dir, "denoising_process", "denoising_process.png"))
    
    # Save individual steps of the denoising process
    for i, img in enumerate(process_images):
        save_image(img, os.path.join(args.output_dir, "denoising_process", f"step_{i}.png"))
    
    # Save results to a text file
    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        f.write(f"Accuracy on test.json: {test_accuracy:.4f}\n")
        f.write(f"Accuracy on new_test.json: {new_test_accuracy:.4f}\n")
    
    print("Testing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a conditional DDPM model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./src", help="Path to the data directory")
    parser.add_argument("--image_dir", type=str, default="./iclevr", help="Path to the image directory")
    parser.add_argument("--output_dir", type=str, default="./results", help="Path to the output directory")
    parser.add_argument("--checkpoint", type=str, default="./output/checkpoints/final.pth", help="Path to the model checkpoint")
    
    # Model arguments
    parser.add_argument("--model_channels", type=int, default=64, help="Base channels in the UNet model")
    parser.add_argument("--time_dim", type=int, default=256, help="Dimension of time embedding")
    parser.add_argument("--use_adagn", action="store_true", help="Use Adaptive Group Normalization")
    parser.add_argument("--num_groups", type=int, default=8, help="Number of groups for Group Normalization")
    
    # Diffusion arguments
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--beta_start", type=float, default=1e-4, help="Starting value for beta schedule")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Ending value for beta schedule")
    
    # Testing arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    
    # Guidance arguments
    parser.add_argument("--guidance_scale", type=float, default=1.5, help="Scale for classifier guidance")
    parser.add_argument("--use_classifier_guidance", action="store_true", help="Use classifier guidance for sampling")
    
    args = parser.parse_args()
    test(args)
