import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import dice_score
import argparse

def evaluate(net, data_loader, device, output_dir=None, save_visualizations=False):
    """
    Evaluate a trained model on a dataset
    
    Args:
        net: The trained neural network model
        data_loader: DataLoader for the evaluation dataset
        device: Device to run the evaluation on (cuda or cpu)
        output_dir: Directory to save visualization results (if None, no visualizations are saved)
        save_visualizations: Whether to save visualizations of segmentation results
        
    Returns:
        avg_dice: Average Dice score across the dataset
        all_dices: List of individual Dice scores for each sample
    """
    net.eval()
    all_dices = []
    
    # Create output directory if it doesn't exist
    if output_dir is not None and save_visualizations:
        os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = net(images)
            
            # Calculate Dice score for each image in the batch
            for j in range(images.size(0)):
                img_dice = dice_score(outputs[j:j+1], masks[j:j+1])
                all_dices.append(img_dice)
            
            # Save visualization for some samples
            if save_visualizations and output_dir is not None and i % 5 == 0:
                for j in range(min(4, images.size(0))):  # Visualize up to 4 images per batch
                    visualize_prediction(
                        images[j].cpu(), 
                        masks[j].cpu(), 
                        outputs[j].cpu(),
                        os.path.join(output_dir, f'sample_{i}_{j}.png')
                    )
    
    # Calculate average Dice score
    avg_dice = sum(all_dices) / len(all_dices)
    
    return avg_dice, all_dices

def visualize_prediction(image, true_mask, pred_logits, save_path=None):
    """
    Visualize the model's prediction alongside the input image and ground truth mask
    
    Args:
        image: Input image tensor [C, H, W]
        true_mask: Ground truth mask tensor [1, H, W]
        pred_logits: Model's prediction logits tensor [1, H, W]
        save_path: Path to save the visualization image
    """
    # Denormalize the image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    
    # Convert tensors to numpy for visualization
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)
    
    true_mask = true_mask.squeeze().numpy()
    
    # Apply sigmoid to get probabilities and threshold to get binary mask
    pred_probs = torch.sigmoid(pred_logits).squeeze().numpy()
    pred_mask = (pred_probs > 0.5).astype(np.float32)
    
    # Create the figure with three subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Plot the original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot the ground truth mask
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Plot the predicted probability map
    axes[2].imshow(pred_probs, cmap='viridis')
    axes[2].set_title('Prediction Probabilities')
    axes[2].axis('off')
    
    # Plot the thresholded prediction mask
    axes[3].imshow(pred_mask, cmap='gray')
    axes[3].set_title('Predicted Mask')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_metrics(all_dices):
    """
    Calculate various metrics from the list of Dice scores
    
    Args:
        all_dices: List of Dice scores for all evaluated samples
        
    Returns:
        Dictionary of metrics including mean, median, std, min, max
    """
    metrics = {
        'dice_mean': np.mean(all_dices),
        'dice_median': np.median(all_dices),
        'dice_std': np.std(all_dices),
        'dice_min': np.min(all_dices),
        'dice_max': np.max(all_dices)
    }
    return metrics

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained segmentation model')
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--data_path', '-d', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='Output directory for visualization results')
    parser.add_argument('--model_type', '-t', type=str, default='unet', choices=['unet', 'resnet34_unet'], help='Model type')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--num_workers', '-w', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--save_viz', action='store_true', help='Save visualization of predictions')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import modules dynamically to prevent circular imports
    from oxford_pet import load_dataset
    
    # Load test dataset
    test_dataset = load_dataset(args.data_path, mode='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load the model
    if args.model_type == 'unet':
        from models.unet import UNet
        model = UNet(n_channels=3, n_classes=1)
    else:  # resnet34_unet
        from models.resnet34_unet import ResNet34_UNet
        model = ResNet34_UNet(n_channels=3, n_classes=1)
    
    # Load model weights
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Model loaded from {args.model}")
    print(f"Test Dice score: {checkpoint['test_dice']}")
    
    # Evaluate the model
    avg_dice, all_dices = evaluate(model, test_loader, device, args.output_dir, args.save_viz)
    
    # Calculate and display metrics
    metrics = calculate_metrics(all_dices)
    
    print("\nEvaluation Results:")
    print(f"Average Dice Score: {metrics['dice_mean']:.4f}")
    print(f"Median Dice Score: {metrics['dice_median']:.4f}")
    print(f"Standard Deviation: {metrics['dice_std']:.4f}")
    print(f"Min Dice Score: {metrics['dice_min']:.4f}")
    print(f"Max Dice Score: {metrics['dice_max']:.4f}")
    
    # Plot histogram of Dice scores
    if args.output_dir:
        plt.figure(figsize=(10, 6))
        plt.hist(all_dices, bins=20, alpha=0.7, color='blue')
        plt.axvline(metrics['dice_mean'], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {metrics["dice_mean"]:.4f}')
        plt.axvline(metrics['dice_median'], color='green', linestyle='dashed', linewidth=2, label=f'Median: {metrics["dice_median"]:.4f}')
        plt.title('Distribution of Dice Scores')
        plt.xlabel('Dice Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'dice_distribution.png'), dpi=150)
        print(f"Dice score distribution saved to {os.path.join(args.output_dir, 'dice_distribution.png')}")
        
        # Save metrics to a text file
        with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Model: {args.model_type}\n")
            f.write(f"Average Dice Score: {metrics['dice_mean']:.4f}\n")
            f.write(f"Median Dice Score: {metrics['dice_median']:.4f}\n")
            f.write(f"Standard Deviation: {metrics['dice_std']:.4f}\n")
            f.write(f"Min Dice Score: {metrics['dice_min']:.4f}\n")
            f.write(f"Max Dice Score: {metrics['dice_max']:.4f}\n")
        print(f"Metrics saved to {os.path.join(args.output_dir, 'metrics.txt')}")