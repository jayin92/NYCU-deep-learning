import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
import argparse
import glob

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weight')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--model_type', type=str, default='unet', choices=['unet', 'resnet34_unet'], help='model architecture')
    parser.add_argument('--output_dir', type=str, default='predictions', help='output directory for saving predictions')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--img_size', type=int, default=256, help='input image size')
    
    return parser.parse_args()

def load_model(model_path, model_type, device):
    """
    Load a trained segmentation model
    
    Args:
        model_path: Path to the saved model weights
        model_type: Type of model architecture ('unet' or 'resnet34_unet')
        device: Device to load the model on
        
    Returns:
        The loaded model
    """
    if model_type == 'unet':
        from models.unet import UNet
        model = UNet(n_channels=3, n_classes=1)
    else:  # resnet34_unet
        from models.resnet34_unet import ResNet34_UNet
        model = ResNet34_UNet(n_channels=3, n_classes=1)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} model from {model_path}")
    return model

def preprocess_image(image_path, img_size=256):
    """
    Load and preprocess an image for inference
    
    Args:
        image_path: Path to the image file
        img_size: Size to resize the image to
        
    Returns:
        Processed image tensor ready for model input
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Define preprocessing transforms
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    image_tensor = transform(image)
    
    return image_tensor, image

def predict(model, image_tensor, device, threshold=0.5):
    """
    Generate a segmentation mask prediction
    
    Args:
        model: The trained segmentation model
        image_tensor: Input image tensor
        device: Device to run inference on
        threshold: Threshold for binary segmentation
        
    Returns:
        Predicted mask as numpy array
    """
    with torch.no_grad():
        # Add batch dimension and move to device
        x = image_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        output = model(x)
        
        # Apply sigmoid and threshold
        pred_mask = torch.sigmoid(output) > threshold
        
        # Convert to numpy
        pred_mask = pred_mask.cpu().squeeze().numpy().astype(np.uint8) * 255
        
    return pred_mask

def save_prediction(image, mask, output_path):
    """
    Visualize and save the prediction results
    
    Args:
        image: Original PIL image
        mask: Predicted mask as numpy array
        output_path: Path to save the visualization
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot predicted mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')
    
    # Plot overlay
    image_np = np.array(image.resize((mask.shape[1], mask.shape[0])))
    mask_rgb = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=2)
    overlay = image_np * 0.7 + mask_rgb * 0.3
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save raw mask as well
    mask_path = output_path.replace('.png', '_mask.png')
    mask_img = Image.fromarray(mask)
    mask_img.save(mask_path)

def process_batch(model, image_paths, device, output_dir, img_size):
    """
    Process a batch of images
    
    Args:
        model: The trained segmentation model
        image_paths: List of paths to image files
        device: Device to run inference on
        output_dir: Directory to save results
        img_size: Input image size
    """
    for image_path in tqdm(image_paths, desc="Processing images"):
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Preprocess image
        image_tensor, original_image = preprocess_image(image_path, img_size)
        
        # Generate prediction
        pred_mask = predict(model, image_tensor, device)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"{filename}_prediction.png")
        save_prediction(original_image, pred_mask, output_path)

def main():
    args = get_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model, args.model_type, device)
    
    # Get image paths
    if os.path.isdir(args.data_path):
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(args.data_path, ext)))
    else:
        # Single image
        image_paths = [args.data_path]
    
    if not image_paths:
        print(f"No images found in {args.data_path}")
        return
    
    print(f"Found {len(image_paths)} images for inference")
    
    # Process images in batches
    batch_size = args.batch_size
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        process_batch(model, batch_paths, device, args.output_dir, args.img_size)
    
    print(f"Inference completed. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()