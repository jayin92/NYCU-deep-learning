import torch
import matplotlib.pyplot as plt
import argparse
from Trainer import VAE_Model
from torchvision import transforms
import os
from torch.utils.data import DataLoader
from dataloader import Dataset_Dance
from math import log10
import torch.nn.functional as F

def generate_psnr(img1, img2, data_range=1.):
    mse = F.mse_loss(img1, img2)
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr.item()

@torch.no_grad()
def evaluate_psnr_per_frame(model, args):
    model.eval()
    val_loader = model.val_dataloader()
    frame_psnr = None

    for img, label in val_loader:
        img = img.to(args.device)
        label = label.to(args.device)

        prev_img = img[:, 0]
        predicted = [prev_img.squeeze(0).cpu()]
        ground_truth = [label[:, 0].squeeze(0).cpu()]
        frame_psnr = []

        for idx in range(1, args.val_vi_len):
            next_img = img[:, idx]
            next_label = label[:, idx]
            output, _, _ = model.forward(prev_img, next_img, next_label)
            output = output.clamp(0, 1)
            output[output != output] = 0.5  # Replace NaNs
            output[output > 1] = 1
            output[output < 0] = 0

            predicted.append(output.squeeze(0).cpu())
            ground_truth.append(next_img.squeeze(0).cpu())

            psnr = generate_psnr(predicted[-1], ground_truth[-1])
            frame_psnr.append(psnr)

            prev_img = output

        break  # Only process the first sequence

    return frame_psnr


def plot_psnr(psnr_values, save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(psnr_values) + 1), psnr_values, marker='o', linewidth=1.5, markersize=1)
    plt.xlabel('Frame Index')
    plt.ylabel('PSNR (dB)')
    plt.title('Per-frame PSNR on Validation Sequence')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    from Trainer import get_parser

    args = get_parser().parse_args()
    args.test = True
    args.store_visualization = False
    args.use_wandb = False

    os.makedirs(args.save_root, exist_ok=True)

    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()

    psnr_values = evaluate_psnr_per_frame(model, args)
    plot_psnr(psnr_values, save_path=os.path.join(args.save_root, 'val_frame_psnr.png'))
