import os
import argparse
import math
import wandb
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    # Clamp logvar to prevent extreme values
    logvar = torch.clamp(logvar, min=-10, max=10)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalize by batch size and ALL feature dimensions
    total_elements = batch_size * mu.shape[1] * mu.shape[2] * mu.shape[3]  # batch × 12 × 32 × 64
    KLD /= total_elements

    return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.type = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle
        self.ratio = args.kl_anneal_ratio
        self.current_epoch = current_epoch
        self.current_epoch -= 1  # Because update() will increment it
        self.betas = self.frange_cycle_linear(args.num_epoch, start=0.0, stop=1.0, n_cycle=self.cycle, ratio=self.ratio)
        self.update()
        self.epoch_per_cycle = args.num_epoch / self.cycle
        
    def update(self):
        self.current_epoch += 1
        if self.type == 'Cyclical':
            try:
                beta = self.betas[self.current_epoch]
            except:
                beta = 1.0
        elif self.type == 'Monotonic':
            beta = min(1.0, self.current_epoch / self.epoch_per_cycle)
        elif self.type == 'None':
            beta = 1.0
        
        self.beta = beta
    
    def get_beta(self):
        return self.beta
    
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        epoch_per_cycle = n_iter / n_cycle
        betas = []
        for i in range(n_iter):
            tau = (i % math.ceil(epoch_per_cycle)) / epoch_per_cycle
            if tau <= ratio:
                beta = start + tau / ratio * (stop - start)
            else:
                beta = stop
            betas.append(beta)
        
        betas.append(stop)
        return betas


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        if args.optim == "Adam":
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        elif args.optim == "AdamW":
            self.optim      = optim.AdamW(self.parameters(), lr=self.args.lr)
        else:
            raise ValueError("Optimizer not supported")
        # self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=args.num_epoch, eta_min=args.lr_min)
        
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 1
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        
        
    def forward(self, img_prev, img_next, label_next):
        img_prev_encoded = self.frame_transformation(img_prev).detach()  # [batch_size, F_dim]
        img_next_encoded = self.frame_transformation(img_next)  # [batch_size, F_dim]
        label_next_encoded = self.label_transformation(label_next)  # [batch_size, L_dim]

        # Concatenate the transformed image and label
        z, mu, logvar = self.Gaussian_Predictor.forward(img_next_encoded, label_next_encoded)  # [batch_size, N_dim]
        
        # Decoder Fusion
        x = self.Decoder_Fusion.forward(img_prev_encoded, label_next_encoded, z)  # [batch_size, D_out_dim]
        output = self.Generator.forward(x)  # [batch_size, 3, H, W]

        return output, mu, logvar

    def training_stage(self):

        for i in range(self.args.num_epoch - self.current_epoch + 1):
            print(f"Epoch {self.current_epoch}/{self.args.num_epoch}")
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            # Track metrics for the epoch
            epoch_loss = 0
            epoch_mse_loss = 0
            epoch_kld_loss = 0
            step_count = 0
            
            for (imgs, labels) in (pbar := tqdm(train_loader)):
                imgs = imgs.to(self.args.device)
                labels = labels.to(self.args.device)
                loss, mse_loss, kld_loss = self.training_one_step(imgs, labels, adapt_TeacherForcing)
                
                epoch_loss += loss.item()
                epoch_mse_loss += mse_loss.item()
                epoch_kld_loss += kld_loss.item()
                step_count += 1
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {:.3e}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {:.3e}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
            
            # Calculate average metrics for the epoch
            avg_loss = epoch_loss / max(1, step_count)
            avg_mse_loss = epoch_mse_loss / max(1, step_count)
            avg_kld_loss = epoch_kld_loss / max(1, step_count)
            
            # Log metrics to wandb
            if self.args.use_wandb:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/mse_loss": avg_mse_loss,
                    "train/kld_loss": avg_kld_loss,
                    "train/beta": self.kl_annealing.get_beta(),
                    "train/tfr": self.tfr,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "epoch": self.current_epoch,
                })
            
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
                # Save model checkpoint to wandb
                if self.args.use_wandb:
                    wandb.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
                
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
        
        # Save the final model checkpoint
        self.save(os.path.join(self.args.save_root, f"final_model.ckpt"))
        if self.args.use_wandb:
            wandb.save(os.path.join(self.args.save_root, f"final_model.ckpt"))
        print("Training complete. Model saved.")
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        
        total_val_loss = 0.0
        total_val_mse = 0.0
        total_val_kld = 0.0
        total_psnr = 0.0
        valid_psnr_count = 0
        step_count = 0
        all_generated_frames = []
        
        for (img, label) in (pbar := tqdm(val_loader)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, mse, kld, psnr, generated_frames = self.val_one_step(img, label)
            
            total_val_loss += loss.item()
            total_val_mse += mse.item()
            total_val_kld += kld.item()
            if psnr is not None and psnr > 0:
                total_psnr += psnr
                valid_psnr_count += 1
            step_count += 1
            
            # Save frames for visualization
            if step_count == 1:  # Only keep the first sequence for visualization
                all_generated_frames = generated_frames
                
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        
        # Calculate average metrics
        avg_val_loss = total_val_loss / max(1, step_count)
        avg_val_mse = total_val_mse / max(1, step_count)
        avg_val_kld = total_val_kld / max(1, step_count)
        avg_psnr = total_psnr / max(1, valid_psnr_count) if valid_psnr_count > 0 else 0
        
        # Log validation metrics and media to wandb
        if self.args.use_wandb:
            log_dict = {
                "val/loss": avg_val_loss,
                "val/mse_loss": avg_val_mse,
                "val/kld_loss": avg_val_kld,
                "epoch": self.current_epoch,
            }
            
            if avg_psnr > 0:
                log_dict["val/psnr"] = avg_psnr
                
            # Log generated images if we're on a save epoch
            if self.args.store_visualization and self.current_epoch % self.args.per_save == 0 and len(all_generated_frames) > 0:
                # Create a GIF for wandb
                gif_path = os.path.join(self.args.save_root, f"visualizations/epoch_{self.current_epoch}/val_seq.gif")
                if os.path.exists(gif_path):
                    log_dict["val/generated_sequence"] = wandb.Video(gif_path, fps=15, format="gif")
                    
            wandb.log(log_dict)
    
    def training_one_step(self, imgs, labels, adapt_TeacherForcing):
        assert imgs.shape[1] == self.train_vi_len, "Batch size must be equal to the video length"
        assert labels.shape[1] == self.train_vi_len, "Batch size must be equal to the video length"

        # imgs.shape: [batch_size, video_len, 3, H, W]
        # labels.shape: [batch_size, video_len, 3, H, W]
        batch_size = imgs.shape[0]

        total_kld = 0.0
        total_mse = 0.0

        prev_img = imgs[:, 0]

        for idx in range(1, self.train_vi_len):
            next_img = imgs[:, idx]
            next_label = labels[:, idx]
            
            # Forward pass
            output, mu, logvar = self.forward(prev_img, next_img, next_label)
            output = output.clamp(0, 1)  # Clamp output to [0, 1]
            # Replace nan and inf with 0.5
            output[output != output] = 0.5
            
            # Compute the loss
            kld = kl_criterion(mu, logvar, batch_size)
            mse = self.mse_criterion(output, next_img)
            
            total_kld += kld
            total_mse += mse

            if adapt_TeacherForcing:
                # Teacher forcing
                prev_img = next_img
            else:
                # Use the generated image as input for the next step
                prev_img = output.detach()
        
        loss = total_mse
        if self.kl_annealing.get_beta() > 0:
            loss += self.kl_annealing.get_beta() * total_kld
        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()
            
        return loss, total_mse, total_kld
    
    def val_one_step(self, img, label):
        # Get the first frame as starting frame
        prev_img = img[:, 0]  # Shape: [1, 3, H, W]
        
        total_mse = 0.0
        total_kld = 0.0
        
        generated_frames = [prev_img.squeeze(0).cpu()]
        
        # Process each frame in the sequence
        for idx in range(1, self.val_vi_len):
            next_img = img[:, idx]  # Shape: [1, 3, H, W]
            next_label = label[:, idx]  # Shape: [1, 3, H, W]
            
            # Forward pass
            output, mu, logvar = self.forward(prev_img, next_img, next_label)
            output = output.clamp(0, 1)  # Clamp output to [0, 1]
            # Replace nan and inf with 0.5
            output[output != output] = 0.5
            
            # Compute losses
            kld = kl_criterion(mu, logvar, 1)
            mse = self.mse_criterion(output, next_img)
            
            total_kld += kld
            total_mse += mse
            
            # Save generated frame
            generated_frames.append(output.squeeze(0).cpu())
            
            # Always use the model's output as the next input during validation
            prev_img = output
        
        # Combine losses with KL annealing
        loss = total_mse + self.kl_annealing.get_beta() * total_kld
        
        avg_psnr = None
        original_frames = [img[0, i].cpu() for i in range(self.val_vi_len)]
        psnr_values = [Generate_PSNR(original_frames[i], generated_frames[i]) for i in range(1, self.val_vi_len)]
        avg_psnr = sum(psnr_values) / len(psnr_values)
            
        print(f"Validation PSNR: {avg_psnr:.2f} dB")
        # Save visualization if enabled
        if self.args.store_visualization and self.current_epoch % self.args.per_save == 0:
            vis_dir = os.path.join(self.args.save_root, f"visualizations/epoch_{self.current_epoch}")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save as GIF
            self.make_gif(generated_frames, os.path.join(vis_dir, f"val_seq.gif"))
                        
            # Save sample frames as images for detailed inspection
            if self.args.use_wandb and len(psnr_values) > 0:
                # Save first, middle, and last frames for comparison
                indices = [1, len(psnr_values) // 2, len(psnr_values) - 1]
                for idx in indices:
                    # Save original and generated frames side by side
                    orig_img_path = os.path.join(vis_dir, f"orig_frame_{idx}.png")
                    gen_img_path = os.path.join(vis_dir, f"gen_frame_{idx}.png")
                    
                    # Save original frame
                    orig_frame = transforms.ToPILImage()(original_frames[idx])
                    orig_frame.save(orig_img_path)
                    
                    # Save generated frame
                    gen_frame = transforms.ToPILImage()(generated_frames[idx])
                    gen_frame.save(gen_img_path)
                    
                    # Log images to wandb
                    wandb.log({
                        f"val/comparison_frame_{idx}": [
                            wandb.Image(orig_img_path, caption=f"Original Frame {idx}"),
                            wandb.Image(gen_img_path, caption=f"Generated Frame {idx}")
                        ]
                    })
                    
        return loss, total_mse, total_kld, avg_psnr, generated_frames
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
       if self.current_epoch >= self.tfr_sde:
            self.tfr = max(0.0, self.tfr - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr:.3e}" , refresh=False)
        pbar.set_postfix(loss=f"{float(loss):.3e}", refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()


def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    
    # Initialize wandb
    if not args.test and args.use_wandb:
        # Set wandb run name if not provided
        
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
       
    model = VAE_Model(args).to(args.device)
        
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()
        
    # Close wandb
    if not args.test and args.use_wandb:
        wandb.finish()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--lr_min',            type=float,  default=0.0,     help="final learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=32)
    parser.add_argument('--num_epoch',     type=int, default=70   ,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    
    # Checkpoint path
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="", choices=["Cyclical", "Monotonic", "None"])
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.5,              help="")
    
    # WandB related arguments
    parser.add_argument('--use_wandb',          action='store_true',         help="Whether to use Weights & Biases for experiment tracking")
    parser.add_argument('--wandb_project',      type=str, default='NYCU-DL-lab04', help="WandB project name")
    parser.add_argument('--wandb_run_name',     type=str, default=None,      help="WandB run name")

    args = parser.parse_args()
    
    main(args)