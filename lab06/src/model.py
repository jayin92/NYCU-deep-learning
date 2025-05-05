import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import AdaGroupNorm

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for time steps
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """
    Basic convolutional block with residual connection and adaptive group normalization
    """
    def __init__(self, in_ch, out_ch, emb_dim=None, up=False, use_adagn=False, num_groups=8):
        super().__init__()
        self.use_adagn = use_adagn
        self.up = up
        
        if up:
            self.conv1 = nn.Conv2d(in_ch*2, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
            
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch if not up else in_ch*2, out_ch, 1) if in_ch != out_ch or up else nn.Identity()
        
        # Normalization layers
        if use_adagn and emb_dim is not None:
            # Use AdaGroupNorm from diffusers
            self.norm1 = AdaGroupNorm(emb_dim, out_ch, num_groups)
            self.norm2 = AdaGroupNorm(emb_dim, out_ch, num_groups)
        else:
            # Standard time embedding if not using AdaGN
            self.time_mlp = nn.Linear(emb_dim, out_ch) if emb_dim else None
            self.norm1 = nn.BatchNorm2d(out_ch)
            self.norm2 = nn.BatchNorm2d(out_ch)
            
        self.activation = nn.SiLU()
        
    def forward(self, x, emb=None):
        # Residual connection
        residual = self.shortcut(x)
        
        # First Conv
        h = self.conv1(x)
        
        # Apply normalization
        if self.use_adagn and emb is not None:
            # AdaGroupNorm approach
            h = self.norm1(h, emb)
            h = self.activation(h)
        else:
            # Standard approach
            h = self.norm1(h)
            h = self.activation(h)
            
            # Add time embedding if using standard time embedding
            if hasattr(self, 'time_mlp') and self.time_mlp is not None and emb is not None:
                time_emb = self.activation(self.time_mlp(emb))
                h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        # Second Conv
        h = self.conv2(h)
        
        # Apply normalization
        if self.use_adagn and emb is not None:
            # AdaGroupNorm approach for second norm
            h = self.norm2(h, emb)
            h = self.activation(h)
        else:
            h = self.norm2(h)
            h = self.activation(h)
        
        # Add residual connection
        h = h + residual
        
        # Down or Upsample
        return self.transform(h)

class ConditionalUNet(nn.Module):
    """
    Conditional U-Net model for DDPM with concatenated time and label embeddings
    """
    def __init__(self, in_channels=3, model_channels=64, out_channels=3, num_classes=24, 
                 time_dim=256, use_adagn=False, num_groups=8, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.num_classes = num_classes
        self.use_adagn = use_adagn
        
        # Combined embedding dimension after concatenation
        self.emb_dim = time_dim * 2  # time_dim + time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Label embedding
        self.label_emb = nn.Sequential(
            nn.Linear(num_classes, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling
        self.down1 = Block(model_channels, model_channels*2, self.emb_dim, up=False, use_adagn=use_adagn, num_groups=num_groups)
        self.down2 = Block(model_channels*2, model_channels*4, self.emb_dim, up=False, use_adagn=use_adagn, num_groups=num_groups)
        self.down3 = Block(model_channels*4, model_channels*8, self.emb_dim, up=False, use_adagn=use_adagn, num_groups=num_groups)
        
        # Bottleneck
        self.bottleneck1 = nn.Conv2d(model_channels*8, model_channels*8, kernel_size=3, padding=1)
        self.bottleneck2 = nn.Conv2d(model_channels*8, model_channels*8, kernel_size=3, padding=1)
        
        # Bottleneck normalization (AdaGroupNorm or BatchNorm)
        if use_adagn:
            self.bottleneck_norm1 = AdaGroupNorm(self.emb_dim, model_channels*8, num_groups)
            self.bottleneck_norm2 = AdaGroupNorm(self.emb_dim, model_channels*8, num_groups)
        else:
            self.bottleneck_norm1 = nn.BatchNorm2d(model_channels*8)
            self.bottleneck_norm2 = nn.BatchNorm2d(model_channels*8)
        
        # Upsampling
        self.up1 = Block(model_channels*8, model_channels*4, self.emb_dim, up=True, use_adagn=use_adagn, num_groups=num_groups)
        self.up2 = Block(model_channels*4, model_channels*2, self.emb_dim, up=True, use_adagn=use_adagn, num_groups=num_groups)
        self.up3 = Block(model_channels*2, model_channels, self.emb_dim, up=True, use_adagn=use_adagn, num_groups=num_groups)
        
        # Output
        self.conv_out = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, model_channels) if use_adagn else nn.BatchNorm2d(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, t, labels):
        # Embed time and labels
        t_emb = self.time_mlp(t)
        c_emb = self.label_emb(labels)
        
        # Concatenate time and label embeddings instead of adding
        emb = torch.cat([t_emb, c_emb], dim=1)
        
        # Initial conv
        x = self.conv_in(x)
        
        # Downsample
        d1 = self.down1(x, emb)
        d2 = self.down2(d1, emb)
        d3 = self.down3(d2, emb)
        
        # Bottleneck
        bottleneck = self.bottleneck1(d3)
        
        # Apply normalization to bottleneck
        if self.use_adagn:
            # Use AdaGroupNorm from diffusers
            bottleneck = self.bottleneck_norm1(bottleneck, emb)
            bottleneck = F.silu(bottleneck)
        else:
            bottleneck = self.bottleneck_norm1(bottleneck)
            bottleneck = F.silu(bottleneck)
            
        bottleneck = self.bottleneck2(bottleneck)
        
        # Apply normalization to bottleneck
        if self.use_adagn:
            # Use AdaGroupNorm from diffusers
            bottleneck = self.bottleneck_norm2(bottleneck, emb)
            bottleneck = F.silu(bottleneck)
        else:
            bottleneck = self.bottleneck_norm2(bottleneck)
            bottleneck = F.silu(bottleneck)
        
        # Upsample with skip connections
        up1 = self.up1(torch.cat([bottleneck, d3], dim=1), emb)
        up2 = self.up2(torch.cat([up1, d2], dim=1), emb)
        up3 = self.up3(torch.cat([up2, d1], dim=1), emb)
        
        # Output
        return self.conv_out(up3)

def cosine_beta_schedule(timesteps, s=0.008, device="cuda"):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, device=device)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model
    """
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000, beta_schedule="linear", device="cuda"):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        elif beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(timesteps, device=device)
        
        # Pre-calculate different terms for closed form
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )
        
    def get_noisy_image(self, x_start, t):
        """
        Add noise to the input image according to the given timestep
        """
        noise = torch.randn_like(x_start)
        return (
            self.sqrt_alphas_cumprod[t, None, None, None] * x_start
            + self.sqrt_one_minus_alphas_cumprod[t, None, None, None] * noise
        ), noise
        
    def forward(self, x_start, labels, t=None):
        """
        Forward pass for training
        """
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
            
        # Add noise to the input image
        x_noisy, noise = self.get_noisy_image(x_start, t)
        
        # Predict the noise
        noise_pred = self.model(x_noisy, t, labels)
        
        return noise_pred, noise
    
    @torch.no_grad()
    def sample(self, labels, image_size=64, batch_size=16, channels=3, classifier_guidance_scale=0.0, classifier=None):
        """
        Sample new images from the trained model
        """
        # Start from pure noise
        img = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        
        # Iteratively denoise
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.model(img, t, labels)
            
            # Apply classifier guidance if provided
            if classifier is not None and classifier_guidance_scale > 0:
                with torch.enable_grad():
                    img_in = img.detach().requires_grad_(True)
                    
                    # Resize to 64x64 for classifier if needed
                    if img_in.shape[2] != 64 or img_in.shape[3] != 64:
                        img_resized = F.interpolate(img_in, size=(64, 64), mode='bilinear', align_corners=False)
                        logits = classifier(img_resized)
                    else:
                        logits = classifier(img_in)
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected_logprobs = torch.sum(labels * log_probs, dim=-1)
                    
                    # Compute gradient of log probability with respect to input image
                    grad = torch.autograd.grad(selected_logprobs.sum(), img_in)[0]
                
                predicted_noise = predicted_noise - classifier_guidance_scale * grad
            
            # Get alpha and beta values for current timestep
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            # No noise for the last step
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)
                
            # Update image using the reverse diffusion process
            img = (1 / torch.sqrt(alpha)) * (
                img - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
            
        # Normalize to [0, 1] range
        img = (img.clamp(-1, 1) + 1) / 2
        
        return img