# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet import Up, OutConv


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsample to identity if needed
        identity = self.downsample(identity)
        
        # Add residual connection
        out = out + identity
        out = self.relu(out)
        
        return out

class ResNet34Encoder(nn.Module):
    def __init__(self):
        super(ResNet34Encoder, self).__init__()
        # Define the ResNet34 encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(64, 64, 3)
        self.conv3_x = self._make_layer(64, 128, 4, stride=2)
        self.conv4_x = self._make_layer(128, 256, 6, stride=2)
        self.conv5_x = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResNetBasicBlock(in_channels, out_channels, stride))
            in_channels = out_channels
            stride = 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        
        # First block after maxpool
        x2 = self.maxpool(x1)
        x2 = self.conv2_x(x2)
        
        # Remaining blocks
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)
        
        # Return feature maps for skip connections
        return x1, x2, x3, x4, x5


class ResNet34_UNet(nn.Module):
    def __init__(self, n_channels=3, num_classes=2):
        super(ResNet34_UNet, self).__init__()
        
        self.encoder = ResNet34Encoder()
        
        self.up1 = Up(512 + 256, 256)  # Combining x5 (512) and x4 (256)
        self.up2 = Up(256 + 128, 128)  # Combining up1 output (256) and x3 (128)
        self.up3 = Up(128 + 64, 64)    # Combining up2 output (128) and x2 (64)
        self.up4 = Up(64 + 64, 32)     # Combining up3 output (64) and x1 (64)
        
        # Final convolution
        self.outc = OutConv(32, num_classes)
    
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final convolution
        pred = self.outc(x)
        
        return pred
    
# Example usage
if __name__ == "__main__":
    # Create a sample input tensor: batch_size x channels x height x width
    x = torch.randn((1, 1, 572, 572))
    
    # Initialize the model
    model = ResNet34_UNet(n_channels=1, n_classes=2)
    
    # Forward pass
    output = model(x)
    
    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")