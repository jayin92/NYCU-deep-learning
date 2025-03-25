import sys
sys.path.append('src/models')
import torch.nn.functional as F
import torch.nn as nn
import torch
from unet import Up, OutConv



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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(32, 64, 3)
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
    def __init__(self, n_channels=3, n_classes=1):
        super(ResNet34_UNet, self).__init__()

        self.encoder = ResNet34Encoder()

        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = Up(64, 32)
        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        # Final convolution
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # Encoder path
        x1, x2, x3, x4, x5 = self.encoder(x)

        # Bridge
        x = self.bridge(x5)

        # Decoder path with skip connections
        x = self.up1(x, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        x = self.final_up(x)

        # Final convolution
        x = self.outc(x)

        return x


# Example usage
if __name__ == "__main__":
    # Create a sample input tensor: batch_size x channels x height x width
    x = torch.randn((1, 3, 572, 572))

    # Initialize the model
    model = ResNet34_UNet(n_channels=3, n_classes=1)

    # Print model parameter count
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")
    

    # Forward pass
    output = model(x)

    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
