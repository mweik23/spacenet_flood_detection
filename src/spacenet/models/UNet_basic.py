import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two 3x3 convs with BatchNorm + ReLU.
    Keeps spatial size (padding=1).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Down(nn.Module):
    """
    Encoder step:
      - DoubleConv to produce features at this scale (skip connection).
      - 3x3 stride-2 conv to downsample for next scale (learnable).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.down = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x):
        x = self.conv(x)       # features at this scale (for skip)
        skip = x
        x = self.down(x)       # downsampled features for next level
        return x, skip

class Up(nn.Module):
    """
    Decoder step:
      - Bilinear upsample by factor 2.
      - Concatenate with encoder skip features.
      - DoubleConv to fuse.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        in_channels:   channels coming from lower (deeper) level
        skip_channels: channels from encoder skip connection
        out_channels:  desired output channels after fusion
        """
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        # Upsample to match skip's spatial size exactly
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    """
    UNet with:
      - Learnable stride-2 conv downsampling
      - Bilinear interpolation upsampling
      - Same spatial size in/out (for 'same padded' inputs)

    Args:
        in_channels:  input channels (e.g., 3 for RGB)
        num_classes:  output channels (e.g., K object types)
        base_channels: number of channels at first level
        depth:       number of encoder/decoder levels (excluding bottleneck)
    """
    def __init__(self, in_channels=3, num_classes=1,
                 base_channels=64, depth=4):
        super().__init__()

        assert depth >= 1

        # --- Encoder (Down path) ---
        # We’ll build a channel schedule like:
        # base, 2*base, 4*base, 8*base, ... for `depth` levels.
        enc_channels = [base_channels * (2 ** i) for i in range(depth)]
        self.down_blocks = nn.ModuleList()

        # First down block: in_channels -> base_channels
        self.down_blocks.append(Down(in_channels, enc_channels[0]))

        # Remaining down blocks
        for i in range(1, depth):
            self.down_blocks.append(Down(enc_channels[i - 1], enc_channels[i]))

        # --- Bottleneck ---
        bottleneck_channels = enc_channels[-1] * 2
        self.bottleneck = DoubleConv(enc_channels[-1], bottleneck_channels)

        # --- Decoder (Up path) ---
        self.up_blocks = nn.ModuleList()
        # Decoder goes from bottleneck back to first encoder level
        # At each step:
        #   in_channels = current feature channels
        #   skip_channels = encoder channels at that level
        #   out_channels = encoder channels at that level
        dec_in_channels = bottleneck_channels
        for i in reversed(range(depth)):
            skip_ch = enc_channels[i]
            out_ch = enc_channels[i]
            self.up_blocks.append(Up(dec_in_channels, skip_ch, out_ch))
            dec_in_channels = out_ch  # for next level up

        # --- Final 1x1 conv to get per-pixel logits ---
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = []
        for down in self.down_blocks:
            x, skip = down(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder (iterate skips in reverse order)
        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        # Final logits
        logits = self.head(x)  # [B, num_classes, H, W]
        return logits
