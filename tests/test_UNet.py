import torch
from spacenet.models.UNet_basic import UNet

#create fake input tensor
batch_size = 2
in_channels = 3
height = 64
width = 64
x = torch.randn(batch_size, in_channels, height, width)

num_classes = 4
model = UNet(in_channels=3, num_classes=num_classes, base_channels=8, depth=3)
expected_output_shape = (batch_size, num_classes, height, width)
#forward pass
output = model(x)
#check output shape
assert output.shape == expected_output_shape, f"Output shape {output.shape} does not match expected {expected_output_shape}"
print("✅ UNet forward pass output shape test passed.")