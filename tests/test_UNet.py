import torch
from spacenet.models.UNet_basic import UNet
import pytest
from spacenet.configs import ModelConfig

@pytest.fixture
def model_cfg():
    return ModelConfig(in_channels=3,
                       num_classes=4,
                       base_channels=8,
                       depth=3)

#create fake input tensor compatible with UNet
@pytest.fixture
def data(model_cfg):
    batch_size = 2
    in_channels = model_cfg.in_channels
    height = 64
    width = 64
    x = torch.randn(batch_size, in_channels, height, width)
    return x

@pytest.fixture
def model(model_cfg):
    return UNet(in_channels=model_cfg.in_channels,
                num_classes=model_cfg.num_classes,
                base_channels=model_cfg.base_channels,
                depth=model_cfg.depth)
    
def test_unet(model, data, model_cfg):   
    expected_output_shape = (data.size(0), model_cfg.num_classes, data.size(2), data.size(3))
    #forward pass
    output = model(data)
    #check output shape
    assert output.shape == expected_output_shape, f"Output shape {output.shape} does not match expected {expected_output_shape}"