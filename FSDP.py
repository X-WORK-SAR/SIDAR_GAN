from typing import Optional
from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
from networks.SIDAR_GAN.ASP import ASP
from networks.SIDAR_GAN.SPD import SPD



    
def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class FSDP(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion: float = 0.5,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
    ):

        super(FSDP, self).__init__()

        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        

        self.conv1 = ConvModule(in_channels, 2 * hidden_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(2 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        
        self.SPD = (SPD(hidden_channels, hidden_channels))


        self.ASP1 = (ASP(hidden_channels, hidden_channels))
        self.ASP2 = (ASP(hidden_channels, hidden_channels))
        
    def forward(self, x):
        
        x, y = list(self.conv1(x).chunk(2, 1))

        x = self.SPD(x)

        z = [x]
        t = torch.zeros(y.shape, device=y.device)  
        
        t = t + self.ASP2(self.ASP1(y))  

        z.append(t)          
        z = torch.cat(z, dim=1)    
        z = self.conv2(z)
        z = self.conv3(z)

        return z





