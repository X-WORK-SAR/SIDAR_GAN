import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


class Upsample(nn.Module):
    """Applies convolution followed by upsampling."""
    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()

        if scale_factor == 2:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 2, 2, 0, bias=True)  
        elif scale_factor == 4:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 4, 4, 0, bias=True)  
 
    def forward(self, x):
        return self.cv1(x)


##############################################################################################

class DDSA_cat(nn.Module):
    def __init__(self, in_channels, out_channels, M=2, r=16, L=32):

        super(DDSA_cat, self).__init__()
        d = max(in_channels // r, L)  
        self.M = M
        self.out_channels = out_channels
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1) 

        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))  

        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)   
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, input1, input2):    
        batch_size = input1.size(0)  
        h1, w1 = input1.shape[2:]  
        h2, w2 = input2.shape[2:]  
    

        if (h1, w1) != (h2, w2):
            input2 = F.interpolate(input2, size=(h1, w1), mode='bilinear', align_corners=False)
   
        output_fused = []
        output_fused.append(input1)                                  
        output_fused.append(input2)
        
   
        U = reduce(lambda x, y: x + y, output_fused)                 
        s = self.global_pool(U)                                      
        z = self.fc1(s)                                             
        a_b = self.fc2(z)                                            
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1) 
        a_b = self.softmax(a_b)                                      
        a_b = list(a_b.chunk(self.M, dim=1))                          

        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),       
                      a_b))      
        
        
        
        V = list(map(lambda x, y: x * y, output_fused, a_b))  
        V = reduce(lambda x, y: x + y, V)  
        return V
    
    
########################################################################################################################################

class DDSA(nn.Module):
    
 
    def __init__(self, c1, level=0):
        super().__init__()

        c1, c2 = c1[0], c1[1]
        self.level = level
        self.dim = c1, c2
        self.inter_dim = self.dim[self.level]    
        compress_c = 8
       

        if level == 0:
            self.stride_level_1 = Upsample(c2, self.inter_dim)
        if level == 1:
            self.stride_level_0 = nn.Conv2d(c1, self.inter_dim, 2, 2, 0)  

        self.weight_level_0 = nn.Conv2d(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = nn.Conv2d(self.inter_dim, compress_c, 1, 1)
        

        self.weights_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Conv2d(self.inter_dim, self.inter_dim, 3, 1, padding=1)

        self.DDSA_cat = DDSA_cat(8,8)
        self.DDSA_catcon = nn.Conv2d(8, 16, 1, 1)
 
    def forward(self, x):

        x0, x1 = x[0], x[1]
 

        if self.level == 0:
            level_0_resized = x0

            level_1_resized = x1

        elif self.level == 1:
            level_0_resized = self.stride_level_0(x0)
            level_1_resized = x1
 

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)


        levels_weight_v=self.DDSA_cat(level_0_weight_v,level_1_weight_v)
        levels_weight_v=self.DDSA_catcon(levels_weight_v)         
        levels_weight = self.weights_levels(levels_weight_v)    


        levels_weight = F.softmax(levels_weight, dim=1)      
  

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1] + level_1_resized * levels_weight[:, 1:2]
        return self.conv(fused_out_reduced)
    
