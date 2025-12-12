
from.unet_parts import *
from.repvit import *
import timm
from networks.SIDAR_GAN.FSDP import FSDP
from networks.SIDAR_GAN.DDSA import DDSA
from networks.SIDAR_GAN.SI_GAN import SI_GAN





class DimensionMatchingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DimensionMatchingLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x = self.conv(x)
        return x 







class SIDAR_GAN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SIDAR_GAN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        

        
        self.dim_match_layer1= DimensionMatchingLayer(96, 64)        
        self.dim_match_layer2 = DimensionMatchingLayer(96, 128)
        self.dim_match_layer3= DimensionMatchingLayer(192, 256)
        self.dim_match_layer4= DimensionMatchingLayer(384, 512)
        self.dim_match_layer5= DimensionMatchingLayer(768, 1024)  

        self.FSDP1 = (FSDP(64, 64))
        self.FSDP2 = (FSDP(128, 128))
        self.FSDP3 = (FSDP(256, 256))
        self.FSDP4 = (FSDP(512, 512))
        self.FSDP5 = (FSDP(1024, 1024))



        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

        
        
        self.maxvit = timm.create_model(
            'maxxvit_rmlp_small_rw_256.sw_in1k',
            pretrained=False,          
            features_only=True,        
        )
   
        weight_path = '/home/xzh4080/2D框架/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth'
        state_dict = torch.load(weight_path)

 
        new_state_dict = {}
        for key, value in state_dict.items():

            new_key = key.replace('.', '_', 1)
            

            if new_key in ["stem_conv1.weight", "stem_norm1.weight", "stem_norm1.bias", "stem_conv2.weight"]:
                new_key = new_key.replace('_', '.', 1)
            
            new_state_dict[new_key] = value

  
        model_state_dict = self.maxvit.state_dict()
        extra_keys = set(new_state_dict.keys()) - set(model_state_dict.keys())
        for key in extra_keys:
            del new_state_dict[key]


        self.maxvit.load_state_dict(new_state_dict)
        #########################################################

        self.SI_GAN = SI_GAN(n_channels, 3)
        unet_weight_path = '/home/xzh4080/2D框架/networks/add_model/权重/gan/400_net_G.pth'
        self.SI_GAN.load_state_dict(torch.load(unet_weight_path))
        
        
        #########################################################
        self.DDSA1 = DDSA([64,64],0)
        self.DDSA2 = DDSA([128,128],0)
        self.DDSA3 = DDSA([256,256],0)
        self.DDSA4 = DDSA([512,512],0)
        self.DDSA5 = DDSA([1024,1024],0)

    def forward(self, x):
        w,h=x.shape[2], x.shape[3]
        
        
        g1,g2,g3,g4,g5 = self.SI_GAN(x)
        g1 = F.interpolate(g1, scale_factor=0.5, mode='bilinear', align_corners=False)
        g2 = F.interpolate(g2, scale_factor=0.5, mode='bilinear', align_corners=False)
        g3 = F.interpolate(g3, scale_factor=0.5, mode='bilinear', align_corners=False)
        g4 = F.interpolate(g4, scale_factor=0.5, mode='bilinear', align_corners=False)
        g5 = F.interpolate(g5, scale_factor=0.5, mode='bilinear', align_corners=False)
        maxvit_features = self.maxvit(x)

        e1, e2, e3, e4, e5 = maxvit_features[:5] 

        e1=self.dim_match_layer1(e1)    
        e1=self.FSDP1(e1)
        e1=self.DDSA1([e1,g1])    

        e2=self.dim_match_layer2(e2)
        e2=self.FSDP2(e2)     
        e2=self.DDSA2([e2,g2])

        e3=self.dim_match_layer3(e3)
        e3=self.FSDP3(e3)
        e3=self.DDSA3([e3,g3])

        e4=self.dim_match_layer4(e4)
        e4=self.FSDP4(e4)     
        e4=self.DDSA4([e4,g4])

        e5=self.dim_match_layer5(e5)
        e5=self.FSDP5(e5)
        e5=self.DDSA5([e5,g5])  
        
        
        x = self.up1(e5, e4)
        x = self.up2(x, e3)
        x = self.up3(x, e2)
        x = self.up4(x, e1)
        logits = self.outc(x)
        logits = F.interpolate(logits, size=(w,h), mode='bilinear', align_corners=False)
        return logits
    



