import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import REA

class REAM(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(REAM, self).__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
        self.relu = nn.PReLU()
        self.ca = REA(out_channels)

    def forward(self, x):
        res = self.ca(self.conv2(self.relu(self.conv1(x))))
        res += x
        return res

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

      
      
class DBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 n_feats,
                 kernel_size=3):
        super(DBlock, self).__init__()
        reduction = ms_channels//2      
        self.convF  = nn.Conv2d(in_channels=n_feats, out_channels=ms_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.convFT = nn.Conv2d(in_channels=ms_channels, out_channels=n_feats, kernel_size=3, stride=1, padding=1,bias=False)
        channel_input1  = n_feats
        channel_output1 = n_feats
        self.conv1 = Conv(channel_input1, channel_output1)
        self.down1 = Down()
        channel_input2  = channel_output1
        channel_output2 = channel_output1 // 2 
        self.conv2 = Conv(channel_input2, channel_output2)
        self.down2 = Down()
        channel_input3  = channel_output2
        channel_output3 = channel_input3 
        self.conv3 = Conv(channel_input3, channel_output3)
        self.up1 = UP(channel_output3, channel_output3, bicubic=False)
        channel_input4  = channel_output3
        channel_output4 = channel_output1 // 2
        self.conv4 = Conv(channel_input4, channel_output4)
        self.up2 = UP(channel_output4, channel_output1, bicubic=False)
        channel_input5  = channel_output1
        channel_output5 = n_feats
        self.conv5 = Conv(channel_input5, channel_output5)
        self.conv6 = OutConv(channel_output5, n_feats)
    def forward(self, HR, S, D):
        r = D - self.convFT(self.convF(D)-(HR-S))
        x1 = self.conv1(r)
        x2 = self.down1(x1)
        x2 = self.conv2(x2)
        x3 = self.down2(x2)
        x3 = self.conv3(x3)
        x4 = self.up1(x3)
        x4 = self.conv4(x4 + x2)
        x5 = self.up2(x4)
        x5 = self.conv5(x5 + x1)
        x6 = self.conv6(x5)
        D = x6 + r
        return D
      
class HBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 n_feats,
                 kernel_size=3):
        super(HBlock, self).__init__()      
        self.convF = nn.Conv2d(in_channels=n_feats, out_channels=ms_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.rho = nn.Parameter(
            nn.init.normal_(
                torch.empty(1).cuda(), mean=0.1, std=0.03
            ))
        self.prox = REAM(ms_channels, n_feats, ms_channels, kernel_size)
    def forward(self, HR, S, D):
        E = HR - S - self.convF(D)
        R = HR - self.rho * E
        HR = self.prox(R)
        return HR

      
class M3DNet(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feats,
                 n_layer):
        super(M3DNet, self).__init__()
        self.D_blocks = nn.ModuleList([DBlock(ms_channels, n_feats, 3) for i in range(n_layer)])
        self.H_blocks = nn.ModuleList([HBlock(ms_channels, n_feats, 3) for i in range(n_layer)])
        self.convert = nn.Conv2d(in_channels=ms_channels, out_channels=n_feats, kernel_size=3, stride=1, padding=1)
    def forward(self, ms, pan=None):
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        N, ms_channels, h, w = ms.shape
        N, pan_channels, H, W = pan.shape
        HR = upsample(ms, H, W)
        S  = upsample(ms, H, W)
        D = pan
        D = D.expand(-1, ms_channels, -1, -1)
        D = self.convert(D - HR)
        for i in range(len(self.D_blocks)):
            D  = self.D_blocks[i](HR, S, D)
            HR = self.H_blocks[i](HR, S, D)
            
        return HR
      
      
