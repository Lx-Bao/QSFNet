import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter, Softmax
from lib.swin_transformer import SwinTransformer
from lib.IIA_module import IIA
import numpy as np

'''
Third Stage   
Tnet = Initial Feature Extraction Subnet +  Region-guided Selective Fusion Subnet
'''


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=8, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x




class Tnet(nn.Module):
    def __init__(self):
        super(Tnet, self).__init__()
        #Mnet -- Initial Feature Extraction Subnet
        self.swin1 = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.MSD1_3v = MSD(512, 512, not_top=False)
        self.MSD1_2v = MSD(256, 256)
        self.MSD1_1v = MSD(128, 128)

        self.MSD2_2v = MSD(256, 256, not_top=False)
        self.MSD2_1v = MSD(128, 128)

        self.MSD3_1v = MSD(128, 128, not_top=False)

        self.final_3v = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_2v = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_1v = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.MSD1_3d = MSD(512, 512, not_top=False)
        self.MSD1_2d = MSD(256, 256)
        self.MSD1_1d = MSD(128, 128)

        self.MSD2_2d = MSD(256, 256, not_top=False)
        self.MSD2_1d = MSD(128, 128)

        self.MSD3_1d = MSD(128, 128, not_top=False)

        self.final_3d = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_2d = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_1d = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.MSD1_3t = MSD(512, 512, not_top=False)
        self.MSD1_2t = MSD(256, 256)
        self.MSD1_1t = MSD(128, 128)

        self.MSD2_2t = MSD(256, 256, not_top=False)
        self.MSD2_1t = MSD(128, 128)

        self.MSD3_1t = MSD(128, 128, not_top=False)

        self.final_3t = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_2t = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_1t = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.pool4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.pool8 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.pool16 = nn.MaxPool2d(16, 16, ceil_mode=True)


        #Region-guided Selective Fusion Subnet
        self.trans4_d = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.trans3_d = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.trans2_d = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


        self.trans4_t = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.trans3_t = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.trans2_t = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )


 
        self.fuse2_d = IIA(num_blocks=2, x_channels=128, nx=9216, y_channels=128, ny=9216)
        self.fuse3_d = IIA(num_blocks=2, x_channels=128, nx=9216, y_channels=128, ny=9216)
        self.fuse4_d = IIA(num_blocks=2, x_channels=128, nx=9216, y_channels=128, ny=9216)


        self.fuse2_t = IIA(num_blocks=2, x_channels=128, nx=9216, y_channels=128, ny=9216)
        self.fuse3_t = IIA(num_blocks=2, x_channels=128, nx=9216, y_channels=128, ny=9216)
        self.fuse4_t = IIA(num_blocks=2, x_channels=128, nx=9216, y_channels=128, ny=9216)

        self.conv1_out = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.ER = ER(128)

        self.CBAM4 = CBAMLayer(128)
        self.CBAM3 = CBAMLayer(128)
        self.CBAM2 = CBAMLayer(128)
        self.CBAM1 = CBAMLayer(128)

        self.final_4_dt = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_3_dt = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_2_dt = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_1_dt = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )

        self.final_edge = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, 1, 1, bn=False, relu=False)
        )


    def forward(self, rgb, t, d,score1_d, score1_t):       
        #Initial Feature Extraction Subnet
        score_list_t, score_PE = self.swin1(t)
        score_list_rgb, score_PE = self.swin1(rgb)
        score_list_d, score_PE = self.swin1(d)

        x1_v = score_list_rgb[0]
        x2_v = score_list_rgb[1]
        x3_v = score_list_rgb[2]
        x4_v = score_list_rgb[3]

        x1_3v = self.MSD1_3v(x3_v, x4_v, x4_v)
        x1_2v = self.MSD1_2v(x2_v, x3_v, x1_3v)
        x1_1v = self.MSD1_1v(x1_v, x2_v, x1_2v)

        x2_2v = self.MSD2_2v(x1_2v, x1_3v, x1_3v)
        x2_1v = self.MSD2_1v(x1_1v, x1_2v, x2_2v)

        x3_1v = self.MSD3_1v(x2_1v, x2_2v, x2_2v)

        x1_1v_pred = self.final_1v(x1_1v)
        x2_1v_pred = self.final_2v(x2_1v)
        x3_1v_pred = self.final_3v(x3_1v)

        x1e_pred = self.up4(x1_1v_pred)
        x2e_pred = self.up4(x2_1v_pred)
        x3e_pred = self.up4(x3_1v_pred)

        x1_t = score_list_t[0]
        x2_t = score_list_t[1]
        x3_t = score_list_t[2]
        x4_t = score_list_t[3]

        x1_3t = self.MSD1_3t(x3_t, x4_t, x4_t)
        x1_2t = self.MSD1_2t(x2_t, x3_t, x1_3t)
        x1_1t = self.MSD1_1t(x1_t, x2_t, x1_2t)

        x2_2t = self.MSD2_2t(x1_2t, x1_3t, x1_3t)
        x2_1t = self.MSD2_1t(x1_1t, x1_2t, x2_2t)

        x3_1t = self.MSD3_1t(x2_1t, x2_2t, x2_2t)

        x1_1t_pred = self.final_1t(x1_1t)
        x2_1t_pred = self.final_2t(x2_1t)
        x3_1t_pred = self.final_3t(x3_1t)

        x1e_pred_t = self.up4(x1_1t_pred)
        x2e_pred_t = self.up4(x2_1t_pred)
        x3e_pred_t = self.up4(x3_1t_pred)

        x1_d = score_list_d[0]
        x2_d = score_list_d[1]
        x3_d = score_list_d[2]
        x4_d = score_list_d[3]

        x1_3d = self.MSD1_3d(x3_d, x4_d, x4_d)
        x1_2d = self.MSD1_2d(x2_d, x3_d, x1_3d)
        x1_1d = self.MSD1_1d(x1_d, x2_d, x1_2d)

        x2_2d = self.MSD2_2d(x1_2d, x1_3d, x1_3d)
        x2_1d = self.MSD2_1d(x1_1d, x1_2d, x2_2d)

        x3_1d = self.MSD3_1d(x2_1d, x2_2d, x2_2d)

        x1_1d_pred = self.final_1d(x1_1d)
        x2_1d_pred = self.final_2d(x2_1d)
        x3_1d_pred = self.final_3d(x3_1d)

        x1e_pred_d = self.up4(x1_1d_pred)
        x2e_pred_d = self.up4(x2_1d_pred)
        x3e_pred_d = self.up4(x3_1d_pred)



###################################################################################################
        #Region-guided Selective Fusion Subnet

        score_d = self.pool4(score1_d)
        score_t = self.pool4(score1_t)
        
        x4_d = (1 - score_d) * x1_1v + score_d * (x1_1d - ((x1_1v + x1_1t)/2))
        x3_d = (1 - score_d) * x2_1v + score_d * (x2_1d - ((x2_1v + x2_1t)/2))
        x2_d = (1 - score_d) * x3_1v + score_d * (x3_1d - ((x3_1v + x3_1t)/2))


        x4_t = (1 - score_t) * x1_1v + score_t * (x1_1t - ((x1_1d + x1_1v)/2))
        x3_t = (1 - score_t) * x2_1v + score_t * (x2_1t - ((x2_1d + x2_1v)/2))
        x2_t = (1 - score_t) * x3_1v + score_t * (x3_1t - ((x3_1d + x3_1v)/2))

        x4_d = self.trans4_d(x4_d).flatten(2).permute(0, 2, 1)
        x3_d = self.trans3_d(x3_d).flatten(2).permute(0, 2, 1)
        x2_d = self.trans2_d(x2_d).flatten(2).permute(0, 2, 1)


        x4_t = self.trans4_t(x4_t).flatten(2).permute(0, 2, 1)
        x3_t = self.trans3_t(x3_t).flatten(2).permute(0, 2, 1)
        x2_t = self.trans2_t(x2_t).flatten(2).permute(0, 2, 1)


        x4f_d = self.fuse4_d(x4_d, x4_t)
        x3f_d = self.fuse3_d(x3_d, x3_t)
        x2f_d = self.fuse2_d(x2_d, x2_t)

        x4f_t = self.fuse4_t(x4_t, x4_d)
        x3f_t = self.fuse3_t(x3_t, x3_d)
        x2f_t = self.fuse2_t(x2_t, x2_d)


        B, N, C = x2f_t.shape
        x2f_t = x2f_t.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))  # (B,C,N) --> (B,C,H,W) 
        B, N, C = x3f_t.shape
        x3f_t = x3f_t.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
        B, N, C = x4f_t.shape
        x4f_t = x4f_t.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))


        B, N, C = x2f_d.shape
        x2f_d = x2f_d.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
        B, N, C = x3f_d.shape
        x3f_d = x3f_d.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))
        B, N, C = x4f_d.shape
        x4f_d = x4f_d.transpose(1, 2).reshape(B, C, int(np.sqrt(N)), int(np.sqrt(N)))

        x4_dt = x4f_d + x4f_t
        x4e_dt = self.CBAM4(x4_dt)
        x4e_pred_dt = self.final_4_dt(x4e_dt)

        x3_dt = x3f_d + x3f_t + x4e_dt
        x3e_dt = self.CBAM3(x3_dt)
        x3e_pred_dt = self.final_3_dt(x3e_dt)

        x2_dt = x2f_d + x2f_t + x3e_dt
        x2e_dt = self.CBAM2(x2_dt)
        x2e_pred_dt = self.final_2_dt(x2e_dt)

        #ER module
        x1_dt = self.up2(x2e_dt)
        x1_dt = self.conv1_out(x1_dt)
        x1_dt, edge = self.ER(x1_dt)
        x1e_dt = self.CBAM1(x1_dt)
        x1e_pred_dt = self.final_1_dt(x1e_dt)

        edge_pred = torch.sigmoid(self.final_edge(edge))
        edge_pred = self.up2(edge_pred)

        x1e_pred_dt = self.up2(x1e_pred_dt)
        x2e_pred_dt = self.up4(x2e_pred_dt)
        x3e_pred_dt = self.up4(x3e_pred_dt)
        x4e_pred_dt = self.up4(x4e_pred_dt)

        return edge_pred,x3e_pred, x2e_pred, x1e_pred, x3e_pred_t, x2e_pred_t, x1e_pred_t, x3e_pred_d, x2e_pred_d, x1e_pred_d, x4e_pred_dt, x3e_pred_dt, x2e_pred_dt, x1e_pred_dt



class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class DSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=0,dilation=1, relu=True):
        super(DSConv, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=kernel_size, s=stride, p=padding, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class MSD(nn.Module):
    def __init__(self, in_channel, out_channel, not_top=True):
        super(MSD, self).__init__()
        self.not_top = not_top
        self.conv1x1 = Conv(in_channel, out_channel, 1, 1)
        self.conv_Fn = Conv(in_channel*2, out_channel, 1, 1)
        self.DSConv3x3 = DSConv(in_channel, out_channel, kernel_size=3,stride=1, padding=1)
        self.DSConv5x5 = DSConv(in_channel, out_channel, kernel_size=5,stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3x3 = Conv(out_channel, out_channel, 3, 1)
        if not_top:
            self.conv_Fp = Conv(in_channel*2, out_channel, 1, 1)

    def forward(self, F,F_n,F_p):
        F_n = self.up2(self.conv_Fn(F_n))
        F_f = F + F_n
        F_f1 = self.conv1x1(F_f)
        F_f2 = self.DSConv3x3(F_f)
        F_s1 = F_f1 * F_f2
        F_f3 = self.DSConv5x5(F_f)
        F_s2 = F_s1 * F_f3
        if self.not_top:
            F_p = self.up2(self.conv_Fp(F_p))
            F_sout = self.conv3x3(F_s2 + F_p)
        else:
            F_sout = self.conv3x3(F_s2)

        return F_sout


class ER(nn.Module):
    def __init__(self, in_channel):
        super(ER, self).__init__()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge = x - self.avg_pool(x)  # Xi=X-Avgpool(X)
        edge = self.bn1(self.conv_1(edge))
        weight = self.sigmoid(edge)
        out = weight * x + x
        return out, edge