import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter, Softmax
from lib.swin_transformer import SwinTransformer

'''
First Stage  
Mnet = Initial Feature Extraction Subnet
'''

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        self.swin1 = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.MSD1_3v = MSD(512, 512,not_top=False)
        self.MSD1_2v = MSD(256, 256)
        self.MSD1_1v = MSD(128, 128)

        self.MSD2_2v = MSD(256, 256,not_top=False)
        self.MSD2_1v = MSD(128, 128)

        self.MSD3_1v = MSD(128, 128,not_top=False)



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

    def forward(self, rgb, t, d):
        score_list_t, score_PE = self.swin1(t)
        score_list_rgb, score_PE = self.swin1(rgb)
        score_list_d, score_PE = self.swin1(d)

        #V Branch
        x1_v = score_list_rgb[0]
        x2_v = score_list_rgb[1]
        x3_v = score_list_rgb[2]
        x4_v = score_list_rgb[3]

        #First column of MSF modules
        x1_3v = self.MSD1_3v(x3_v,x4_v,x4_v)
        x1_2v = self.MSD1_2v(x2_v,x3_v,x1_3v)
        x1_1v = self.MSD1_1v(x1_v,x2_v,x1_2v)
        #Second column of MSF modules
        x2_2v = self.MSD2_2v(x1_2v,x1_3v,x1_3v)
        x2_1v = self.MSD2_1v(x1_1v,x1_2v,x2_2v)
        #Third column of MSF modules
        x3_1v = self.MSD3_1v(x2_1v,x2_2v,x2_2v)

        x1_1v_pred = self.final_1v(x1_1v)
        x2_1v_pred = self.final_2v(x2_1v)
        x3_1v_pred = self.final_3v(x3_1v)

        x1e_pred = self.up4(x1_1v_pred)
        x2e_pred = self.up4(x2_1v_pred)
        x3e_pred = self.up4(x3_1v_pred)

        #T branch
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

        #D branch
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


        #First step of generating PGT
        x1_vd = self.sigmoid((x1e_pred + x1e_pred_d)/2)
        PGTT_P = self.ReLU(self.sigmoid(x1e_pred_t) - x1_vd)
        PGTT_N = x1_vd * self.sigmoid(x1e_pred_t)

        x1_vt = self.sigmoid((x1e_pred + x1e_pred_t)/2)
        PGTD_P = self.ReLU(self.sigmoid(x1e_pred_d) - x1_vt)
        PGTD_N = x1_vt * self.sigmoid(x1e_pred_d)


        return x3e_pred, x2e_pred, x1e_pred, x3e_pred_t, x2e_pred_t, x1e_pred_t, x3e_pred_d, x2e_pred_d, x1e_pred_d,PGTD_P,PGTD_N,PGTT_P,PGTT_N



    def load_pretrained_model(self):
        self.swin1.load_state_dict(torch.load('./swin_base_patch4_window12_384_22k.pth')['model'],strict=False)
        print('loading pretrained model success!')


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