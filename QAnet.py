import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Parameter, Softmax
from torchvision.models import resnet34 as resnet

'''
Second Stage   
QAnet = Quality-aware Region Selection Subnet
'''

class QAnet(nn.Module):
    def __init__(self):
        super(QAnet, self).__init__()
        self.resnet_D = resnet()
        self.resnet_T = resnet()
        self.ReLU = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

##################################################################################################################
        self.conv4_vdt_D = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv3_vdt_D = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv2_vdt_D = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv1_vdt_D = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv0_vdt_D = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_4_vdt_D = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_3_vdt_D = nn.Sequential(
            Conv(128, 64, 1, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_2_vdt_D = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1_vdt_D = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_0_vdt_D = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )



##########################################################################################################################
        self.conv4_vdt_T = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv3_vdt_T = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv2_vdt_T = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv1_vdt_T = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv0_vdt_T = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_4_vdt_T = nn.Sequential(
            Conv(256, 64, 1, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_3_vdt_T = nn.Sequential(
            Conv(128, 64, 1, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_2_vdt_T = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_1_vdt_T = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.final_0_vdt_T = nn.Sequential(
            Conv(64, 64, 3, bn=True, relu=True),
            Conv(64, 1, 3, bn=False, relu=False)
        )

        self.resconvV1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.resconvD1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.resconvT1 = nn.Conv2d(in_channels=9, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

    def forward(self, rgb, t, d):

        vdt = torch.cat((rgb, t, d), dim=1)
        #D Branch
        #Encoder
        x_u_D = self.resconvD1(vdt)
        x_u_D = self.resnet_D.bn1(x_u_D)
        x_u_D = self.resnet_D.relu(x_u_D)

        x0_vdt_D = self.resnet_D.maxpool(x_u_D)
        x1_vdt_D = self.resnet_D.layer1(x0_vdt_D)
        x2_vdt_D = self.resnet_D.layer2(x1_vdt_D)
        x3_vdt_D = self.resnet_D.layer3(x2_vdt_D)
        x4_vdt_D = self.resnet_D.layer4(x3_vdt_D)
        #Decoder
        x4_vdt_D = self.up2(x4_vdt_D)
        x4e_vdt_D = self.conv4_vdt_D(x4_vdt_D)

        x3e_vdt_D = self.conv3_vdt_D(torch.cat((x4e_vdt_D, x3_vdt_D), dim=1))
        x3e_vdt_D = self.up2(x3e_vdt_D)

        x2e_vdt_D = self.conv2_vdt_D(torch.cat((x3e_vdt_D, x2_vdt_D), dim=1))
        x2e_vdt_D = self.up2(x2e_vdt_D)

        x1e_vdt_D = self.conv1_vdt_D(torch.cat((x2e_vdt_D, x1_vdt_D), dim=1))
        x1e_vdt_D = self.up2(x1e_vdt_D)

        x0e_vdt_D = self.conv0_vdt_D(torch.cat((x1e_vdt_D, x_u_D), dim=1))
        x0e_vdt_D = self.up2(x0e_vdt_D)
        x0e_pred_vdt_D = self.final_0_vdt_D(x0e_vdt_D)

        #T Branch
        x_u_T = self.resconvT1(vdt)
        x_u_T = self.resnet_T.bn1(x_u_T)
        x_u_T = self.resnet_T.relu(x_u_T)

        x0_vdt_T = self.resnet_T.maxpool(x_u_T)
        x1_vdt_T = self.resnet_T.layer1(x0_vdt_T)
        x2_vdt_T = self.resnet_T.layer2(x1_vdt_T)
        x3_vdt_T = self.resnet_T.layer3(x2_vdt_T)
        x4_vdt_T = self.resnet_T.layer4(x3_vdt_T)

        x4_vdt_T = self.up2(x4_vdt_T)
        x4e_vdt_T = self.conv4_vdt_T(x4_vdt_T)

        x3e_vdt_T = self.conv3_vdt_T(torch.cat((x4e_vdt_T, x3_vdt_T), dim=1))
        x3e_vdt_T = self.up2(x3e_vdt_T)

        x2e_vdt_T = self.conv2_vdt_T(torch.cat((x3e_vdt_T, x2_vdt_T), dim=1))
        x2e_vdt_T = self.up2(x2e_vdt_T)

        x1e_vdt_T = self.conv1_vdt_T(torch.cat((x2e_vdt_T, x1_vdt_T), dim=1))
        x1e_vdt_T = self.up2(x1e_vdt_T)

        x0e_vdt_T = self.conv0_vdt_T(torch.cat((x1e_vdt_T, x_u_T), dim=1))
        x0e_vdt_T = self.up2(x0e_vdt_T)
        x0e_pred_vdt_T = self.final_0_vdt_T(x0e_vdt_T)


        return x0e_pred_vdt_D, x0e_pred_vdt_T

    def load_pretrained_model(self):
        self.resnet_D.load_state_dict(torch.load('./resnet34-333f7ec4.pth'))
        self.resnet_T.load_state_dict(torch.load('./resnet34-333f7ec4.pth'))
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