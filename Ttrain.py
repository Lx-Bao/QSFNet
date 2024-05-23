coding = 'utf-8'
import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.dataset import Data
from lib.data_prefetcher import DataPrefetcher
from torch.nn import functional as F
import pytorch_iou
from torch import nn
from QAnet import QAnet
from Tnet import Tnet

'''
Third Stage 
'''

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
IOU = pytorch_iou.IOU(size_average = True)

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


if __name__ == '__main__':

    # dataset
    img_root = './VDT-2048 dataset/Train/'
    out_path_PGT1 = './output_PGT1/'
    save_path_T = './modelT'
    if not os.path.exists(save_path_T): os.mkdir(save_path_T)
    lr = 0.0001
    batch_size = 4
    epoch = 200
    num_params = 0
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers = 8)
    net = Tnet().cuda()
    net.load_state_dict(torch.load('./model/final.pth'),strict=False)
    qnet = QAnet().cuda()
    qnet.load_state_dict(torch.load('./modelQA/final.pth'))

    params = net.parameters()
    optimizer = torch.optim.Adam(params, lr, betas=(0.5, 0.999))
    for p in net.parameters():
        num_params += p.numel()
    print("The number of parameters of Tnet: {}".format(num_params))
    iter_num = len(loader)
    net.train()
    qnet.train()

    for epochi in tqdm(range(1, epoch + 1)):
        prefetcher = DataPrefetcher(loader)
        rgb, t, d, eg, label = prefetcher.next()
        r_T_loss = 0
        epoch_ave_loss = 0
        i = 0
        while rgb is not None:
            i += 1
            with torch.no_grad():
                score1_d, score1_t = qnet(rgb, t, d)
            score_eg, score3, score2, score1, score3_t, score2_t, score1_t, score3_d, score2_d, score1_d, score4_out, score3_out, score2_out, score1_out = net(
                rgb, t, d, score1_d, score1_t)

            losseg_out = bce_loss(score_eg, eg)

            loss4_out = structure_loss(score4_out, label)
            loss3_out = structure_loss(score3_out, label)
            loss2_out = structure_loss(score2_out, label)
            loss1_out = structure_loss(score1_out, label)


            loss3 = structure_loss(score3, label)
            loss2 = structure_loss(score2, label)
            loss1 = structure_loss(score1, label)

            loss3_t = structure_loss(score3_t, label)
            loss2_t = structure_loss(score2_t, label)
            loss1_t = structure_loss(score1_t, label)

            loss3_d = structure_loss(score3_d, label)
            loss2_d = structure_loss(score2_d, label)
            loss1_d = structure_loss(score1_d, label)

            T_loss = losseg_out + loss1 + loss2 + loss3 + loss1_t + loss2_t + loss3_t + loss1_d + loss2_d + loss3_d + loss1_out*2 + loss2_out + loss3_out + loss4_out
            #T_loss = losseg_out + loss1 + loss2 + loss3 + loss1_t + loss2_t + loss3_t + loss1_d + loss2_d + loss3_d + loss1_out + loss2_out + loss3_out + loss4_out

            r_T_loss += T_loss.data
            T_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f, lr: %7.6f' % (
                    epochi, epoch, i, iter_num, r_T_loss / 100, lr,))
                epoch_ave_loss += (r_T_loss / 100)
                r_T_loss = 0
            rgb, t, d, eg, label = prefetcher.next()
        print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_ave_loss / (10.5 / batch_size))))
        if epochi % 10 == 0:
            model_path = '%s/epoch_T_%d.pth' % (save_path_T, epochi)
            torch.save(net.state_dict(), '%s/epoch_T_%d.pth' % (save_path_T, epochi))

    torch.save(net.state_dict(), '%s/final.pth' % (save_path_T))
