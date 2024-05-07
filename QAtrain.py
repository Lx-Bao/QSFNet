coding = 'utf-8'
import os
from Mnet import Mnet
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

'''
Second Stage 
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
IOU = pytorch_iou.IOU(size_average = True)

def bce_loss(pred,target):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')

    return bce


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
    save_path_QA = './modelQA'
    if not os.path.exists(save_path_QA): os.mkdir(save_path_QA)
    lr = 0.0001
    batch_size = 4
    epoch = 200
    num_params = 0
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers = 8)
    net = Mnet().cuda()
    net.load_state_dict(torch.load('./model/final.pth'))
    qnet = QAnet().cuda()
    params = qnet.parameters()
    optimizer = torch.optim.Adam(params, lr, betas=(0.5, 0.999))
    for p in qnet.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))
    iter_num = len(loader)
    net.train()
    qnet.train()

    for epochi in tqdm(range(1, epoch + 1)):    

        prefetcher = DataPrefetcher(loader)
        rgb, t, d, eg, label = prefetcher.next()
        B, C, H, W = label.shape
        r_QA_loss = 0
        epoch_ave_loss = 0
        net.zero_grad()
        qnet.zero_grad()
        i = 0
        while rgb is not None:
            i += 1
            with torch.no_grad():
                x3e_pred, x2e_pred, x1e_pred, x3e_pred_t, x2e_pred_t, x1e_pred_t, x3e_pred_d, x2e_pred_d, x1e_pred_d,PGTD_P,PGTD_N,PGTT_P,PGTT_N = net(
                    rgb, t, d)

            x0e_pred_vdt_D, x0e_pred_vdt_T = qnet(rgb, t, d)

            #Computing PGTs of D and T branch
            PGTD = PGTD_P * label + PGTD_N * (1 - label)
            PGTT = PGTT_P * label + PGTT_N * (1 - label)

            #Supervised by PGT
            loss2 = bce_loss(x0e_pred_vdt_D, PGTD)
            loss1 = bce_loss(x0e_pred_vdt_T, PGTT)

            QA_loss = loss1 + loss2
            r_QA_loss += QA_loss.data
            QA_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f, lr: %7.6f' % (
                    epochi, epoch, i, iter_num, r_QA_loss / 100, lr,))
                epoch_ave_loss += (r_QA_loss / 100)
                r_QA_loss = 0
            rgb, t, d,eg, label = prefetcher.next()
        print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_ave_loss / (10.5 / batch_size))))
        if epochi % 20 == 0:
            model_path = '%s/epoch_QA_%d.pth' % (save_path_QA, epochi)
            torch.save(qnet.state_dict(), '%s/epoch_QA_%d.pth' % (save_path_QA, epochi))


    torch.save(qnet.state_dict(), '%s/final.pth' % (save_path_QA))