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

'''
First Stage
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    save_path = './model'
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.0001
    batch_size = 4
    epoch = 200
    num_params = 0
    data = Data(img_root)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers = 8)
    net = Mnet().cuda()
    net.load_pretrained_model()

    params = net.parameters()
    optimizer = torch.optim.Adam(params, lr, betas=(0.5, 0.999))

    for p in net.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))
    iter_num = len(loader)
    net.train()

    for epochi in tqdm(range(1, epoch + 1)):
        prefetcher = DataPrefetcher(loader)
        rgb, t, d, eg, label = prefetcher.next()
        r_sal_loss = 0
        epoch_ave_loss = 0
        i = 0
        while rgb is not None:
            i += 1
            score3, score2, score1,score3_d, score2_d, score1_d,score3_t, score2_t, score1_t,PGTD_P,PGTD_N,PGTT_P,PGTT_N = net(rgb, t, d)

            loss3 = structure_loss(score3, label)
            loss2 = structure_loss(score2, label)
            loss1 = structure_loss(score1, label)
            
            loss6 = structure_loss(score3_d, label)
            loss5 = structure_loss(score2_d, label)
            loss4 = structure_loss(score1_d, label)
            
            loss9 = structure_loss(score3_t, label)
            loss8 = structure_loss(score2_t, label)
            loss7 = structure_loss(score1_t, label)

            sal_loss = loss1 + loss2 + loss3 + loss4 +loss5 + loss6 +loss7 +loss8+ loss9
            r_sal_loss += sal_loss.data
            sal_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0:
                print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  loss : %5.4f, lr: %7.6f' % (
                    epochi, epoch, i, iter_num, r_sal_loss / 100, lr,))
                epoch_ave_loss += (r_sal_loss / 100)
                r_sal_loss = 0
            rgb, t, d, eg, label = prefetcher.next()
        print('epoch-%2d_ave_loss: %7.6f' % (epochi, (epoch_ave_loss / (10.5 / batch_size))))
        if epochi % 20 == 0:
            model_path = '%s/epoch_%d.pth' % (save_path, epochi)
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))

    torch.save(net.state_dict(), '%s/final.pth' % (save_path))