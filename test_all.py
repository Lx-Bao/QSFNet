import os
from torch.utils.data import DataLoader
from lib.dataset import Data
import torch.nn.functional as F
import torch
import cv2
from Tnet import Tnet
import numpy as np
from QAnet import QAnet
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    model_path = './modelT/final.pth'
    out_path = './output_out/'
    data = Data(root='./VDT-2048 dataset/Test/', mode='test')
    loader = DataLoader(data, batch_size=1, shuffle=False)
    qnet = QAnet().cuda()
    qnet.load_state_dict(torch.load('./modelQA/final.pth'))
    tnet = Tnet().cuda()
    print('loading model from %s...' % model_path)
    tnet.load_state_dict(torch.load(model_path))
    if not os.path.exists(out_path): os.mkdir(out_path)
    img_num = len(loader)
    qnet.eval()
    tnet.eval()
    with torch.no_grad():
        for rgb, t, d,_, _, (H, W), name in loader:
            score1_d, score1_t = qnet(rgb.cuda().float(), t.cuda().float(), d.cuda().float())
            score_eg,score3, score2, score1, score3_t, score2_t, score1_t, score3_d, score2_d, score1_d, score4_out,score3_out, score2_out, score1_out = tnet(rgb.cuda().float(), t.cuda().float(), d.cuda().float(), score1_d, score1_t)
            
            score = F.interpolate(score1_out, size=(H, W), mode='bilinear',align_corners=True)
            pred = np.squeeze(torch.sigmoid(score).cpu().data.numpy())
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '.png'), 255 * pred)
            




