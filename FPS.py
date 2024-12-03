import numpy as np
import torch
import time
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def computeTime(model1,model2, device='cuda'):
    inputs1 = torch.randn(1, 3, 384, 384)
    inputs2 = torch.randn(1, 1, 384, 384)
    if device == 'cuda':
        model1 = model1.cuda()
        model2 = model2.cuda()
        inputs1 = inputs1.cuda()
        inputs2 = inputs2.cuda()

    model1.eval()
    model2.eval()

    time_spent = []
    for idx in tqdm(range(1000)):
        start_time = time.time()
        with torch.no_grad():
            _ = model1(inputs1,inputs1,inputs1)
            _ = model2(inputs1,inputs1,inputs1,inputs2,inputs2)

        if device == 'cuda':
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if idx > 10:
            time_spent.append(time.time() - start_time)
    print('Average speed: {:.4f} fps'.format(1 / np.mean(time_spent)))


torch.backends.cudnn.benchmark = True

from QAnet import QAnet
from Tnet import Tnet

model1 = QAnet().cuda()
model2 = Tnet().cuda()

computeTime(model1,model2)
