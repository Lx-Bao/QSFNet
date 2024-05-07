from QAnet import QAnet
from Tnet import Tnet
from thop import profile
import torch

model1 = QAnet().cuda()
model2 = Tnet().cuda()
input = torch.randn(1, 3, 384, 384).cuda()
input2 = torch.randn(1, 3, 384, 384).cuda()
input3 = torch.randn(1, 1, 384, 384).cuda()
flops1, params1 = profile(model1, inputs=(input,input2,input2 ))
flops2, params2 = profile(model2, inputs=(input,input2,input2, input3, input3 ))
print('params:%.2f(M)'%((params1+params2)/1000000))
print('flops:%.2f(G)'%((flops1+flops2)/1000000000))