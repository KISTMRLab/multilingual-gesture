import math
import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self,input_dim,latent=10,width=150):
        super().__init__(); self.proj=nn.Linear(input_dim,width)
        layer=nn.TransformerEncoderLayer(width,5,512,batch_first=True,activation="gelu"); self.net=nn.TransformerEncoder(layer,3,enable_nested_tensor=False)
        pos=torch.arange(256).float().unsqueeze(1); div=torch.exp(torch.arange(0,width,2).float()*(-math.log(10000.)/width)); pe=torch.zeros(256,width); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); self.register_buffer("pe",pe)
        self.out=nn.Sequential(nn.Linear(width,latent),nn.BatchNorm1d(latent),nn.LeakyReLU(.1))
    def forward(self,x):
        h=self.net(self.proj(x)+self.pe[:x.shape[1]]); return F.normalize(self.out(h.mean(1)),dim=-1)

class GestureCLR(nn.Module):
    def __init__(self,d2,d3): super().__init__(); self.pose2d=Encoder(d2); self.motion3d=Encoder(d3)
    def forward(self,a,b): return self.pose2d(a),self.motion3d(b)

def ntxent(a,b,t=.07):
    logits=a@b.T/t; labels=torch.arange(len(a),device=a.device)
    return (F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
