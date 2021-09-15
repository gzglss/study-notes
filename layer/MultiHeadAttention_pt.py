import json
import os
import torch
import random
from torch import nn
from torch import optim
import pandas as pd
import copy
import math
import torch.nn.functional as F

def encode_mask(src,pad=0):
    src_mask=(src==pad).unsqueeze(-2)  #返回一个true/false矩阵，size = [batch , 1 , src_L]
    return src_mask
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  #检测word embedding维度是否能被h整除
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h  # 头的个数
        self.linears = clones(nn.Linear(d_model, d_model), 4) #四个线性变换，前三个为QKV三个变换矩阵，最后一个用于attention后
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) 
        nbatches = query.size(0) 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # view中给-1可以推测这个位置的维度
             for l, x in zip(self.linears, (query, key, value))]#[batch_size,seq_len,h,d]
        x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)
        
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask=mask, value=torch.tensor(-1e9))
    p_attn = F.softmax(scores, dim = -1) 
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn