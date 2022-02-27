#多头自注意力
#FNN
#add+ln
def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
def encode_mask(x,pad):
    x_mask=(x==pad).unsqueeze(-2)#[batch_size,seq_len]-->[batch_size,1,seq_len]
    return x_mask
    
def attention(q,k,v,scale=True,mask=None,dropout=None):
    qkt=torch.matmul(q,k.transpose(1,2))
    if scale:
        qkt/=math.sqrt(q.size(-1))
    if mask:
        qkt=qkt.masked_fill(mask=mask,value=torch.tensor(-1e9))
    attn=F.softmax(qkt,dim=-1)
    if dropout:
        attn=dropout(attn)
    return torch.matmul(attn,v)
    
class MutilHeadAttention(nn.Module):
    def __init__(self,h,d_model,dropout):
        super(MutilHeadAttention,self).__init__()
        assert d_model%h==0
        self.d_k=d_model//h
        self.h=h
        self.linear=clones(nn.linear(d_model,d_model),3)
        self.dropout=nn.Dropout(p=dropout)
        
    def forward(self,x,mask=None,scale=True):
        if mask:
            mask=encode(mask,pad=0).unsqueeze(1)#[batch_size,1,1,len(x)]
        bs,seq_len=x.shape[0],x.shape[1]
        q,k,v=[l(x).view(bs,self.h,seq_len,self.d_k) for l in self.linear]
        x=attention(q,k,v,scale,mask,self.dropout)
        x=x.transpose(1,2).contiguous().view(bs,seq_len,self.h*self.d)
        #.contiguous()：改变tensor的size后使用，可以保证得到的是resize后的tensor
        return x
        
class LayerNorm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.features=features
        self.eps=eps
        self.a=nn.Parameter(torch.ones(features))
        self.b=nn.Parameter(torch.zeros(features))
        
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a*(x-mean)/(std+self.eps)+self.b
        
class PositionwiseFeedFoward(nn.Module):
    def __init__(self,d_model,d_ff,dropout):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super(SublayerConnection,self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x,sublayer):
        return self.norm(x+self.dropout(sublayer(x)))#残差连接

class EncoderLayer(nn.module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer,self).__init__()
        self.attn=self_attn
        self.ffn=feed_forward
        self.sublayer=clone(SublayerConnection(size,dropout),2)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        x=self.sublayer[0](x,self.attn)
        return self.sublayer[1](x,self.fnn)

class Transformer(nn.Module):
    def __init__(self,unit,layer_nums,head_nums,d_model,dropout):
        super(Transformer,self).__init__()
        self.l=layer_nums
        self.h=head_nums
        self.d=d_model
        self.dropout=dropout
        self.linear=nn.Linear(d_model,unit)
        
    def forward(self,x):
        mha=MutilHeadAttention(self.h,self.d,self,doprout)
        fnn=PositionwiseFeedForward(self.d_model,self.d_model*2,dropout)
        for _ in range(len(self.l)):
            x=EncoderLayer(d_model,mha,ffn,self.dropout)(x)
        return F.softmax(self.linear(nn.Dropout(self.dropout)(x)))