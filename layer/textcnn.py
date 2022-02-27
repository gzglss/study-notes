class TextCnn:
    '''
    units：类别数
    kernel_num：每个卷积核的数量
    '''
    def __init__(self,units,kernel_num,kernel_size,emb_size,n_vocab,p,cuda):
        super(TextCnn,self).__init__()
        self.embedding=nn.Embedding(n_vocab,emb_size)
        if cuda:
            self.convs=[nn.Conv1d(emb_size,kernel_num,i) for i in kernel_size]
        self.dropout=nn.Dropout(p)
        self.fc=nn.Linear(kernel_num*kernel_size,units)

    def cnn_pool(self,x,conv):
        x=F.relu(conv(x))
        x=F.max_pool1d(x,x.size).squeeze(2)
        return x
        
    def forward(self,text):
        out=self.embedding(text)#[batch_size,list_len,emb_size]
        out=torch.reshape(out,(out.shape[0],out.shape[2],out.shape[1]]]))#[bs,es,l]
        out=torch.cat([self.cnn_pool(out,conv) for conv in self.convs],1)#[bs,kernel_num*kernel_size]
        out=self.dropout(out)
        out=self.fc(out)
        return out