def load_data(path,vocab_path,max_sample):
    map_dic,vocab_size=word_to_id(vocab_path)
    max_len=32
    with open(path,'r') as file:
        texts,labels=[],[]
        i=0
        while True:
            line=file.readline()
            if i>=max_sample:
                break
            if line:
                line=line.strip().split('\t')
                text=[]
                for w in line[0]:
                    if w in map_dic:
                        text.append(map_dic[w])
                    else:
                        text.append(0)
                if len(text)<max_len:
                    text=text+[0]*(max_len-len(text))
                if len(text)>max_len:
                    text=text[:max_len]
                label=[0,1] if line[1]=='michat' else [1,0]
                texts.append(text)
                labels.append(label)
            else:
                break
            i+=1
            # print('\r{}'.format(i),end='',flush=True)
        return texts,labels,vocab_size
    
def data_generate(text,label,batch_size,epochs):
    steps=len(label)//batch_size
    if len(label)%batch_size!=0:
        steps+=1
    print("data batch step:",steps)
    for epoch in range(epochs):
        for step in range(steps):
            start=step*batch_size
            end=min((step+1)*batch_size,len(label))
            yield text[start:end],label[start:end]

def word_to_id(path):
    map_dic={}
    with open(path,'r') as vfile:
        idx=0
        for line in vfile.readlines():
            line=line.strip()
            if line not in map_dic:
                map_dic[line]=idx
            idx+=1
    return map_dic,idx