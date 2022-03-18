import jieba
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

stopword_path='../michat-data/stopword.txt'
train_path='../michat-data/dataset/train.txt'
dev_path= '../michat-data/dataset/dev.tsv'
bad_case_path='../michat-data/tfidf-lr-basecase.txt'

def read_stopword(path):
    stopword=[]
    with open(path,'r') as sw:
        for line in sw.readlines():
            if line!=' ':
                line=line.strip()
            if line not in stopword:
                stopword.append(line)
    return stopword

stopword=read_stopword(stopword_path)

def jiebacut(s):
    s_cut=jieba.lcut(s)
    s_cut=[w for w in s_cut if w not in stopword]
    return s_cut


def read_file(path):
    with open(path,'r') as file:
        texts,labels=[],[]
        i=0
        while True:
            line=file.readline()
            if line:
                line=line.strip().split('\t')
                text=line[0]
                text=jiebacut(text)
                label=1 if line[1]=='michat' else 0
                texts.append(text)
                labels.append(label)
            else:
                break
            i+=1
            print('\r{}'.format(i),end='',flush=True)
        return texts,labels

def acc(y_true,y_pred,bad_case):
    assert len(y_true)==len(y_pred)
    num=0
    idx=0
    for i,j in zip(y_true,y_pred):
        idx+=1
        if i==j:
            num+=1
        else:
            bad_case.append((dev_data[idx],dev_y[idx]))
    return num/len(y_true),bad_case

train_data,train_y=read_file(train_path)
dev_data,dev_y=read_file(dev_path)

vec=TfidfVectorizer()
train_tfidf=vec.fit_transform(train_data)
dev_tfidf=vec.transform(dev_data)


lr=LogisticRegression()
lr.fit(train_tfidf,train_y)
pred_y_lr=lr.predict(dev_tfidf)
lr_acc,bad_case=acc(dev_y,pred_y_lr)
lr_f1=f1_score(dev_y,pred_y_lr)

print("lr-acc:{}\tlr-f1:{}".format(lr_acc,lr_f1))


with open(bad_case_path,'a') as bfile:
    for i in bad_case:
        bfile.write('%s\n'%str(i))
    bfile.close()