"""
公交情感分析，01分类
数据来源：微博
作者：gzg
2022-12-3
"""
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re

tqdm.pandas()
data_path = "/kaggle/input/bus-emo-900/bus_emo_900.csv"

# texts,labels=[],[]
# with open(data_path,'r',encoding='gbk',errors='ignore') as f:
#     while True:
#         line=f.readline()
#         if line:
#             line=line.strip().split(',')
#             if len(line[-2])==0:
#                 break
#             texts.append(line[-2])
#             labels.append(line[-1])
#         else:
#             break
# dic={}
# dic['text'],dic['label']=texts,labels
# df=pd.DataFrame(dic)

# #数据去重
# df=df.drop_duplicates()

# train_df,valid_df=train_test_split(df,test_size=0.2,random_state=1)

# #需要对数据的长度进行处理，长度太过参差
# #工业上一般短文本和长文本使用不同的判断手段
# df['text'].apply(len).describe()

df = pd.read_csv(data_path, encoding='gbk')
# 数据去重
df = df.drop_duplicates()
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=1)


def char_change(text):
    """
    转换【***】为。***。的形式
    """
    if "【" in text:
        lst = list(text)
        s = lst.index('【')
        e = 0
        if '】' in text:
            e = lst.index('】')
        if e == 0:
            lst.remove('【')
        else:
            if s == 0:
                lst[e] = '。'
                lst.remove('【')
            else:
                lst[s] = '。'
                lst[e] = '。'
        return ''.join(lst)
    elif "】" in text:
        lst = list(text)
        lst.remove('】')
        return ''.join(lst)
    return text


def clean_suffix(text):
    # 主要用来清除@和#后面的文字
    chars = ":：!！?？,，.。@#（）()<>《》“”"";；"
    lst = list(text)
    while '@' in lst:
        idx = lst.index('@')
        i = idx + 1
        e = -1
        while i < len(lst):
            if lst[i] in chars:
                e = i - 1
                break
            if i == len(lst) - 1:
                e = i
            i += 1
        if e >= 0:
            tmp = lst[:idx][:] + lst[e + 1:][:]
            lst = tmp[:]
        else:
            break
    if len(lst) == 0:
        return text
    text = ''.join(lst)
    if '#' in text:
        pattern = re.compile('\#.*\#')
        text = re.sub(pattern, '', text)
    return text


def clean_url(text):
    sentences = text.split(' ')
    # 处理http://类链接
    url_pattern = re.compile(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', re.S)
    # 处理无http://类链接
    domain_pattern = re.compile(r'(\b)*(.*?)\.(com|cn)')
    if len(sentences) > 0:
        result = []
        for item in sentences:
            text = re.sub(url_pattern, '', item)
            text = re.sub(domain_pattern, '', text)
            result.append(text)
        return ' '.join(result)
    else:
        return re.sub(url_pattern, '', sentences)


def clean_html(text):
    html_pattern = re.compile('</?\w+[^>]*>', re.S)
    text = re.sub(html_pattern, '', text)
    return text


def clean_tag(text):
    tag_pattern = re.compile('(\[|\#|【)(.*?)(\#|\]|\】)', re.S)
    text = re.sub(tag_pattern, '', text)
    return text.strip()


def clean_at(text):
    # 暂时不用，因为有的@出现在句中或靠前，此时就会将后面的所有字都去除
    at_pattern = re.compile('@\S*', re.S)
    text = re.sub(at_pattern, '', text)
    return text.strip()


def clean_nan(text, label):
    """
    清除空数据
    """
    new_text, new_label = [], []
    for i in range(len(label)):
        text[i] = text[i].strip()
        if text[i] != '':
            new_text.append(text[i])
            new_label.append(label[i])
    return new_text, new_label


texts_df = train_df['text'].progress_apply(char_change)
texts_df = texts_df.progress_apply(clean_suffix)
texts_df = texts_df.progress_apply(clean_url)
texts_df = texts_df.progress_apply(clean_html)
labels_df = train_df['label']
train_texts, train_labels = clean_nan(texts_df.values, labels_df.values)

test_texts = valid_df['text'].apply(char_change)
test_texts = test_texts.progress_apply(clean_suffix)
test_texts = test_texts.progress_apply(clean_url)
test_texts = test_texts.progress_apply(clean_html)
test_labels = valid_df['label']
test_texts, test_labels = clean_nan(test_texts.values, test_labels.values)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


class Bert(nn.Module):
    def __init__(self, num_class, emb_size, max_length):
        super(Bert, self).__init__()
        self.max_length = max_length
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(emb_size, num_class)

    def forward(self, inputs):
        data_token = tokenizer.batch_encode_plus(inputs,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=self.max_length)

        input_ids = torch.tensor(data_token["input_ids"]).to(device)
        attention_mask = torch.tensor(data_token["attention_mask"]).to(device)
        token_type_ids = torch.tensor(data_token["token_type_ids"]).to(device)
        encode = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.dropout(encode[0][:, 0, :])
        output = self.linear(output)
        return output


class BertLstm(nn.Module):
    def __init__(self, num_class, emb_size, max_length, unit):
        super(BertLstm, self).__init__()
        self.max_length = max_length
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.bilstm = nn.LSTM(input_size=emb_size, hidden_size=unit, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(unit * 2, num_class)

    def forward(self, inputs):
        data_token = tokenizer.batch_encode_plus(inputs,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=self.max_length)

        input_ids = torch.tensor(data_token["input_ids"]).to(device)
        attention_mask = torch.tensor(data_token["attention_mask"]).to(device)
        token_type_ids = torch.tensor(data_token["token_type_ids"]).to(device)
        encode = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.bilstm(encode[0][:, 0, :])
        output = self.dropout(output[0])
        output = self.linear(output)
        return output


class TextCNN(nn.Module):
    def __init__(self, filter_sizes, emb_size, in_channels, num_filter):
        super(TextCNN, self).__init__()
        self.convlist = nn.ModuleList([nn.Conv2d(in_channels=in_channels,
                                                 out_channels=num_filter,
                                                 kernel_size=[filter_size, emb_size]) for filter_size in filter_sizes])

    def forward(self, inputs):
        encodes = []
        inputs = inputs.unsqueeze(1)
        for conv in self.convlist:
            output = conv(inputs)
            output = F.relu(output)
            output = output.squeeze(3)
            output = F.max_pool1d(output, output.shape[2]).squeeze(2)
            encodes.append(output)
        encode = torch.concat(encodes, dim=-1)
        return encode


class bert_textcnn(nn.Module):
    def __init__(self, num_filter, filter_sizes, in_channels, num_class, emb_size, max_length):
        super(bert_textcnn, self).__init__()
        self.max_length = max_length
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.textcnn = TextCNN(filter_sizes=filter_sizes,
                               emb_size=emb_size,
                               in_channels=in_channels,
                               num_filter=num_filter)
        self.textcnn.to(device)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(num_filter * len(filter_sizes), num_class)

    def forward(self, inputs):
        token_dic = tokenizer.batch_encode_plus(inputs,
                                                padding=True,
                                                truncation=True,
                                                max_length=self.max_length)
        input_ids = torch.tensor(token_dic["input_ids"]).to(device)
        attention_mask = torch.tensor(token_dic["attention_mask"]).to(device)
        token_type_ids = torch.tensor(token_dic["token_type_ids"]).to(device)
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)
        last_hidden_out = bert_output[0][:, :, :]
        textcnn_output = self.textcnn(last_hidden_out)
        output = self.dropout(textcnn_output)
        output = self.linear(output)
        return output


# #bert-base
# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bert=Bert(2,768,128)
# bert.to(device)

# #bert+bilstm
# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bert_bilstm=BertLstm(num_class=2,emb_size=768,max_length=150,unit=128)
# bert_bilstm.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_textcnn = bert_textcnn(num_filter=128, filter_sizes=[3, 4, 5], in_channels=1, num_class=2, emb_size=768,
                            max_length=150)
bert_textcnn.to(device)


class mydataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.texts = data[0]
        self.labels = list(map(int, data[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


dataset = mydataset((train_texts, train_labels))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# 分层学习率
optim_group = [
    {"params": [p for n, p in bert_textcnn.named_parameters() if 'bert' in n], "lr": 1e-5},
    {"params": [p for n, p in bert_textcnn.named_parameters() if 'bert' not in n], "lr": 1e-4}
]

lf = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(optim_group, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=len(dataloader), gamma=0.6)


def notrain(model, test_data, num_pos, num_neg):
    """
    不进行微调，直接使用预训练模型进行预测
    """
    texts, labels = test_data[0], test_data[1]
    dataloader = torch.utils.data.DataLoader(mydataset((texts, labels)), batch_size=1, shuffle=False)
    pos, neg = 0, 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            text, label = batch[0], batch[1]
            probas = model(text)
            pred = torch.argmax(probas, dim=1)
            label = label.numpy()[0]
            pred = pred.detach().cpu().numpy()[0]
            if pred == label:
                if pred == 1:
                    pos += 1
                if pred == 0:
                    neg += 1
            print("\rprocess:{}/{}".format(i + 1, len(dataloader)), end='')
    print()
    print("正类的准确率：", pos / num_pos)
    print("负类的准确率：", neg / num_neg)
    print("整体的准确率：", (pos + neg) / len(dataloader))


def train(model):
    """
    没进行一个epoch，进行一次模型评估
    """
    for epoch in range(5):
        model.train()  # 因为每个epoch都会进行验证了，此时模型进入了eval模式，所有需要将train模型放到循环内部
        print('=' * 10, "epoch:{}/{}".format(epoch + 1, 5), '=' * 10)
        total_loss = 0
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            text, label = batch[0], batch[1]
            label = label.to(device)
            output = model(text)
            loss = lf(output, label).to(device)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print("\rprocess:{}/{} | loss={} | lr={}".format(i + 1, len(dataloader), total_loss / 10,
                                                                 scheduler.get_lr()[0]), end='')
                total_loss = 0
            scheduler.step()
        print()
        # 根据loss保存最优模型，每个epoch结束后都进行一次预测
        print("开始预测...")
        predict(epoch + 1, model)
    return model


def test(model, test_data, num_neg, num_pos, epoch):
    bad_text, bad_pred, bad_label = [], [], []
    texts, labels = test_data[0], test_data[1]
    dataloader = torch.utils.data.DataLoader(mydataset((texts, labels)), batch_size=1, shuffle=False)
    model.eval()
    pos, neg = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            text, label = batch[0], batch[1]
            probas = model(text)
            pred = torch.argmax(probas, dim=1)
            label = label.numpy()[0]
            pred = pred.detach().cpu().numpy()[0]
            if pred == label:
                if pred == 0:
                    neg += 1
                if pred == 1:
                    pos += 1
            else:
                # 输出bad case
                bad_text.append(text[0])
                bad_pred.append(pred)
                bad_label.append(label)
            print("\rprocess:{}/{}".format(i + 1, len(dataloader)), end='')
    print()
    # 保存bad case文件
    bad_df = pd.DataFrame({'text': bad_text, 'label': bad_label, 'pred': bad_pred})
    bad_df.to_csv('./bad_case_{}_5e-6.csv'.format(epoch), index=False)
    print("正类的准确率：", pos / num_pos)
    print("负类的准确率：", neg / num_neg)
    print("整体的准确率：", (pos + neg) / len(dataloader))


def predict(epoch, model):
    num_pos, num_neg = 0, 0
    for i in test_labels:
        if i == 0:
            num_neg += 1
        elif i == 1:
            num_pos += 1
        else:
            print('error')
    test(model, (test_texts, test_labels), num_neg, num_pos, epoch)


model = train(bert_textcnn)
