环境：
- tf 1.14
- py 3.6
- cloudml

介绍：

以estimator设计文本分类

数据集：

头条新闻数据集

训练方法：

- bert+textcnn
- bert+bilstm
- lexicon-bert+bilstm
  - 但是使用的是小爱的lexicon，并不是t-news领域的词表，而且更新词表需要更新预训练模型，所以为了与词表对应，使用的是小爱训练好的lexicon-bert，所以对结果不仅没有提升，还存在大幅的下降

结果：

| 下游网络    | batch_size | lr | epoch       | acc   |
|---------|------------|------|-------------|-------|
| textcnn | 256        | warmup| 5           | 0.808 |
| bilstm  | 128        | warmup| 6           | 0.891 |
| lexicon | 128|warmup| 30(5-6就已收敛) | 0.857 |



