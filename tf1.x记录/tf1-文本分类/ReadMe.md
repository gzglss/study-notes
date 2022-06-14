环境：
- tf 1.14
- py 3.6
- cloudml

介绍：

以estimator设计文本分类

数据集：

头条新闻数据集

结果：

| 下游网络    | batch_size | lr     | acc   |
|---------|------------|--------|-------|
| textcnn | 256        | warmup | 0.808 |
| bilstm  | 128        | warmup | 0.889 |

