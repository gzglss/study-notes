import numpy as np

def viterbi(nodes,trans):
    '''
    :param nodes:shape=[seq_len,num_labels]     每个token属于每个标注的概率 相当于当前输入
    :param trans: shape=[num_labels,num_labels] 每个token的标注转移到其他token的概率 相当于上一个时刻的标注
    :return: 序列标注
    '''
    seq_len,num_label=len(nodes),len(trans)
    # [num_labels,1]，索引从0开始，第0个token的标注（一般第0个token的标注是确定的）到第一个token的num_labels个标注的得分
    #如果对scores取argmax，就表示选择当前token的最大得分标注
    scores=nodes[0].reshape((-1,1))#[num_labels,1]表示第一个token属于每个tag的得分，发射得分
    #第0个token到第1个token时，各个tag的发射概率

    paths=[]
    for t in range(1,seq_len):
        scores_rep=np.repeat(scores,num_label,axis=1)#[num_labels,num_labels]，初始化得分矩阵
        observe=nodes[t].reshape((1,-1))#[1,num_labels]，当前token属于每个序列的概率
        observe_rep=np.repeat(observe,num_label,axis=0)#[num_labels,num_labels]
        m=scores_rep+trans+observe_rep#这也是HMM的三要素：初始化矩阵、转移矩阵、发射矩阵
        scores=np.max(m,axis=0).reshape((-1,1))#[num_labels,1]，第t-1个token到第t个token的每个标注的得分
        #idxs表示第t个token的所有标注到下一个token每个标注得分的最优得分的索引
        #第t个token有num_labels个标注，其中每个标注预测的下一个token的标注的得分都不一样
        #idxs表示第t-1个token到第t个token中，概率最大的第t-1个token的tag
        #t_0     t_1  t_2  ... t_n
        #   \     \    |        /
        #           t-1_max
        #idxs=[[t_0,t_1,t_2,...,t_n]]
        idxs=np.argmax(m,axis=0)#m:[num_labels,num_labels]，idxs:[1,num_labels]
        #paths存储的是从当前时刻预测下一时刻时，num_labels产生的num_labels个最优的结果的索引
        paths.append(idxs.tolist())#idxs:[num_labels,],paths:[seq_len-1,num_labels]

    #此时所有路径的得分已经算出来了，我们最优路径需要倒推，也就是我们已经知道最优路径的最后一个token选择的标注，现在需要一步步往前推，得到整个路径

    best_path=[0]*seq_len
    #根据上面的for循环，scores每次循环都会更新
    #所以当前的scores表示最后一个token到结束符的num_labels个得分
    best_path[-1]=np.argmax(scores)#当未指定axis时，表示对所有的数取最大值的索引，也就是为[0,num_labels)中的一个，选择最后一个token的最优标注
    for i in range(seq_len-2,-1,-1):
        idx=best_path[i+1]#第i+1个时刻的最优标注的索引
        #paths[i]：第i个token的num_labels个标注预测第i+1个token的最优的结果
        #而第i+1的最优标注的索引已经确定，所以只需要选取第i+1个最优标注所对应的第i个标注就可以确定第i个token的最优标注
        #以此类推，得到最优的路径
        best_path[i]=paths[i][idx]

    return best_path