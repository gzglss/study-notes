import numpy as np
def viterbi(node,trans):
	'''
	node.shape=[seq_len,num_label]
	trans.shape=[num_label,num_label]
	'''
	paths=[]
	score=node[0].reshape(-1,1)#[num_label,1]
	for t in range(1，node.shape[0]):
		#寻找t时刻的最优路径
		observe=node[t].reshape(-1,1)
		#从t-1到t的得分计算
		m=score+trans+observe#numpy自动补全维度
		score=np.max(m,axis=0).reshape(-1,1)
		idx=np.argmax(m,axis=0)
		paths.append(idx.tolist())#将时刻t的路径路径存入paths

	best_path=[0]*(node.shape[0])
	#score为最后时刻的得分，也就是最优一个路径可以直接通过score算的
	best_path[-1]=np.argmax(score)
	for t in range(node.shape[0]-2,-1,-1):
		idx=best_path[t+1]#有点不太理解：为什么t+1时刻得到的idx可以用于t时刻的最优路径判断？
		best_path[t]=paths[t][idx]
	return best_path