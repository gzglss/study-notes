import numpy as np
import random
def distance(x1,x2):
    #计算两点距离
    return np.sqrt(np.squeen(x1[0]-x2[0]),np.squeen(x1[1]-x2[1]))
    
def centerpoint(x):
    #计算中心点坐标
    x1=[i[0] for i in x]
    x2=[i[1] for i in x]
    return [np.means(x1),np.means(x2)]

def kmeans(data,cluster,iteration):
    '''
    data=[[x1,x2],[x1,x2],...]
    '''
    all_k=[]
    for k in cluster:
        #k表示选取k个初始点，此处仅使用随机选取
        centerp=[0 for _ in range(k)]
        kmeansres=None
        for _ in range(iteration):
            flag=0
            if not centerp:
                centerp=random.sample(data,k)
            pointlist=[[] for _ in range(k)]
            for i in data:
                if i not in init_point:
                    dist=[distance(i,j) for j in centerp]
                    minp=np.argmin(dist)
                    pointlist[minp].append(i)
            for idx,i in enumerate(pointlist):
                cp=centerpoint(i)
                if cp!=centerp[idx]:
                    centerp[idx]=cp
                else:
                    flag+=1
            if flag==4 or _==iteration-1:
                kmeansres=pointlist
                break
        all_center.append(kmeansres)
    return all_center