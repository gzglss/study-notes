def maxmatrix(matrix:List[List[int]]) -> int:
    if len(matrix)==1 or len(matrix[0])==1:
        if sum(matrix)!=0:return 1
        else:return 0
    if len(matrix)==0 or len(matrix[0])==0:return 0
    ans=0
    #定义temp，并初始化
    temp=[[] for _ in range(len(matrix[0])) for i in range(len(matrix))]
    temp[0]=matrix[0]
    for i in range(len(matrix)):
        temp[i][0]=matrix[i][0]
    #遍历剩余值，应用dp规律给temp元素赋值
    for i in range(1,len(matrix)):
        for j in range(1,len(matrix[0])):
            if matrix[i][j]==0:
                temp[i][j]==0
            else:
                temp[i][j]=min(temp[i-1][j],temp[i-1][j-1],temp[i][j-1])+1
                if temp[i][j]>ans:
                    ans=temp[i][j]
    return ans