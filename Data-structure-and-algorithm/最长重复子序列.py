def longestrepeatsubsentence(a:List[int],b:List[int]) -> List[int]:
    res=[[0]*(len(a)+1) for _ in range(len(b)+1)]
    #res[i][j]表示s1[:i-1]与s2[:j-1]的最长公共子序列
    ans=[]#存放公共子序列的索引
    ans_res=[]#存放公共子序列的长度值，每个长度只存一个，且存的方向需要保持一致
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            if a[i]==b[j]:
                res[i][j]=res[i-1][j-1]+1
                if res[i][j] not in ans_res:
                    #因为是j为第二层遍历，所以会先遍历到横向列表得到的最长子序列，相当于已经选好了方向
                    ans.append(i)
                    ans_res.append(res[i][j])
            else:
                if res[i-1][j]!=res[i][j-1]:
                    res[i][j]=max(res[i-1][j],res[i][j-1])
                else:
                    res[i][j]=res[i-1][j]
    return [a[i] for i in ans]