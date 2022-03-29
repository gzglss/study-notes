#第一种dp
#dp[i][j]表示s[i:j+1]以s[j]结尾的最长回文子串长度
def longestPalindrome(self, s: str) -> str:
    dp=[[0 for _ in range(len(s))] for _ in range(len(s))]
    for i in range(len(s)):
        dp[i][i]=1
    maxlen=1
    mlstr=s[0]
    for i in range(len(s)-2,-1,-1):
        for j in range(i+1,len(s)):
            if s[j]==s[i] and dp[i+1][j-1]==j-i-1:
                dp[i][j]=dp[i+1][j-1]+2
            else:
                dp[i][j]=dp[i+1][j]
            if dp[i][j]>maxlen:
                maxlen=dp[i][j]
                mlstr=s[j-dp[i][j]+1:j+1]
    return mlstr


#第二种dp
#dp[i][j]表示s[i:j+1]是否为回文子串True or False
def longestPalindrome(self, s: str) -> str:
    dp=[[False]*len(s) for _ in range(len(s))]
    for i in range(len(s)):
        dp[i][i]=True
    
    maxlen=1
    mlstr=s[0]
    for l in range(2,len(s)+1):
        for i in range(len(s)):
            j=l+i-1
            if j>=len(s):
                break
            if s[j]==s[i]:
                if j-i<3:
                    dp[i][j]=True
                else:
                    dp[i][j]=dp[i+1][j-1]
            if dp[i][j] and l>maxlen:
                maxlen=l
                mlstr=s[i:j+1]
    return mlstr


#中心扩展法
#以每个元素作为中心，想两端同时增减
#base case：单个元素开始  or  双元素开始
def longestPalindrome(self, s: str) -> str:
    def expand(l,r):
        while l>=0 and r<len(s) and s[l]==s[r] :
            l-=1
            r+=1
        slen=r-l-1
        return slen,s[l+1:r]

    maxlen=1
    s0=s[0]
    for i in range(len(s)):
        slen1,s1=expand(i,i)
        slen2,s2=expand(i,i+1)
        if slen1>slen2 and slen1>maxlen:
            maxlen=slen1
            s0=s1
        elif slen1<=slen2 and slen2>maxlen:
            maxlen=slen2
            s0=s2
    return s0
