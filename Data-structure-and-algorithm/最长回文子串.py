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
