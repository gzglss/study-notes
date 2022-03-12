def integerBreak(self, n: int) -> int:
    dp=[0]*(n+1)
    dp[1],dp[2]=0,1
    for i in range(3,n+1):
        for j in range(1,i//2+1):
            dp[i]=max(dp[i],j*(i-j),j*dp[i-j])
    return dp[n]



def integerBreak(self, n: int) -> int:
    if n<4:
        return n-1
    a,b,x=n//3,n%3,3
    res=1
    while a:
        if a%2:
            res=res*x
        x*=x#快速幂
        a//=2
    if b==1:
        return res//3*4
    if b==0:
        return res
    if b==2:
        return res*2