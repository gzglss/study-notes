def cutline(m,n):
	dp=[[0 for i in range(n+1)] for _ in range(m+1)]
	#注意dp=[[0]*(n+1)]*(m+1)与上面代码的区别
	for i in range(m+1):
		dp[i][1]=1
	for i in range(2,m+1):
		for j in range(2,min(i+1,n+1)):
			if i-j<=1:
				dp[i][j]=1
			else:
				for k in range(1,j+1):
					dp[i][j]+=dp[i-j][k]
	return dp[m][n]

print(cutline(6,3))

# a=[[0]*5]*5
# a[1][1]=1
# print(a)#一变都变，更像是一种repeat增维的模式，而不是双层列表