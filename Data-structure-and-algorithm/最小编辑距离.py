def mineditdist(s1,s2):
	dp=[[0]*len(s1) for _ in s2]
	#base case
	for i in range(len(s2)):
		dp[i][0]=i
	for j in range(len(s1)):
		dp[0][j]=j

	#dp
	for i in range(1,len(s2)):
		for j in range(1,len(s1)):
			if s2[i]==s1[j]:
				dp[i][j]=dp[i-1][j-1]
			else:
				dp[i][j]=min(dp[i-1][j],dp[i-1][j-1],dp[i][j-1])+1
	return dp[len(s2)-1][len(s1)-1]

s1='aaaa'
s2='abc'
print(mineditdist(s1,s2))