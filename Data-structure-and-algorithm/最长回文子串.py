def longesthuiwensubs(s):
	dp=[[0]*len(s) for _ in range(len(s))]
	#base case
	for i in range(len(s)):
		for j in range(len(s)):
			if i==j:
				dp[i][j]=1
			if i<j:
				dp[i][j]=0

	#dp
	huiwensub=s[0]
	max_len=1
	for i in range(len(s)-2,0,-1):
		for j in range(i+1,len(s)):
			if s[i]!=s[j]:
				dp[i][j]=0
			else:
				a=max(dp[i][j-1],dp[i+1][j-1],dp[i+1][j])
				if a>0:
					dp[i][j]=j+1-i
					if dp[i][j]>max_len:
						max_len=dp[i][j]
						huiwensub=s[i:j+1]
				else:
					dp[i][j]=0

	return huiwensub
s='ABCDZJUDCBA'
print(longesthuiwensubs(s))