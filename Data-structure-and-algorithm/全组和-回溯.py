def allgather(arr,target):
	#类比凑硬币的题目，arr相当于面值，target相当于需凑金额
	#但是还有所不同，每个面值的硬币可以重复使用，但是arr的元素不能
	#可以使用动态规划的办法
	#状态：arr、target
	#选择：选择当前值arr.remove()、target-；不选择当前值arr[i+1]
	#状态转移方程：dp[i][j]=dp[i-1][j]  OR  dp[i][j]=dp[i-1][j-arr[i]]
	arr.sort()
	res,path=[],[]
	def backtrack(arr,target,start):
		if target==0:
			res.append(list(path))
			return
		for i in range(start,len(arr)):
			if i>start and arr[i]==arr[i-1]:
				continue
			if target-arr[i]>=0:
				path.append(arr[i])
				backtrack(arr,target-arr[i],i+1)
				path.pop()
	backtrack(arr,target,0)
	return res
arr=[6,5,5,6,5]
target=11
print(allgather(arr,target))