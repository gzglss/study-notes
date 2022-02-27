def findtarget(nums,target):
	res=[]
	for i in nums:
		if i==target:
			res.append([i])
		elif i<target:
			temp=findtarget(nums,target-i)
			for a in temp:
				a.append(i)
				res.append(a)
	print(res)
	return res
nums=[2,3,5]
target=8
res=[]
print(findtarget(nums,target))