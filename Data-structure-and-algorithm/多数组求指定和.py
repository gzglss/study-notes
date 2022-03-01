#给定三个等长数组，每个数组选出一个元素，要求相加和为64，要求输出每种三个数组合，它们的元素出现次数乘积，再对所以可能求和，要求时间复杂度O(n)，空间复杂度O(1)。
#举个例子，三个数组分别是[1,1,1,3,3],[1,1,1,5,5],[62,62,62,56,100]，
#输出3*3*3+2*2*1=27+4=31，因为一种组合是1，1，62，分别出现了3次，3次，3次，另一种组合是3，5，56，分别出现了2次，2次1次
#需假设target一定存在，同时题目并未说都是正数，所以没必要对每个数组卡64的阈值
def mutilarrtarget(a,b,c,target):
	dic_a,dic_b,dic_c,res={},{},{},0
	for i in range(len(a)):
		if a[i] not in dic_a:
			dic_a[a[i]]=1
		else:
			dic_a[a[i]]+=1
	for i in range(len(b)):
		if b[i] not in dic_b:
			dic_b[b[i]]=1
		else:
			dic_b[b[i]]+=1
	for i in range(len(c)):
		if c[i] not in dic_c:
			dic_c[c[i]]=1
		else:
			dic_c[c[i]]+=1
	for k1,v1 in dic_a.items():
		for k2,v2 in dic_b.items():
			e=64-k1-k2
			d=dic_c.get(e)
			if d:
				res+=v1*v2*d
	return res
a=[-1,1,3,3]
b=[-5,-5,1,1,5,5]
c=[70,62,62,62,56,100]
target=64
print(mutilarrtarget(a,b,c,target))