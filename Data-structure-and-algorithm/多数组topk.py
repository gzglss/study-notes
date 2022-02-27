def topkmutilarr(arrs,k):
	if len(arrs)==1:return arrs[-k:]
	if len(arrs)==0:return None
	arrs_new=[]
	max_idx=0
	max_num=arrs[0][-1]
	res=[]
	for i in range(len(arrs)):
		if len(arrs[i])>k:
			arrs_new.append(arrs[i][-k:])
		else:
			arrs_new.append(arrs[i])
		if arrs[i][-1]>max_num:
			max_idx=i
			max_num=arrs[i][-1]
	base_arr=arrs_new[max_idx]
	res.append(base_arr[-1])
	base_arr=base_arr[:-1]
	k-=1
	other_arr=arrs_new[:max_idx]+arrs_new[max_idx+1:]
	while k>0:
		max_num_2=other_arr[0][-1]
		max_num_2_idx=0
		for i in range(len(other_arr)):
			if other_arr[i][-1]>max_num_2:
				max_num_2=other_arr[i][-1]
				max_num_2_idx=i
		if max_num_2>=base_arr[-1]:
			other_arr[max_num_2_idx]=other_arr[max_num_2_idx][:-1]
			res.append(max_num_2)
		else:
			res.append(base_arr[-1])
			base_arr=base_arr[:-1]
		k-=1
	return res
arrs=[[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]]
print(topkmutilarr(arrs,8))