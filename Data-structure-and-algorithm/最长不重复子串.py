#动态规划
# def longestnorepeatsubstr(s):
# 	sub=s[0]
# 	dp=[0]*len(s)
# 	dp[0]=1
# 	longest_sub=sub
# 	max_len=1
# 	for i in range(1,len(s)):
# 		if s[i] not in sub:
# 			sub+=s[i]
# 			dp[i]=dp[i-1]+1
# 			if dp[i]>max_len:
# 				max_len=dp[i]
# 				longest_sub=sub
# 		else:
# 			sub=s[i]
# 			dp[i]=1
# 	return longest_sub
# s='aaaaaa'
# print(longestnorepeatsubstr(s))

#双指针
def longestnorepeatsubstr(s):
	l,r=0,1
	longest_sub=s[0]
	sub_len=1
	max_len=1
	sublen_dic={}
	while r<len(s):
		if s[r] not in s[l:r]:
			sub_len+=1
			longest_sub+=s[r]
			r+=1
			if sub_len>max_len:
				max_len=sub_len
			if r==len(s):
				sublen_dic[max_len]=longest_sub
		else:
			sublen_dic[max_len]=longest_sub
			while s[l]!=s[r]:
				l+=1
			l+=1
			r+=1
			sub_len=r-l
			longest_sub=s[l:r]
			if sub_len>max_len:
				max_len=sub_len
	return sublen_dic[max_len]
s='abcedag'
print(longestnorepeatsubstr(s))