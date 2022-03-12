def maxSubArray(self, nums: List[int]) -> int:
    if len(nums)<=1:return nums[0]
    # dp=[0]*len(nums)
    dp_0=nums[0]
    res=dp_0
    for i in range(1,len(nums)):
        a=max(nums[i],dp_0+nums[i])
        dp_0=a
        res=max(res,a)
    return res