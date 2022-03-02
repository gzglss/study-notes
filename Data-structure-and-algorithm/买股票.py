def maxProfit(self, prices: List[int]) -> int:
    if len(prices)<=1:return 0
    l=0
    while l<len(prices)-1 and prices[l]>=prices[l+1]:
        l+=1
    if l==len(prices)-1:
        return 0
    buy_min=prices[l]
    ans=prices[l+1]-buy_min
    for i in range(l+1,len(prices)):
        if i<len(prices)-1 and prices[i]<buy_min:
            buy_min=prices[i]
            continue
        ans=max(ans,prices[i]-buy_min)
    return ans