def fastsort2(l,r):
    if l>=r:return
    i,j=l,r
    while i<j:
        while i<j and num[j]>=num[l]:
            j-=1
        while i<j and num[i]<num[l]:
            i+=1
        num[i],num[j]=num[j],num[i]
    num[l],num[i]=num[i],num[l]
    fastsort2(l,i-1)
    fastsort2(i+1,r)
num=[3,3,3,3,3,3,2,4,6,5,7,4,8,3,2,56,76,45,3,0,1]
fastsort2(0,len(num)-1)
print(num)