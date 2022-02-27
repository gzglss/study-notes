nums=[1,2,3,4,5,5,5,6,4,3,2,1]
target=3
hashdic={}
for i in range(len(nums)):
    if nums[i] not in hashdic:
        hashdic[nums[i]]=[i]
    else:
        hashdic[nums[i]].append(i)

def fastsort(l,r):
    if l>=r:return
    i,j=l,r
    while i<j:
        while i<j and nums[j]>=nums[l]:
            j-=1
        while i<j and nums[i]<nums[l]:
            i+=1
        nums[i],nums[j]=nums[j],nums[i]
    nums[i],nums[l]=nums[l],nums[i]
    fastsort(i+1,r)
    fastsort(l,i-1)
fastsort(0,len(nums)-1)

if target in hashdic:
    print(hashdic[target][0])
else:
    if nums[0]<target and nums[-1]>target:
        l,r=0,len(nums)-1
        while l<=r:
            mid=l+(r-l)//2
            if nums[mid]>target:
                if nums[mid-1]>target:
                    r=mid-1
                elif nums[mid-1]<target:
                    print(mid)
                    break
            elif nums[mid]<target:
                l=mid+1
    else:
        print(len(nums)+1)