# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#技巧：链表a变成a+b，链表b变成b+a，当两者相等的时候就是链表相交的时候，可以进行数学证明
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        cura,curb=headA,headB
        while cura!=curb:
            if cura:
                cura=cura.next
            else:
                cura=headB
            if curb:
                curb=curb.next
            else:
                curb=headA
        return cura