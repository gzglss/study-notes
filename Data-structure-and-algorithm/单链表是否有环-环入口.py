def linkwithcircur(a:ListNode) -> bool:
    if not a.next.next or not a.next:return False 
    slow,fast=a.next,a.next.next
    while fast:
        if slow==fast:
            return True
        slow=slow.next
        fast=fast.next.next
    return False



def circurenter(a:ListNode) -> ListNode:
    meet=None
    if not a.next.next or not a.next:meet=None
    slow,fast=a.next,a.next.next
    while fast and not meet:
        if slow==fast:
            meet=slow
        slow=slow.next
        fast=fast.next.next
    if meet:
        newslow=a
        while newslow!=slow:
            slow=slow.next
            newslow=newslow.next
        return newslow
    else:
        return None