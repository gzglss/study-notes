"""
实现对线段树的定义
"""

class Node:
    def __init__(self,l,r):
        self.left=None
        self.right=None
        self.l=l
        self.r=r
        self.mid=l+(r-l)//2#表示将根结点的区间均分
        self.v=0#线段树节点对应的值
        self.add=0#懒值

class SegmentTree:
    def __init__(self):
        self.root=Node(1,int(1e9))

    def modify(self,l, r, v, node=None):
        #修改区间内的节点值
        if l>r:
            #当l==r的时候表示叶子节点
            return
        if node is None:
            node=self.root#初始化
        if node.l>=l and node.r<=r:
            #表示该node表示的区间在[l,r]内
            #此时该node的所有子树都在[l,r]区间内，修改节点值后返回
            node.v=v#修改该节点的值
            #给懒值赋值，看代码的样子，给懒值赋值后，懒值就不会变化了
            #而且懒值就是所要修正的目标值
            node.add=v#就是一个tmp，用来暂时存储需要修改的值
            return
        self.pushdown(node)#给root的子树赋值
        #需要修改的区间的左端点小于根结点区间的中点，说明左子树与该区间存在交集，所以需要修改左子树
        if l<=node.mid:
            self.modify(l,r,v,node.left)
        if r>node.mid:
            self.modify(l,r,v,node.right)
        self.pushup(node)#通过子树的值改变root的值

    def query(self,l,r,node=None):
        #查找任务
        if l>r:
            return 0
        if node is None:
            node=self.root#初始化
        if node.l>=l and node.r<=r:
            #当该节点不需要再分的时候返回该节点的值
            return node.v
        self.pushdown(node)
        v=0
        if l<=node.mid:
            #v的最终值就是前面的node.v
            v=max(v,self.query(l,r,node.left))
        if r>node.mid:
            v=max(v,self.query(l,r,node.right))
        return v

    def pushup(self,node):
        #之所以回溯的是最大值，是因为该题要求区间内的最大高度
        node.v=max(node.left.v,node.right.v)

    def pushdown(self,node):
        # 将输入区间的左半部分分到左子树，将右半部分分到右子树
        #建立线段树
        if node.left is None:
            node.left=Node(node.l,node.mid)
        if node.right is None:
            node.right=Node(node.mid+1,node.r)
        if node.add:
            #当懒值存在时，将懒值赋给子树的节点，同时将懒值传递给子树的懒值
            #node.add给我的感觉就是一个tmp
            node.left.v=node.add
            node.right.v=node.add
            node.left.add=node.add
            node.right.add=node.add
            node.add=0#懒值使用完后归零

from typing import List
class Solution:
    def fallingSquares(self,positions:List[List[int]]) -> List[int]:
        ans=[]
        mx=0
        tree=SegmentTree()
        for l,w in positions:
            r=l+w-1#因为r处为右端点，并不会发生方块的叠加
            #该方块落下后，方块上端的纵坐标
            #tree.query(l,r)查询[l,r]区间的线段树节点的值
            #查询被[l,r]区间包围的线段树节点的值
            h=tree.query(l,r)+w
            #比较方块当前的高度与此时整体的最高的高度，选择高的
            mx=max(mx,h)
            ans.append(mx)
            tree.modify(l,r,h)#修改对应线段树节点的值，也就是区间方块的最高值
        return ans