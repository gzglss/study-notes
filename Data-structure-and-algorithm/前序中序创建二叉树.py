class TreeNode:
	def __init__(self,val,left=None,right=None):
		self.val=val
		self.left=left
		self.right=right

def getbtree(preorder,inorder):
	dict_inorder={j:i for i,j in enumerate(inorder)}
	def recur(pre_l,pre_r,in_l,in_r):
		if pre_l>pre_r or in_l>in_r:return
		root=TreeNode(preorder[pre_l])
		idx=dict_inorder[preorder[pre_l]]
		root.left=recur(pre_l+1,pre_l+1+idx-in_l,in_l,idx-1)
		root.right=recur(idx-in_l+pre_l+1,pre_r,idx+1,in_r)
		return root
	return recur(0,len(preorder)-1,0,len(inorder)-1)

preorder=[1,2,4,5,3,6]
inorder=[4,5,2,1,3,6]
tree=getbtree(preorder,inorder)
def pre(root):
	if not root:return
	pre(root.left)
	print(root.val)
	pre(root.right)
pre(tree)