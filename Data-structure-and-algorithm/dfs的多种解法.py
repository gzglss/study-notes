#递归
#类似于先序遍历，但是既可以先左后右，也可以先右后左
#根左右  OR   根右左
def dfs(root):
	if not root:return
	'''
	这里对每个节点进行操作
	'''
	dfs(root.left)
	dfs(root.right)



#迭代
#如果先左后右就先存右节点；反之亦反
stack=[root]
while not stack:
	node=stack.pop()
	'''
	这里对每个节点进行操作
	'''
	if node.right:
		stack.append(node.right)
	if node.left:
		stack.append(node.left)