'''
时间：2022-04-27
作者：gzg
字典树的实现
'''

import collections


class TrieNode:
    def __init__(self):
        # 定义字典存储的数据类型为TrieNode
        self.children = collections.defaultdict(TrieNode)
        # 标志符，用以表示当前字符串是否存在，一般可以通过将其置为False来间接删除该字符串
        # 也可以定义数值来统计字符串个数
        self.is_word = False


class MyTrie:
    def __init__(self):
        self.root = TrieNode()
        self.max_depth = 0

    def insert(self, word):
        cur = self.root
        depth = 0
        for c in word:
            cur = cur.children[c]
            depth += 1
        cur.is_word = True
        self.max_depth = max(depth, self.max_depth)

    def seach(self, word):
        cur = self.root
        for c in word:
            cur = cur.children.get(c)
            if not cur:
                return False
        return cur.is_word

    def customfunc(self, string_):
        '''
        自定义相关功能
        '''
        pass
