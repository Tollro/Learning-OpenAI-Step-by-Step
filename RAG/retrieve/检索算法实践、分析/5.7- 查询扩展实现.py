# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 20:19:38 2025

@author: liguo
"""
from collections import defaultdict

class QueryExpander:
    def __init__(self):
        # 初始化同义词词典
        self.synonyms = defaultdict(list)
        self.synonyms.update({
            'car': ['automobile', 'vehicle', 'motorcar'],
            'movie': ['film', 'picture', 'flick'],
            'phone': ['mobile', 'cellphone', 'smartphone']
        })
    
    def expand_with_synonyms(self, query):
        """
        使用同义词扩展查询
        :param query: 原始查询词列表
        :return: 扩展后的查询词列表
        """
        expanded = []
        for term in query:
            expanded.append(term)
            if term in self.synonyms:
                expanded.extend(self.synonyms[term])
        return expanded

# 测试用例
expander = QueryExpander()
original_query = ['car', 'movie']
expanded_query = expander.expand_with_synonyms(original_query)
print(f"原始查询: {original_query}")
print(f"扩展查询: {expanded_query}")

