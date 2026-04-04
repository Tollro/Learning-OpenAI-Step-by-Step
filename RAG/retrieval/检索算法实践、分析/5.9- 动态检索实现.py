# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 20:27:40 2025

@author: liguo
"""
class DynamicRetriever:
    def __init__(self):
        self.query_history = []
        self.feedback_history = []
    
    def add_feedback(self, query, feedback):
        """记录用户反馈"""
        self.query_history.append(query)
        self.feedback_history.append(feedback)
    
    def retrieve(self, query, collection):
        """
        基于历史反馈的动态检索
        :param query: 当前查询
        :param collection: 待检索文档集合
        :return: 检索结果
        """
        # 简单实现：根据历史反馈调整查询
        expanded_query = query.copy()
        for i, past_query in enumerate(self.query_history):
            if set(query) & set(past_query):  # 有共同查询词
                expanded_query.extend(self.feedback_history[i])
        
        # 模拟检索过程
        results = []
        for doc in collection:
            score = sum(1 for term in expanded_query if term in doc)
            if score > 0:
                results.append({'doc': doc, 'score': score})
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

# 测试用例
retriever = DynamicRetriever()
retriever.add_feedback(['python'], ['programming', 'language'])
retriever.add_feedback(['AI'], ['machine', 'learning'])

collection = [
    'python programming language guide',
    'AI and machine learning',
    'data science with python',
    'introduction to AI'
]

current_query = ['python']
results = retriever.retrieve(current_query, collection)
print(f"查询: {current_query}")
print("结果:", [item['doc'] for item in results])

