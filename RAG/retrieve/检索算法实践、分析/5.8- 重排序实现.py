# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 20:23:17 2025

@author: liguo
"""
class ReRanker:
    def __init__(self):
        # 模拟预训练模型权重
        self.model_weights = {
            'relevance': 0.6,
            'popularity': 0.2,
            'freshness': 0.2
        }
    
    def rerank(self, initial_results):
        """
        基于多特征重新排序结果
        :param initial_results: 初始结果列表，每个元素为包含特征的dict
        :return: 重新排序后的结果
        """
        # 计算综合得分
        for item in initial_results:
            item['score'] = sum(
                item.get(feature, 0) * weight 
                for feature, weight in self.model_weights.items()
            )
        
        # 按得分降序排序
        return sorted(initial_results, key=lambda x: x['score'], reverse=True)

# 测试用例
ranker = ReRanker()
initial_results = [
    {'id': 1, 'relevance': 0.9, 'popularity': 0.5, 'freshness': 0.3},
    {'id': 2, 'relevance': 0.7, 'popularity': 0.8, 'freshness': 0.6},
    {'id': 3, 'relevance': 0.8, 'popularity': 0.4, 'freshness': 0.9}
]
reranked_results = ranker.rerank(initial_results)
print("初始排序:", [item['id'] for item in initial_results])
print("重排序后:", [item['id'] for item in reranked_results])