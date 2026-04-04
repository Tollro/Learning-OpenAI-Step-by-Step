# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 19:29:04 2025

@author: liguo
"""
#pip install annoy
from annoy import AnnoyIndex
import numpy as np

# 参数设置
dimension = 64  # 向量维度
n_samples = 10000  # 数据库大小
n_queries = 10  # 查询数量
n_trees = 10  # 树的数量，影响构建时间和搜索精度

# 生成随机数据
np.random.seed(1234)
data = np.random.random((n_samples, dimension)).astype('float32')
queries = np.random.random((n_queries, dimension)).astype('float32')

# 创建索引
index = AnnoyIndex(dimension, 'angular')  # 使用余弦相似度

# 添加向量到索引
for i in range(n_samples):
    index.add_item(i, data[i])

# 构建索引
index.build(n_trees)

# 搜索
k = 5  # 返回最近邻的数量
results = []
for query in queries:
    neighbors = index.get_nns_by_vector(query, k, include_distances=True)
    results.append(neighbors)

print("查询结果:")
for i, (indices, distances) in enumerate(results):
    print(f"查询 {i} 的最近邻索引: {indices}, 距离: {distances}")
