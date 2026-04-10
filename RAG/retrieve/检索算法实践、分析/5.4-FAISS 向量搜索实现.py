# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 19:01:25 2025

@author: liguo
"""
import numpy as np
import faiss

# 生成随机数据
dimension = 64  # 向量维度
n_samples = 10000  # 数据库大小
n_queries = 10  # 查询数量
np.random.seed(1234)
data = np.random.random((n_samples, dimension)).astype('float32')
queries = np.random.random((n_queries, dimension)).astype('float32')

# 创建索引
index = faiss.IndexFlatL2(dimension)  # 使用L2距离(欧式距离)
print(f"索引训练状态: {index.is_trained}")

# 添加向量到索引
index.add(data)
print(f"索引中的向量数: {index.ntotal}")

# 搜索
k = 5  # 返回最近邻的数量
distances, indices = index.search(queries, k)

print("查询结果:")
for i in range(n_queries):
    print(f"查询 {i} 的最近邻索引: {indices[i]}, 距离: {distances[i]}")

