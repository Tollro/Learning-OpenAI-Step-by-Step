# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 19:33:29 2025

@author: liguo
"""
#pip install hnswlib
import hnswlib
import numpy as np

# 参数设置
dimension = 64  # 向量维度
n_samples = 10000  # 数据库大小
n_queries = 10  # 查询数量
max_elements = n_samples  # 索引最大容量
ef_construction = 200  # 影响构建质量
M = 16  # 影响内存使用和构建时间

# 生成随机数据
np.random.seed(1234)
data = np.random.random((n_samples, dimension)).astype('float32')
queries = np.random.random((n_queries, dimension)).astype('float32')

# 创建索引
index = hnswlib.Index(space='l2', dim=dimension)  # 使用L2距离

# 初始化索引
index.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)

# 添加向量到索引
index.add_items(data)

# 设置查询时的ef参数(应 >= k)
ef_search = 100
index.set_ef(ef_search)

# 搜索
k = 5  # 返回最近邻的数量
labels, distances = index.knn_query(queries, k=k)

print("查询结果:")
for i in range(n_queries):
    print(f"查询 {i} 的最近邻索引: {labels[i]}, 距离: {distances[i]}")
