# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 18:13:27 2025

@author: liguo
"""
import numpy as np
import random
from typing import List, Dict, Tuple

class LSH:
    def __init__(self, dim: int, num_tables: int = 10, hash_size: int = 4):
        """
        初始化LSH近似最近邻搜索
        
        参数:
            dim: 数据维度
            num_tables: 哈希表数量(默认10)
            hash_size: 每个哈希函数的位数(默认4)
        """
        self.dim = dim
        self.num_tables = num_tables
        self.hash_size = hash_size
        
        # 生成随机投影向量
        self.projections = [np.random.randn(dim, hash_size) for _ in range(num_tables)]
        
        # 初始化哈希表
        self.tables: List[Dict[int, List[int]]] = [{} for _ in range(num_tables)]
        self.data: List[np.ndarray] = []
    
    def _hash(self, vec: np.ndarray, table_idx: int) -> int:
        """计算向量的哈希值"""
        projection = self.projections[table_idx]
        # 计算投影并二值化
        bits = (np.dot(vec, projection) > 0).astype(int)
        # 将位数组转换为整数哈希值
        return int(''.join(map(str, bits)), 2)
    
    def add(self, vec: np.ndarray) -> int:
        """添加向量到索引中"""
        vec_id = len(self.data)
        self.data.append(vec)
        
        # 将向量添加到所有哈希表中
        for i in range(self.num_tables):
            h = self._hash(vec, i)
            if h not in self.tables[i]:
                self.tables[i][h] = []
            self.tables[i][h].append(vec_id)
        
        return vec_id
    
    def query(self, vec: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        查询近似最近邻
        
        参数:
            vec: 查询向量
            k: 返回的最近邻数量
            
        返回:
            包含(id, 距离)的列表
        """
        candidates = set()
        
        # 从所有哈希表中收集候选向量
        for i in range(self.num_tables):
            h = self._hash(vec, i)
            if h in self.tables[i]:
                candidates.update(self.tables[i][h])

        print(f"候选向量数量: {len(candidates)}")
        
        # 计算候选向量的距离
        distances = []
        for vec_id in candidates:
            dist = np.linalg.norm(vec - self.data[vec_id])
            distances.append((vec_id, dist))
        
        # 按距离排序并返回前k个
        distances.sort(key=lambda x: x[1])
        return distances[:k]

# 测试用例
if __name__ == "__main__":
    # 创建LSH索引
    dim = 100
    lsh = LSH(dim=dim, num_tables=5, hash_size=8)
    
    # 生成随机数据
    num_vectors = 1000
    data = [np.random.randn(dim) for _ in range(num_vectors)]
    
    # 添加到索引
    for vec in data:
        lsh.add(vec)
    
    # 查询测试
    query_vec = np.random.randn(dim)
    results = lsh.query(query_vec, k=5)
    
    print("近似最近邻搜索结果:")
    for vec_id, dist in results:
        print(f"向量ID: {vec_id}, 距离: {dist:.4f}")
