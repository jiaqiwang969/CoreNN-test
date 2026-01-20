#!/usr/bin/env python3
"""
CoreNN Demo - Vector Database Example
使用本地文件系统数据进行向量搜索演示
"""

import numpy as np
import os
from pathlib import Path
from corenn_py import CoreNN

def create_sample_vectors():
    """创建示例向量数据，模拟文件嵌入"""
    # 获取用户主目录下的一些文件
    home = Path.home()

    # 收集一些文件路径作为示例
    file_paths = []
    for root, dirs, files in os.walk(home):
        # 只搜索第一层目录，避免太深
        if root == str(home):
            for file in files[:20]:  # 限制文件数量
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    file_paths.append(file_path)
        # 不递归到子目录
        break

    print(f"找到 {len(file_paths)} 个文件用于演示")

    # 为每个文件生成一个随机向量（模拟文件的嵌入表示）
    # 在实际应用中，这些向量会由嵌入模型生成
    dim = 128  # 向量维度
    vectors = []
    keys = []

    for i, file_path in enumerate(file_paths):
        # 使用文件名作为种子，生成可重复的随机向量
        np.random.seed(hash(file_path) % (2**32))
        vector = np.random.randn(dim).astype(np.float32)
        # 归一化向量
        vector = vector / np.linalg.norm(vector)

        vectors.append(vector)
        keys.append(file_path)

    return keys, np.array(vectors)

def main():
    print("=" * 60)
    print("CoreNN 向量数据库演示")
    print("=" * 60)
    print()

    # 1. 创建示例数据
    print("步骤 1: 创建示例向量数据...")
    keys, vectors = create_sample_vectors()

    if len(keys) == 0:
        print("警告: 没有找到文件，创建一些示例数据...")
        # 如果没有找到文件，创建一些示例数据
        keys = [f"document_{i}" for i in range(100)]
        vectors = np.random.randn(100, 128).astype(np.float32)
        # 归一化
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    print(f"  - 数据条目数: {len(keys)}")
    print(f"  - 向量维度: {vectors.shape[1]}")
    print()

    # 2. 创建数据库
    print("步骤 2: 创建 CoreNN 数据库...")
    db_path = "/tmp/corenn_demo_db"

    # 如果数据库已存在，删除它
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = CoreNN.create(db_path, {
        "dim": vectors.shape[1],
    })
    print(f"  - 数据库路径: {db_path}")
    print()

    # 3. 插入向量
    print("步骤 3: 插入向量到数据库...")
    db.insert_f32(keys, vectors)
    print(f"  - 已插入 {len(keys)} 个向量")
    print()

    # 4. 执行查询
    print("步骤 4: 执行向量相似度搜索...")

    # 使用第一个向量作为查询
    query_key = keys[0]
    query_vector = vectors[0:1]  # 保持2D形状

    print(f"  - 查询向量: {query_key}")
    print(f"  - 查找最相似的 5 个向量...")
    print()

    # 查询最相似的5个向量
    results = db.query_f32(query_vector, 5)

    # 5. 显示结果
    print("步骤 5: 搜索结果")
    print("-" * 60)
    for i, (result_key, distance) in enumerate(results[0], 1):
        similarity = 1 - distance  # 距离越小，相似度越高
        print(f"{i}. 相似度: {similarity:.4f}")
        print(f"   文件: {result_key}")
        print()

    print("=" * 60)
    print("演示完成！")
    print()
    print("CoreNN 特性:")
    print("  ✓ 支持数十亿级向量的快速搜索")
    print("  ✓ 亚线性时间复杂度")
    print("  ✓ 支持多种数据类型 (f32, f16, bf16, i8)")
    print("  ✓ 持久化存储")
    print("=" * 60)

    # 6. 演示重新打开数据库
    print()
    print("步骤 6: 演示数据库持久化...")
    print("  - 关闭并重新打开数据库...")

    # 显式删除第一个数据库实例以释放锁
    del db

    # 重新打开数据库
    db2 = CoreNN.open(db_path)

    # 再次查询以验证数据已持久化
    results2 = db2.query_f32(query_vector, 3)
    print(f"  - 重新打开后查询成功，找到 {len(results2[0])} 个结果")
    print("  ✓ 数据已成功持久化")
    print()

if __name__ == "__main__":
    main()
