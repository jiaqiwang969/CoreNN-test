#!/usr/bin/env python3
"""
测试文件搜索系统
"""

import json
import numpy as np
from corenn_py import CoreNN
import hashlib

def create_query_vector(query_text):
    """为查询文本创建向量 - 必须是 128 维"""
    features = []

    # 1. 查询文本特征 (32 维) - MD5 产生 16 字节，重复两次
    query_lower = query_text.lower()
    query_hash = hashlib.md5(query_lower.encode()).digest()
    query_features = np.frombuffer(query_hash, dtype=np.uint8).astype(np.float32)
    features.extend(query_features)  # 16 维
    features.extend(query_features)  # 再 16 维，共 32 维

    # 2. 扩展名特征 (16 维)
    if '.' in query_text:
        parts = query_text.split('.')
        ext = '.' + parts[-1].lower()
        ext_hash = hashlib.md5(ext.encode()).digest()
        ext_features = np.frombuffer(ext_hash, dtype=np.uint8).astype(np.float32)
        features.extend(ext_features)  # 16 维
    else:
        features.extend([0.0] * 16)

    # 3. 路径特征 (32 维) - 重复两次
    if '/' in query_text or '\\' in query_text:
        path_hash = hashlib.md5(query_text.encode()).digest()
        path_features = np.frombuffer(path_hash, dtype=np.uint8).astype(np.float32)
        features.extend(path_features)  # 16 维
        features.extend(path_features)  # 再 16 维，共 32 维
    else:
        features.extend([0.0] * 32)

    # 4. 填充剩余维度到 128 维 (48 维)
    while len(features) < 128:
        features.append(0.0)

    # 确保正好是 128 维
    vector = np.array(features[:128], dtype=np.float32)

    # 归一化
    if np.linalg.norm(vector) > 0:
        vector = vector / np.linalg.norm(vector)

    return vector

def test_search(query, k=10):
    """测试搜索功能"""
    print(f"\n{'='*70}")
    print(f"搜索: \"{query}\"")
    print(f"{'='*70}")

    # 加载数据库
    db_path = "/tmp/corenn_file_index_db"
    index_path = "/tmp/corenn_file_index.json"

    db = CoreNN.open(db_path)

    with open(index_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # 创建查询向量
    query_vector = create_query_vector(query)
    query_vector = query_vector.reshape(1, -1)

    # 执行搜索
    results = db.query_f32(query_vector, k)

    # 显示结果
    print(f"\n找到 {len(results[0])} 个结果:\n")

    for i, (file_id, distance) in enumerate(results[0], 1):
        if file_id in metadata:
            file_info = metadata[file_id]
            similarity = 1 - distance

            # 文本匹配
            query_lower = query.lower()
            name_lower = file_info['name'].lower()
            path_lower = file_info['path'].lower()

            text_score = 0
            if query_lower in name_lower:
                text_score += 0.5
            if query_lower in path_lower:
                text_score += 0.3

            final_score = similarity * 0.7 + text_score * 0.3

            print(f"{i}. [{final_score:.4f}] {file_info['name']}")
            print(f"   路径: {file_info['path']}")
            print(f"   大小: {file_info['size'] / 1024:.2f} KB")
            print(f"   类型: {file_info['extension']}")
            print()

def main():
    print("=" * 70)
    print("CoreNN 文件搜索系统 - 测试脚本")
    print("=" * 70)

    # 测试不同的查询
    test_queries = [
        ".py",           # Python 文件
        ".dcm",          # 医学影像
        ".txt",          # 文本文件
        "config",        # 配置文件
        ".jpg",          # 图片
    ]

    for query in test_queries:
        test_search(query, k=5)

    print("=" * 70)
    print("测试完成！")
    print("=" * 70)
    print()
    print("现在可以启动 Web 服务器:")
    print("  ./start_server.sh")
    print()
    print("然后访问: http://localhost:5000")
    print("=" * 70)

if __name__ == "__main__":
    main()
