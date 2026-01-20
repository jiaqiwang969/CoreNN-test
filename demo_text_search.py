#!/usr/bin/env python3
"""
CoreNN 文本搜索演示
使用简单的词袋模型（Bag of Words）创建文本向量，演示语义搜索
"""

import numpy as np
import os
from collections import Counter
from corenn_py import CoreNN

# 示例文档集合
DOCUMENTS = [
    "Python is a high-level programming language",
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing helps computers understand human language",
    "Computer vision enables machines to interpret visual information",
    "Data science combines statistics and programming",
    "Artificial intelligence is transforming many industries",
    "Neural networks are inspired by biological neurons",
    "Python is popular for data science and machine learning",
    "Deep neural networks require large amounts of training data",
    "Natural language understanding is a challenging AI problem",
    "Computer vision applications include face recognition",
    "Machine learning algorithms learn from data",
    "Python has many libraries for scientific computing",
    "Artificial neural networks can solve complex problems",
    "Data preprocessing is crucial for machine learning",
    "Deep learning has achieved breakthrough results in AI",
    "Natural language generation creates human-like text",
    "Computer vision uses convolutional neural networks",
    "Python programming is easy to learn and powerful",
]

def create_vocabulary(documents):
    """创建词汇表"""
    vocab = set()
    for doc in documents:
        words = doc.lower().split()
        vocab.update(words)
    return sorted(list(vocab))

def text_to_vector(text, vocab):
    """将文本转换为词袋向量"""
    words = text.lower().split()
    word_counts = Counter(words)

    # 创建向量
    vector = np.zeros(len(vocab), dtype=np.float32)
    for i, word in enumerate(vocab):
        vector[i] = word_counts.get(word, 0)

    # TF-IDF 风格的归一化（简化版）
    if np.sum(vector) > 0:
        vector = vector / np.linalg.norm(vector)

    return vector

def main():
    print("=" * 70)
    print("CoreNN 文本搜索演示 - 词袋模型")
    print("=" * 70)
    print()

    # 1. 创建词汇表
    print("步骤 1: 构建词汇表...")
    vocab = create_vocabulary(DOCUMENTS)
    print(f"  - 词汇表大小: {len(vocab)} 个单词")
    print(f"  - 向量维度: {len(vocab)}")
    print()

    # 2. 将文档转换为向量
    print("步骤 2: 将文档转换为向量...")
    doc_vectors = []
    for doc in DOCUMENTS:
        vector = text_to_vector(doc, vocab)
        doc_vectors.append(vector)
    doc_vectors = np.array(doc_vectors)
    print(f"  - 已转换 {len(DOCUMENTS)} 个文档")
    print()

    # 3. 创建数据库
    print("步骤 3: 创建 CoreNN 数据库...")
    db_path = "/tmp/corenn_text_search_db"

    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = CoreNN.create(db_path, {"dim": len(vocab)})
    print(f"  - 数据库路径: {db_path}")
    print()

    # 4. 插入文档向量
    print("步骤 4: 插入文档向量...")
    keys = [f"doc_{i}" for i in range(len(DOCUMENTS))]
    db.insert_f32(keys, doc_vectors)
    print(f"  - 已插入 {len(keys)} 个文档向量")
    print()

    # 5. 执行搜索查询
    print("步骤 5: 执行文本搜索查询")
    print("=" * 70)

    queries = [
        "Python programming language",
        "artificial intelligence and neural networks",
        "computer vision and image recognition",
        "data science statistics",
    ]

    for query_text in queries:
        print(f"\n查询: \"{query_text}\"")
        print("-" * 70)

        # 将查询转换为向量
        query_vector = text_to_vector(query_text, vocab)
        query_vector = query_vector.reshape(1, -1)

        # 搜索最相似的文档
        results = db.query_f32(query_vector, 5)

        print("最相似的文档:")
        for i, (doc_key, distance) in enumerate(results[0], 1):
            doc_idx = int(doc_key.split('_')[1])
            similarity = 1 - distance
            print(f"  {i}. 相似度: {similarity:.4f}")
            print(f"     {DOCUMENTS[doc_idx]}")
        print()

    # 6. 交互式搜索
    print("=" * 70)
    print("交互式搜索演示")
    print("=" * 70)
    print()

    # 模拟用户输入
    user_queries = [
        "learning from data",
        "visual recognition systems",
    ]

    for user_query in user_queries:
        print(f"用户查询: \"{user_query}\"")
        query_vector = text_to_vector(user_query, vocab)
        query_vector = query_vector.reshape(1, -1)

        results = db.query_f32(query_vector, 3)

        print("搜索结果:")
        for i, (doc_key, distance) in enumerate(results[0], 1):
            doc_idx = int(doc_key.split('_')[1])
            similarity = 1 - distance
            print(f"  {i}. [{similarity:.3f}] {DOCUMENTS[doc_idx]}")
        print()

    print("=" * 70)
    print("演示完成！")
    print()
    print("注意:")
    print("  - 这个演示使用简单的词袋模型（Bag of Words）")
    print("  - 在实际应用中，建议使用预训练的嵌入模型：")
    print("    • Sentence Transformers")
    print("    • OpenAI Embeddings")
    print("    • BERT/RoBERTa")
    print("  - 这些模型能更好地捕捉语义信息")
    print("=" * 70)

if __name__ == "__main__":
    main()
