#!/usr/bin/env python3
"""
改进的向量生成方法 - 使用 N-gram 特征
"""

import numpy as np
import hashlib
from collections import Counter
import re

class ImprovedVectorizer:
    """改进的文件名向量化器"""

    def __init__(self, dim=128, ngram_range=(2, 3)):
        self.dim = dim
        self.ngram_range = ngram_range

    def extract_ngrams(self, text, n):
        """提取 n-gram"""
        text = text.lower()
        # 移除特殊字符，保留字母数字和常见分隔符
        text = re.sub(r'[^\w\s\-_.]', '', text)

        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i:i+n])
        return ngrams

    def text_to_vector(self, text):
        """将文本转换为向量"""
        # 提取不同长度的 n-grams
        all_ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            all_ngrams.extend(self.extract_ngrams(text, n))

        if not all_ngrams:
            # 如果没有 n-grams，使用整个文本的哈希
            text_hash = hashlib.md5(text.lower().encode()).digest()
            features = np.frombuffer(text_hash, dtype=np.uint8).astype(np.float32)
            # 重复到 128 维
            features = np.tile(features, (self.dim // len(features)) + 1)[:self.dim]
            return features / np.linalg.norm(features) if np.linalg.norm(features) > 0 else features

        # 计算 n-gram 频率
        ngram_counts = Counter(all_ngrams)

        # 使用哈希技巧将 n-grams 映射到固定维度
        vector = np.zeros(self.dim, dtype=np.float32)

        for ngram, count in ngram_counts.items():
            # 使用哈希函数将 n-gram 映射到向量的某个位置
            hash_val = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            idx = hash_val % self.dim
            # 累加频率（带符号，避免哈希冲突）
            sign = 1 if (hash_val // self.dim) % 2 == 0 else -1
            vector[idx] += sign * count

        # L2 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def create_file_vector(self, file_path, file_info):
        """为文件创建特征向量 - 128 维"""
        import os

        # 提取文件名（不含路径）
        filename = os.path.basename(file_path)

        # 分离文件名和扩展名
        name_without_ext, ext = os.path.splitext(filename)

        # 1. 文件名向量 (64 维)
        name_vector = self.text_to_vector(name_without_ext)[:64]

        # 2. 扩展名向量 (16 维)
        if ext:
            ext_vector = self.text_to_vector(ext)[:16]
        else:
            ext_vector = np.zeros(16, dtype=np.float32)

        # 3. 目录名向量 (32 维)
        dir_name = os.path.basename(os.path.dirname(file_path))
        if dir_name:
            dir_vector = self.text_to_vector(dir_name)[:32]
        else:
            dir_vector = np.zeros(32, dtype=np.float32)

        # 4. 文件大小特征 (8 维)
        size = file_info.get('size', 0)
        size_feature = np.log1p(size)
        size_vector = np.array([size_feature] * 8, dtype=np.float32)

        # 5. 时间特征 (8 维)
        mtime = file_info.get('mtime', 0)
        time_feature = mtime / 1e9
        time_vector = np.array([time_feature] * 8, dtype=np.float32)

        # 组合所有特征
        vector = np.concatenate([
            name_vector,      # 64 维
            ext_vector,       # 16 维
            dir_vector,       # 32 维
            size_vector,      # 8 维
            time_vector       # 8 维
        ])  # 总共 128 维

        # 最终归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def create_query_vector(self, query_text):
        """为查询创建向量"""
        # 查询主要关注文件名匹配
        # 分配更多维度给查询文本本身

        # 1. 完整查询向量 (80 维)
        query_vector = self.text_to_vector(query_text)[:80]

        # 2. 如果包含扩展名，提取扩展名向量 (16 维)
        if '.' in query_text:
            parts = query_text.split('.')
            ext = '.' + parts[-1]
            ext_vector = self.text_to_vector(ext)[:16]
        else:
            ext_vector = np.zeros(16, dtype=np.float32)

        # 3. 如果包含路径，提取路径向量 (32 维)
        if '/' in query_text or '\\' in query_text:
            path_parts = re.split(r'[/\\]', query_text)
            path_text = ' '.join(path_parts)
            path_vector = self.text_to_vector(path_text)[:32]
        else:
            path_vector = np.zeros(32, dtype=np.float32)

        # 组合向量
        vector = np.concatenate([
            query_vector,     # 80 维
            ext_vector,       # 16 维
            path_vector       # 32 维
        ])  # 总共 128 维

        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector


def test_vectorizer():
    """测试向量化器"""
    vectorizer = ImprovedVectorizer()

    # 测试文件名
    test_files = [
        "config.json",
        "configuration.json",
        "config.yaml",
        "my_config.json",
        "test_config.json"
    ]

    print("测试 N-gram 向量化器")
    print("=" * 60)

    query = "config.json"
    query_vec = vectorizer.create_query_vector(query)

    print(f"\n查询: {query}")
    print(f"查询向量维度: {len(query_vec)}")
    print(f"查询向量范数: {np.linalg.norm(query_vec):.4f}")

    print("\n相似度排名:")
    similarities = []
    for filename in test_files:
        file_info = {'size': 1000, 'mtime': 1234567890}
        file_vec = vectorizer.text_to_vector(filename)[:128]

        # 计算余弦相似度
        similarity = np.dot(query_vec, file_vec)
        similarities.append((filename, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    for i, (filename, sim) in enumerate(similarities, 1):
        print(f"{i}. {filename:30s} - 相似度: {sim:.4f}")


if __name__ == "__main__":
    test_vectorizer()
