# CoreNN 完整演示指南

## 概述

本项目包含多个演示脚本，展示 CoreNN 向量数据库的各种功能和应用场景。

## 环境设置

已完成的环境配置：
- ✅ Python 虚拟环境 (venv)
- ✅ CoreNN Python 包 (v0.3.1)
- ✅ 依赖包：numpy, maturin

## 演示脚本列表

### 1. demo.py - 基础演示
**功能**: 使用本地文件系统数据演示基本的向量数据库操作

**运行**:
```bash
./venv/bin/python demo.py
```

**演示内容**:
- 从主目录收集文件
- 为每个文件生成向量（模拟嵌入）
- 创建数据库并插入向量
- 执行相似度搜索
- 验证数据持久化

**输出示例**:
```
找到 19 个文件用于演示
数据条目数: 19
向量维度: 128

搜索结果:
1. 相似度: 1.0000
   文件: /Users/jqwang/.claude.json.backup
```

---

### 2. demo_text_search.py - 文本搜索演示
**功能**: 使用词袋模型实现文本语义搜索

**运行**:
```bash
./venv/bin/python demo_text_search.py
```

**演示内容**:
- 构建词汇表（84 个单词）
- 将 20 个文档转换为向量
- 执行多个搜索查询
- 展示交互式搜索

**查询示例**:
```
查询: "Python programming language"
最相似的文档:
  1. 相似度: 0.4142
     Python is a high-level programming language
  2. 相似度: -0.1835
     Python programming is easy to learn and powerful
```

**应用场景**:
- 文档检索系统
- 问答系统
- 内容推荐
- 语义搜索引擎

---

### 3. demo_benchmark.py - 性能基准测试
**功能**: 测试不同数据规模和维度下的性能

**运行**:
```bash
./venv/bin/python demo_benchmark.py
```

**测试配置**:
- 1,000 向量 @ 128 维
- 10,000 向量 @ 128 维
- 50,000 向量 @ 128 维
- 10,000 向量 @ 256 维
- 10,000 向量 @ 512 维

**性能结果**:
```
数据规模            维度       插入吞吐            查询QPS           查询延迟
----------------------------------------------------------------------
     1,000 个    128        14,911 v/s      15,572 q/s      64.22 μs
    10,000 个    128         3,212 v/s       4,271 q/s     234.14 μs
    50,000 个    128         1,776 v/s       2,809 q/s     356.00 μs
```

**关键发现**:
- 查询时间随数据量对数增长（HNSW 算法特性）
- 50K 向量仍能保持 2,809 QPS
- 维度增加会影响性能，但影响可控

---

### 4. demo_dtype_comparison.py - 数据类型比较
**功能**: 比较 F32 和 F16 的性能和精度

**运行**:
```bash
./venv/bin/python demo_dtype_comparison.py
```

**测试数据**: 10,000 向量 @ 128 维

**比较结果**:
```
数据类型         插入时间            查询时间            内存占用
----------------------------------------------------------------------
F32               4.675 s          2.29 ms          4.88 MB
F16               5.829 s          2.14 ms          2.44 MB
```

**精度对比**:
- F32 和 F16 的查询结果几乎相同
- 距离差异在 0.0001 以内
- F16 内存占用减半

**选择建议**:
- **F32**: 需要最高精度的应用
- **F16**: 大多数应用的最佳选择（内存减半，精度足够）
- **BF16**: 深度学习应用（动态范围大）
- **I8**: 超大规模部署（内存最小，速度最快）

---

## 性能总结

### 插入性能
- 小数据集 (1K): ~15K 向量/秒
- 中等数据集 (10K): ~3K 向量/秒
- 大数据集 (50K): ~1.8K 向量/秒

### 查询性能
- 小数据集 (1K): ~15K QPS, 64μs 延迟
- 中等数据集 (10K): ~4K QPS, 234μs 延迟
- 大数据集 (50K): ~2.8K QPS, 356μs 延迟

### 可扩展性
- ✅ 亚线性查询时间（对数增长）
- ✅ 支持数十亿级向量
- ✅ 并行插入和查询
- ✅ 持久化存储

---

## 实际应用场景

### 1. 文档搜索系统
```python
# 使用预训练模型生成嵌入
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# 生成文档嵌入
embeddings = model.encode(documents)

# 插入 CoreNN
db.insert_f32(doc_ids, embeddings)

# 搜索
query_embedding = model.encode([query])
results = db.query_f32(query_embedding, k=10)
```

### 2. 图像相似度搜索
```python
# 使用 ResNet 或 CLIP 提取图像特征
from torchvision.models import resnet50
model = resnet50(pretrained=True)

# 提取特征向量
features = extract_features(images, model)

# 插入 CoreNN
db.insert_f32(image_ids, features)

# 查找相似图像
query_features = extract_features([query_image], model)
similar_images = db.query_f32(query_features, k=20)
```

### 3. 推荐系统
```python
# 用户/物品嵌入
user_embeddings = generate_user_embeddings(users)
item_embeddings = generate_item_embeddings(items)

# 插入物品向量
db.insert_f32(item_ids, item_embeddings)

# 为用户推荐物品
recommendations = db.query_f32(user_embeddings, k=50)
```

---

## 优化建议

### 内存优化
1. **使用 F16**: 内存减半，精度损失可忽略
2. **使用 I8 量化**: 内存降至 1/4，适合超大规模
3. **批量插入**: 利用并行处理提升吞吐量

### 性能优化
1. **选择合适的维度**: 128-512 维通常是最佳平衡
2. **调整 k 值**: 只返回需要的结果数量
3. **使用 SSD**: 持久化存储性能更好

### 精度优化
1. **向量归一化**: 提升余弦相似度计算准确性
2. **使用 F32**: 对精度要求极高的场景
3. **定期重建索引**: 保持最佳查询性能

---

## 与其他向量数据库对比

| 特性 | CoreNN | Faiss | Milvus | Pinecone |
|------|--------|-------|--------|----------|
| 规模 | 数十亿 | 数十亿 | 数十亿 | 数十亿 |
| 部署 | 本地 | 本地 | 分布式 | 云服务 |
| 持久化 | ✅ | ❌ | ✅ | ✅ |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 性能 | 优秀 | 优秀 | 优秀 | 优秀 |
| 成本 | 免费 | 免费 | 免费/付费 | 付费 |

**CoreNN 优势**:
- 简单易用，单机部署
- 内置持久化，无需额外配置
- 支持多种数据类型
- 开源免费

---

## 下一步探索

### 1. 集成真实嵌入模型
```bash
pip install sentence-transformers
```

### 2. 构建 Web API
```python
from flask import Flask, request
app = Flask(__name__)

@app.route('/search', methods=['POST'])
def search():
    query = request.json['query']
    embedding = model.encode([query])
    results = db.query_f32(embedding, k=10)
    return {'results': results}
```

### 3. 添加过滤功能
- 元数据过滤
- 时间范围过滤
- 类别过滤

### 4. 监控和分析
- 查询延迟监控
- 吞吐量统计
- 内存使用分析

---

## 常见问题

### Q: 如何选择向量维度？
A: 128-512 维是常见选择。更高维度提供更多信息，但增加计算成本。

### Q: 数据库可以存储多少向量？
A: CoreNN 设计用于数十亿级向量。实际限制取决于可用磁盘空间。

### Q: 如何提升查询速度？
A: 使用 F16/I8、减少 k 值、使用 SSD、增加内存。

### Q: 支持实时更新吗？
A: 支持。可以随时插入新向量，无需重建整个索引。

### Q: 如何备份数据？
A: 直接复制数据库目录即可。

---

## 技术支持

- GitHub: https://github.com/wilsonzlin/corenn
- 博客文章: https://blog.wilsonl.in/corenn/
- 问题反馈: GitHub Issues

---

## 总结

CoreNN 是一个强大、易用的向量数据库，适合：
- ✅ 需要本地部署的应用
- ✅ 中小规模到大规模的向量搜索
- ✅ 需要持久化存储的场景
- ✅ 对性能和成本敏感的项目

通过本演示项目，你已经掌握了 CoreNN 的核心功能。现在可以开始构建自己的向量搜索应用了！
