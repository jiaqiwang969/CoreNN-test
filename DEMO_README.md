# CoreNN 演示说明

## 项目简介

CoreNN 是一个高性能向量数据库，能够在普通硬件上以亚线性时间复杂度查询数十亿级向量。

## 已完成的设置

1. ✅ 创建了 Python 虚拟环境
2. ✅ 安装了必要的依赖（maturin, numpy）
3. ✅ 成功构建了 CoreNN Python 包
4. ✅ 创建并运行了演示脚本

## 运行演示

```bash
# 激活虚拟环境并运行演示
./venv/bin/python demo.py
```

## 演示内容

演示脚本 `demo.py` 展示了以下功能：

1. **数据准备**：从你的主目录收集文件，为每个文件生成一个 128 维的向量（模拟文件嵌入）
2. **创建数据库**：在 `/tmp/corenn_demo_db` 创建一个新的 CoreNN 数据库
3. **插入向量**：将所有文件向量插入数据库
4. **相似度搜索**：查询与某个文件最相似的其他文件
5. **持久化验证**：关闭并重新打开数据库，验证数据已正确保存

## 演示结果

演示成功找到了 19 个文件，并展示了向量相似度搜索的结果。例如：

```
步骤 5: 搜索结果
------------------------------------------------------------
1. 相似度: 1.0000
   文件: /Users/jqwang/.claude.json.backup

2. 相似度: -0.5827
   文件: /Users/jqwang/.gmshrc

3. 相似度: -0.6167
   文件: /Users/jqwang/TH管道声腔特性测试方法研究.zip
...
```

## CoreNN 主要特性

- ✅ **高性能**：支持数十亿级向量的快速搜索
- ✅ **亚线性复杂度**：查询时间随数据量增长缓慢
- ✅ **多种数据类型**：支持 f32, f16, bf16, i8（量化）
- ✅ **持久化存储**：数据可以保存到磁盘并重新加载
- ✅ **并行处理**：自动利用多核 CPU 加速

## 自定义使用

你可以修改 `demo.py` 来：

1. 使用不同的向量维度
2. 搜索不同的目录
3. 调整查询返回的结果数量
4. 使用不同的数据类型（f16, bf16, i8）

## API 示例

```python
from corenn_py import CoreNN
import numpy as np

# 创建数据库
db = CoreNN.create("/path/to/db", {"dim": 128})

# 插入向量
keys = ["doc1", "doc2", "doc3"]
vectors = np.random.randn(3, 128).astype(np.float32)
db.insert_f32(keys, vectors)

# 查询
query = np.random.randn(1, 128).astype(np.float32)
results = db.query_f32(query, k=10)  # 返回最相似的 10 个结果

# 重新打开数据库
db = CoreNN.open("/path/to/db")
```

## 技术细节

- **构建工具**：使用 maturin 构建 Rust + Python 混合项目
- **Python 版本**：3.14（使用 ABI3 前向兼容模式）
- **向量归一化**：演示中使用了 L2 归一化
- **距离度量**：使用欧氏距离（可以转换为余弦相似度）

## 下一步

你可以：

1. 集成真实的嵌入模型（如 OpenAI embeddings, Sentence Transformers）
2. 构建文档搜索系统
3. 实现图像相似度搜索
4. 创建推荐系统
5. 尝试不同的向量维度和数据类型以优化性能

## 问题排查

如果遇到问题：

1. 确保虚拟环境已激活
2. 检查 Python 版本（需要 3.7+）
3. 确保 Rust 工具链已安装
4. 查看 CoreNN 的官方文档：https://github.com/wilsonzlin/corenn
