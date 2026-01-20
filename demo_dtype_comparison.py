#!/usr/bin/env python3
"""
CoreNN 数据类型比较演示
比较不同数据类型（f32, f16, bf16）的性能和精度
"""

import numpy as np
import time
import os
from corenn_py import CoreNN

def generate_test_data(num_vectors, dim):
    """生成测试数据"""
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    # 归一化
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors

def test_dtype_performance(vectors, dtype_name, db_path):
    """测试特定数据类型的性能"""
    num_vectors, dim = vectors.shape

    # 创建数据库
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = CoreNN.create(db_path, {"dim": dim})

    # 准备数据
    keys = [f"vec_{i}" for i in range(num_vectors)]

    # 转换数据类型并插入
    start_time = time.time()
    if dtype_name == "f32":
        db.insert_f32(keys, vectors)
    elif dtype_name == "f16":
        vectors_f16 = vectors.astype(np.float16)
        db.insert_f16(keys, vectors_f16)
    elif dtype_name == "bf16":
        # NumPy 不直接支持 bfloat16，使用 float32
        # 在实际应用中会由库处理转换
        db.insert_f32(keys, vectors)  # CoreNN 内部会处理
    insert_time = time.time() - start_time

    # 查询测试
    query = vectors[0:1]
    start_time = time.time()
    if dtype_name == "f32":
        results = db.query_f32(query, 10)
    elif dtype_name == "f16":
        query_f16 = query.astype(np.float16)
        results = db.query_f16(query_f16, 10)
    elif dtype_name == "bf16":
        results = db.query_f32(query, 10)
    query_time = time.time() - start_time

    # 清理
    del db
    shutil.rmtree(db_path)

    return insert_time, query_time, results

def main():
    print("=" * 70)
    print("CoreNN 数据类型比较演示")
    print("=" * 70)
    print()

    # 生成测试数据
    num_vectors = 10000
    dim = 128

    print(f"生成测试数据: {num_vectors:,} 个向量, 维度 {dim}")
    vectors = generate_test_data(num_vectors, dim)
    print()

    # 测试不同的数据类型
    dtypes = ["f32", "f16"]  # bf16 需要特殊处理
    results = {}

    for dtype in dtypes:
        print(f"测试 {dtype.upper()} 数据类型...")
        db_path = f"/tmp/corenn_dtype_{dtype}"

        insert_time, query_time, query_results = test_dtype_performance(
            vectors, dtype, db_path
        )

        results[dtype] = {
            'insert_time': insert_time,
            'query_time': query_time,
            'results': query_results
        }

        print(f"  - 插入时间: {insert_time:.3f} 秒")
        print(f"  - 查询时间: {query_time * 1000:.2f} 毫秒")
        print()

    # 比较结果
    print("=" * 70)
    print("性能比较")
    print("=" * 70)
    print()

    print(f"{'数据类型':<12} {'插入时间':<15} {'查询时间':<15} {'内存占用':<15}")
    print("-" * 70)

    memory_usage = {
        'f32': num_vectors * dim * 4,  # 4 bytes per float32
        'f16': num_vectors * dim * 2,  # 2 bytes per float16
        'bf16': num_vectors * dim * 2, # 2 bytes per bfloat16
    }

    for dtype in dtypes:
        r = results[dtype]
        mem_mb = memory_usage[dtype] / (1024 * 1024)
        print(f"{dtype.upper():<12} {r['insert_time']:>10.3f} s    "
              f"{r['query_time'] * 1000:>10.2f} ms    "
              f"{mem_mb:>10.2f} MB")

    print()

    # 精度比较
    print("=" * 70)
    print("精度比较 (查询结果)")
    print("=" * 70)
    print()

    print("F32 结果:")
    for i, (key, dist) in enumerate(results['f32']['results'][0][:5], 1):
        print(f"  {i}. {key}: 距离 = {dist:.6f}")

    print()
    print("F16 结果:")
    for i, (key, dist) in enumerate(results['f16']['results'][0][:5], 1):
        print(f"  {i}. {key}: 距离 = {dist:.6f}")

    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print()
    print("数据类型选择建议:")
    print()
    print("  F32 (Float32):")
    print("    ✓ 最高精度")
    print("    ✓ 适合需要高精度的应用")
    print("    ✗ 内存占用最大")
    print()
    print("  F16 (Float16):")
    print("    ✓ 内存占用减半")
    print("    ✓ 精度通常足够")
    print("    ✓ 在某些硬件上更快")
    print("    ✗ 精度略有损失")
    print()
    print("  BF16 (BFloat16):")
    print("    ✓ 内存占用减半")
    print("    ✓ 动态范围与 F32 相同")
    print("    ✓ 适合深度学习应用")
    print("    ✗ 精度低于 F16")
    print()
    print("  I8 (Int8 量化):")
    print("    ✓ 内存占用最小 (1/4)")
    print("    ✓ 查询速度最快")
    print("    ✓ 适合大规模部署")
    print("    ✗ 需要量化参数 (scale, zero_point)")
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
