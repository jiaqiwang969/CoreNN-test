#!/usr/bin/env python3
"""
CoreNN 性能基准测试
测试不同数据规模和向量维度下的性能
"""

import numpy as np
import time
import os
from corenn_py import CoreNN

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"

def benchmark_insert(db, num_vectors, dim):
    """测试插入性能"""
    keys = [f"vec_{i}" for i in range(num_vectors)]
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)

    start_time = time.time()
    db.insert_f32(keys, vectors)
    end_time = time.time()

    elapsed = end_time - start_time
    throughput = num_vectors / elapsed

    return elapsed, throughput

def benchmark_query(db, num_queries, dim, k):
    """测试查询性能"""
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    start_time = time.time()
    results = db.query_f32(queries, k)
    end_time = time.time()

    elapsed = end_time - start_time
    avg_query_time = elapsed / num_queries
    qps = num_queries / elapsed

    return elapsed, avg_query_time, qps

def run_benchmark(num_vectors, dim, num_queries=100, k=10):
    """运行完整的基准测试"""
    print(f"\n{'=' * 70}")
    print(f"基准测试: {num_vectors:,} 个向量, 维度 {dim}")
    print(f"{'=' * 70}")

    # 创建数据库
    db_path = f"/tmp/corenn_benchmark_{num_vectors}_{dim}"
    import shutil
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    db = CoreNN.create(db_path, {"dim": dim})

    # 测试插入
    print(f"\n插入测试:")
    insert_time, insert_throughput = benchmark_insert(db, num_vectors, dim)
    print(f"  - 总时间: {format_time(insert_time)}")
    print(f"  - 吞吐量: {insert_throughput:,.0f} 向量/秒")
    print(f"  - 平均每个向量: {format_time(insert_time / num_vectors)}")

    # 测试查询
    print(f"\n查询测试 (k={k}):")
    query_time, avg_query_time, qps = benchmark_query(db, num_queries, dim, k)
    print(f"  - 总时间: {format_time(query_time)}")
    print(f"  - 平均查询时间: {format_time(avg_query_time)}")
    print(f"  - QPS (查询/秒): {qps:,.0f}")

    # 清理
    del db
    shutil.rmtree(db_path)

    return {
        'num_vectors': num_vectors,
        'dim': dim,
        'insert_time': insert_time,
        'insert_throughput': insert_throughput,
        'avg_query_time': avg_query_time,
        'qps': qps
    }

def main():
    print("=" * 70)
    print("CoreNN 性能基准测试")
    print("=" * 70)
    print()
    print("测试配置:")
    print("  - 查询数量: 100")
    print("  - 返回结果数 (k): 10")
    print("  - 数据类型: float32")
    print()

    results = []

    # 测试不同的数据规模
    test_configs = [
        (1000, 128),      # 1K 向量, 128 维
        (10000, 128),     # 10K 向量, 128 维
        (50000, 128),     # 50K 向量, 128 维
        (10000, 256),     # 10K 向量, 256 维
        (10000, 512),     # 10K 向量, 512 维
    ]

    for num_vectors, dim in test_configs:
        result = run_benchmark(num_vectors, dim)
        results.append(result)

    # 显示汇总结果
    print(f"\n{'=' * 70}")
    print("性能汇总")
    print(f"{'=' * 70}")
    print()
    print(f"{'数据规模':<15} {'维度':<8} {'插入吞吐':<15} {'查询QPS':<15} {'查询延迟':<15}")
    print("-" * 70)

    for r in results:
        print(f"{r['num_vectors']:>10,} 个  {r['dim']:>5}    "
              f"{r['insert_throughput']:>10,.0f} v/s  "
              f"{r['qps']:>10,.0f} q/s  "
              f"{format_time(r['avg_query_time']):>12}")

    print()
    print("=" * 70)
    print("测试完成！")
    print()
    print("性能提示:")
    print("  - CoreNN 使用 HNSW 算法，查询时间随数据量对数增长")
    print("  - 更高的维度会增加计算成本")
    print("  - 使用 f16/bf16 可以减少内存使用")
    print("  - 使用 i8 量化可以进一步提升性能")
    print("=" * 70)

if __name__ == "__main__":
    main()
