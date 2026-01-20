#!/usr/bin/env python3
"""
CoreNN 文件搜索服务器
提供 Web 界面进行文件搜索
"""

from flask import Flask, render_template, request, jsonify
import json
import os
import numpy as np
from corenn_py import CoreNN
import hashlib
from datetime import datetime

app = Flask(__name__)

# 全局变量
db = None
metadata = {}
db_path = "/tmp/corenn_file_index_db"
index_path = "/tmp/corenn_file_index.json"

def load_index():
    """加载索引"""
    global db, metadata

    if not os.path.exists(db_path):
        return False

    print("加载数据库...")
    db = CoreNN.open(db_path)

    print("加载元数据...")
    with open(index_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"✓ 索引加载完成！共 {len(metadata)} 个文件")
    return True

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

def format_size(size):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def format_time(timestamp):
    """格式化时间"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

@app.route('/')
def index():
    """主页"""
    return render_template('index.html', total_files=len(metadata))

@app.route('/search', methods=['POST'])
def search():
    """搜索文件"""
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 50)

    if not query:
        return jsonify({'error': '查询不能为空'}), 400

    # 创建查询向量
    query_vector = create_query_vector(query)
    query_vector = query_vector.reshape(1, -1)

    # 执行搜索
    results = db.query_f32(query_vector, k)

    # 处理结果
    search_results = []
    for file_id, distance in results[0]:
        if file_id in metadata:
            file_info = metadata[file_id]

            # 计算相似度分数
            similarity = 1 - distance

            # 文本匹配分数（简单的字符串匹配）
            query_lower = query.lower()
            name_lower = file_info['name'].lower()
            path_lower = file_info['path'].lower()

            text_score = 0
            if query_lower in name_lower:
                text_score += 0.5
            if query_lower in path_lower:
                text_score += 0.3

            # 综合分数
            final_score = similarity * 0.7 + text_score * 0.3

            search_results.append({
                'name': file_info['name'],
                'path': file_info['path'],
                'directory': file_info['directory'],
                'size': format_size(file_info['size']),
                'size_bytes': file_info['size'],
                'mtime': format_time(file_info['mtime']),
                'extension': file_info['extension'],
                'mime_type': file_info['mime_type'],
                'similarity': f"{similarity:.4f}",
                'score': f"{final_score:.4f}",
            })

    # 按综合分数排序
    search_results.sort(key=lambda x: float(x['score']), reverse=True)

    return jsonify({
        'results': search_results,
        'total': len(search_results),
        'query': query
    })

@app.route('/stats')
def stats():
    """统计信息"""
    # 按扩展名统计
    ext_count = {}
    total_size = 0

    for file_info in metadata.values():
        ext = file_info['extension'] or 'no_ext'
        ext_count[ext] = ext_count.get(ext, 0) + 1
        total_size += file_info['size']

    sorted_exts = sorted(ext_count.items(), key=lambda x: x[1], reverse=True)[:20]

    return jsonify({
        'total_files': len(metadata),
        'total_size': format_size(total_size),
        'extensions': [{'ext': ext, 'count': count} for ext, count in sorted_exts]
    })

@app.route('/similar/<file_id>')
def similar(file_id):
    """查找相似文件"""
    if file_id not in metadata:
        return jsonify({'error': '文件不存在'}), 404

    # 使用该文件的向量查找相似文件
    # 这里简化处理，实际应该重新生成向量
    file_info = metadata[file_id]
    query = file_info['name']

    query_vector = create_query_vector(query)
    query_vector = query_vector.reshape(1, -1)

    results = db.query_f32(query_vector, 20)

    similar_files = []
    for fid, distance in results[0]:
        if fid != file_id and fid in metadata:
            info = metadata[fid]
            similar_files.append({
                'name': info['name'],
                'path': info['path'],
                'similarity': f"{1 - distance:.4f}"
            })

    return jsonify({'similar_files': similar_files})

def main():
    print("=" * 70)
    print("CoreNN 文件搜索服务器")
    print("=" * 70)
    print()

    # 加载索引
    if not load_index():
        print("错误: 索引不存在！")
        print("请先运行 file_indexer.py 创建索引")
        return

    print()
    print("=" * 70)
    print("服务器启动成功！")
    print("=" * 70)
    print()
    print("访问地址: http://localhost:5000")
    print()
    print("按 Ctrl+C 停止服务器")
    print("=" * 70)
    print()

    # 启动服务器
    app.run(host='0.0.0.0', port=5000, debug=False)  # 关闭 debug 模式避免数据库锁定

if __name__ == "__main__":
    main()
