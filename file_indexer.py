#!/usr/bin/env python3
"""
CoreNN 文件索引器
扫描文件系统并创建向量索引
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from corenn_py import CoreNN

class FileIndexer:
    def __init__(self, db_path, index_path):
        self.db_path = db_path
        self.index_path = index_path
        self.metadata = {}

    def create_file_vector(self, file_path, file_info):
        """为文件创建特征向量 - 128 维"""
        features = []

        # 1. 文件名特征 (32 维) - MD5 产生 16 字节，重复两次
        name = os.path.basename(file_path).lower()
        name_hash = hashlib.md5(name.encode()).digest()
        name_features = np.frombuffer(name_hash, dtype=np.uint8).astype(np.float32)
        features.extend(name_features)  # 16 维
        features.extend(name_features)  # 再 16 维，共 32 维

        # 2. 扩展名特征 (16 维)
        ext = os.path.splitext(file_path)[1].lower()
        ext_hash = hashlib.md5(ext.encode()).digest()
        ext_features = np.frombuffer(ext_hash, dtype=np.uint8).astype(np.float32)
        features.extend(ext_features)  # 16 维

        # 3. 目录路径特征 (32 维) - 重复两次
        dir_path = os.path.dirname(file_path).lower()
        dir_hash = hashlib.md5(dir_path.encode()).digest()
        dir_features = np.frombuffer(dir_hash, dtype=np.uint8).astype(np.float32)
        features.extend(dir_features)  # 16 维
        features.extend(dir_features)  # 再 16 维，共 32 维

        # 4. 文件大小特征 (16 维)
        size = file_info.get('size', 0)
        size_feature = np.log1p(size)
        features.extend([size_feature] * 16)

        # 5. 修改时间特征 (16 维)
        mtime = file_info.get('mtime', 0)
        time_feature = mtime / 1e9  # 归一化
        features.extend([time_feature] * 16)

        # 6. MIME 类型特征 (16 维)
        mime_type = file_info.get('mime_type', 'unknown')
        mime_hash = hashlib.md5(mime_type.encode()).digest()
        mime_features = np.frombuffer(mime_hash, dtype=np.uint8).astype(np.float32)
        features.extend(mime_features)  # 16 维

        # 确保正好是 128 维
        vector = np.array(features[:128], dtype=np.float32)

        # 归一化
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)

        return vector

    def get_file_info(self, file_path):
        """获取文件信息"""
        try:
            stat = os.stat(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)

            return {
                'path': file_path,
                'name': os.path.basename(file_path),
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'ctime': stat.st_ctime,
                'mime_type': mime_type or 'unknown',
                'extension': os.path.splitext(file_path)[1],
                'directory': os.path.dirname(file_path),
            }
        except Exception as e:
            return None

    def scan_directory(self, root_dir, max_files=None, exclude_dirs=None):
        """扫描目录并收集文件信息"""
        if exclude_dirs is None:
            exclude_dirs = {
                '.git', '.svn', 'node_modules', '__pycache__',
                '.venv', 'venv', '.cache', 'Library', 'Applications',
                '.Trash', '.npm', '.cargo', 'Cache', 'Caches',
                '.local/share/Trash', 'VirtualBox VMs'
            }

        files = []
        count = 0

        print(f"开始扫描目录: {root_dir}")
        print(f"排除目录: {exclude_dirs}")
        print()

        for root, dirs, filenames in os.walk(root_dir):
            # 过滤排除的目录
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

            for filename in filenames:
                if max_files and count >= max_files:
                    print(f"\n已达到最大文件数限制: {max_files}")
                    return files

                # 跳过隐藏文件
                if filename.startswith('.'):
                    continue

                file_path = os.path.join(root, filename)

                # 跳过符号链接
                if os.path.islink(file_path):
                    continue

                file_info = self.get_file_info(file_path)
                if file_info:
                    files.append(file_info)
                    count += 1

                    if count % 1000 == 0:
                        print(f"\r已扫描: {count:,} 个文件...", end='', flush=True)

        print(f"\n扫描完成！共找到 {len(files):,} 个文件")
        return files

    def build_index(self, files):
        """构建向量索引"""
        print(f"\n开始构建索引...")

        # 创建数据库
        if os.path.exists(self.db_path):
            import shutil
            shutil.rmtree(self.db_path)

        db = CoreNN.create(self.db_path, {"dim": 128})

        # 准备数据
        keys = []
        vectors = []

        for i, file_info in enumerate(files):
            file_id = f"file_{i}"
            vector = self.create_file_vector(file_info['path'], file_info)

            keys.append(file_id)
            vectors.append(vector)
            self.metadata[file_id] = file_info

            if (i + 1) % 100 == 0:
                print(f"\r已处理: {i + 1}/{len(files)} 个文件...", end='', flush=True)

        print(f"\n插入向量到数据库...")
        vectors_array = np.array(vectors, dtype=np.float32)
        db.insert_f32(keys, vectors_array)

        # 保存元数据
        print(f"保存元数据...")
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        print(f"✓ 索引构建完成！")
        print(f"  - 数据库路径: {self.db_path}")
        print(f"  - 元数据路径: {self.index_path}")
        print(f"  - 索引文件数: {len(files)}")

        return db

def main():
    print("=" * 70)
    print("CoreNN 文件索引器 - 完整扫描模式")
    print("=" * 70)
    print()

    # 配置
    home_dir = str(Path.home())
    db_path = "/tmp/corenn_file_index_db"
    index_path = "/tmp/corenn_file_index.json"

    # 创建索引器
    indexer = FileIndexer(db_path, index_path)

    # 扫描整个主目录，不设置文件数量限制
    print(f"⚠️  警告: 将扫描整个主目录 {home_dir}")
    print(f"⚠️  这可能需要较长时间，取决于文件数量")
    print()

    all_files = indexer.scan_directory(home_dir, max_files=None)

    if not all_files:
        print("未找到任何文件！")
        return

    print(f"\n总共找到 {len(all_files):,} 个文件")

    # 构建索引
    db = indexer.build_index(all_files)

    # 显示统计信息
    print("\n" + "=" * 70)
    print("索引统计")
    print("=" * 70)

    # 按扩展名统计
    ext_count = {}
    for file_info in all_files:
        ext = file_info['extension'] or 'no_ext'
        ext_count[ext] = ext_count.get(ext, 0) + 1

    print("\n文件类型分布 (Top 20):")
    sorted_exts = sorted(ext_count.items(), key=lambda x: x[1], reverse=True)[:20]
    for ext, count in sorted_exts:
        print(f"  {ext:20s}: {count:7,} 个文件")

    # 总大小
    total_size = sum(f['size'] for f in all_files)
    print(f"\n总文件大小: {total_size / (1024**3):.2f} GB")
    print(f"平均文件大小: {(total_size / len(all_files)) / 1024:.2f} KB")

    print("\n" + "=" * 70)
    print("索引创建完成！现在可以运行 file_search_server.py 启动搜索服务")
    print("=" * 70)

if __name__ == "__main__":
    main()
