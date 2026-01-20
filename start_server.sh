#!/bin/bash
# 启动 CoreNN 文件搜索服务器

echo "========================================"
echo "CoreNN 文件搜索系统"
echo "========================================"
echo ""
echo "正在启动服务器..."
echo ""

# 取消 CONDA_PREFIX
unset CONDA_PREFIX

# 启动服务器
./venv/bin/python file_search_server.py
