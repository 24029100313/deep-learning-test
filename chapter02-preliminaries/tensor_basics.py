#!/usr/bin/env python3
"""
第二章：张量操作基础
深度学习预备知识 - 张量操作练习
"""

import torch
import numpy as np


def basic_tensor_operations():
    """基础张量操作演示"""
    print("=== 基础张量操作 ===")
    
    # 创建张量
    x = torch.arange(12)
    print(f"创建张量: {x}")
    print(f"形状: {x.shape}")
    print(f"元素个数: {x.numel()}")
    
    # 改变形状
    X = x.reshape(3, 4)
    print(f"重塑为3x4:\n{X}")
    
    # 创建特殊张量
    zeros = torch.zeros(2, 3, 4)
    ones = torch.ones(2, 3, 4)
    randn = torch.randn(3, 4)
    
    print(f"全零张量形状: {zeros.shape}")
    print(f"随机张量:\n{randn}")
    
    return X


def tensor_arithmetic():
    """张量运算演示"""
    print("\n=== 张量运算 ===")
    
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2.0, 2, 2, 2])
    
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"x + y: {x + y}")
    print(f"x - y: {x - y}")
    print(f"x * y: {x * y}")
    print(f"x / y: {x / y}")
    print(f"x ** y: {x ** y}")
    
    # 矩阵运算
    A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
    
    print(f"A:\n{A}")
    print(f"A + B:\n{A + B}")
    print(f"A * B (逐元素乘法):\n{A * B}")


def broadcasting_demo():
    """广播机制演示"""
    print("\n=== 广播机制 ===")
    
    a = torch.arange(3).reshape(3, 1)
    b = torch.arange(2).reshape(1, 2)
    
    print(f"a ({a.shape}):\n{a}")
    print(f"b ({b.shape}):\n{b}")
    print(f"a + b ({(a + b).shape}):\n{a + b}")


def indexing_and_slicing():
    """索引和切片演示"""
    print("\n=== 索引和切片 ===")
    
    X = torch.arange(12).reshape(3, 4)
    print(f"原始张量:\n{X}")
    
    # 索引
    print(f"X[-1]: {X[-1]}")  # 最后一行
    print(f"X[1:3]: \n{X[1:3]}")  # 第2到第3行
    
    # 写入元素
    X[1, 2] = 9
    print(f"修改后:\n{X}")
    
    # 多个元素赋相同值
    X[0:2, :] = 12
    print(f"批量修改:\n{X}")


def main():
    """主函数"""
    print("🚀 开始深度学习第二章：张量操作学习")
    print("=" * 50)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("💻 使用CPU")
        device = torch.device('cpu')
    
    print(f"📦 PyTorch版本: {torch.__version__}")
    print("=" * 50)
    
    # 执行各种操作
    X = basic_tensor_operations()
    tensor_arithmetic()
    broadcasting_demo()
    indexing_and_slicing()
    
    print("\n🎉 第二章张量基础学习完成！")
    print("💡 接下来可以学习：")
    print("   - 数据预处理 (data_preprocessing.py)")
    print("   - 线性代数 (linear_algebra.py)")
    print("   - 微积分 (calculus.py)")


if __name__ == "__main__":
    main()
