# GraphVul

基于多关系图神经网络的代码漏洞检测框架

## 简介

GraphVul 是一个创新的漏洞检测框架，通过将多关系图神经网络与预训练代码语义相结合，实现对源代码中安全漏洞的自动检测。该框架解决了现有方法在处理代码结构关系和语义理解方面的局限性。

## 核心特性

- **多关系图建模**：显式建模代码属性图中的异构边类型（AST、CFG、DDG），学习不同程序关系对漏洞检测的重要性权重
- **预训练语义注入**：使用 CodeBERT 初始化节点嵌入，将丰富的代码语义知识融入图推理过程
- **对比学习优化**：引入监督对比学习目标，学习更具判别性的漏洞表示，提升模型鲁棒性

## 项目结构

```
.
├── dataprocess.py    # 数据预处理模块
├── graphvul.py       # GraphVul 模型主体
├── experiments/      # 实验脚本和配置
└── README.md
```

## 环境要求

```
Python >= 3.8
PyTorch >= 1.10
torch-geometric
transformers
```

## 快速开始

### 1. 安装依赖

```
pip install -r requirements.txt
```

### 2. 数据预处理

```
python dataprocess.py --input <源代码目录> --output <输出目录>
```

### 3. 模型训练

```
python graphvul.py --mode train --data <数据路径>
```

### 4. 漏洞检测

```
python graphvul.py --mode predict --model <模型路径> --input <待检测代码>
```

## 方法概述

GraphVul 采用四阶段语义-结构协同演化流程：

1. **拓扑保持多关系提取**：构建包含 AST、CFG、DDG、CALL 四种边类型的异构代码图
2. **跨模态知识注入**：通过 CodeBERT 将预训练语义嵌入到图节点
3. **关系感知消息传播**：使用可学习的关系权重进行异构消息聚合
4. **双目标表示优化**：联合优化分类损失和对比损失
