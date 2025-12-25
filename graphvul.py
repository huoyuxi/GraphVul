#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CausalVulDetect: 优化版（整合改进 + UnixCoder 函数级全局编码复用）
---------------------------------------------------------------
关键点：
1. 仍然使用你原来的 CPG -> 图 数据管线和缓存结构（./cache/<dataset>/...）
2. 新增使用 UnixCoder 函数级编码（来自你之前跑的 unixcode.py 的 cache_unixcoder_seq）
   - 第一次需要根据 cache_unixcoder_seq 跑一遍 CLS，缓存到 ./cache/<dataset>/unixcoder/step2_unixcoder_func_emb.pkl
   - 之后所有实验都只读这个缓存，不再跑 UnixCoder
3. 在图模型里，把 GNN graph_feat 与 UnixCoder 函数级编码拼接，作为更强的全局表示
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import time
import random
import re
import hashlib
import glob
import gc
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATv2Conv,
    GINConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool
)
from torch_geometric.utils import softmax

from transformers import RobertaTokenizer, RobertaModel
import transformers
transformers.logging.set_verbosity_error()

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import pygraphviz as pgv

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# ============================================================================
# 损失函数
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for binary classification"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        pos_probs = probs[:, 1]
        targets_float = targets.float()
        
        intersection = pos_probs * targets_float
        union = pos_probs + targets_float
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        
        return dice_loss.mean()


class CombinedLoss(nn.Module):
    """组合损失：Dice Loss + Cross Entropy"""
    def __init__(self, dice_weight=0.6, ce_weight=0.4, class_weights=None, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.class_weights = class_weights
    
    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        
        if self.class_weights is not None:
            weight = self.class_weights.to(dtype=logits.dtype, device=logits.device)
            ce = F.cross_entropy(logits, targets, weight=weight)
        else:
            ce = F.cross_entropy(logits, targets)
        
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        return total_loss

# ============================================================================
# 配置
# ============================================================================

@dataclass
class ExperimentConfig:
    """实验配置（增加 UnixCoder 函数级全局编码相关配置）"""

    # 基础
    project_name: str = "CausalVulDetect_Reborn"
    experiment_name: str = "reborn_experiment"
    random_seed: int = 42

    # 数据集
    dataset_name: str = "Reveal"
    data_root: str = "/home/huoguoyuxi/name/yuxi/graph"
    min_nodes: int = 2
    max_nodes: int = 500

    # 编码器
    encoder_type: str = "unixcoder"          # codebert / unixcoder
    codebert_model: str = "microsoft/codebert-base"
    unixcoder_model: str = "microsoft/unixcoder-base"

    # ⭐ 函数级 UnixCoder 全局编码相关
    use_global_func: bool = True                         # 是否使用函数级全局编码
    global_func_dim: int = 768                           # unixcoder-base hidden size = 512
    unixcoder_seq_cache_root: str = "./cache_unixcoder_seq"
    unixcoder_max_len: int = 512                         # 你跑 unixcode.py 时的 max-len

    # 模型维度
    hidden_dim: int = 512
    num_gnn_layers: int = 6
    num_classes: int = 2
    dropout: float = 0.3

    # OneHot（可选）
    use_onehot: bool = False
    onehot_dim: int = 256
    onehot_max_tokens: int = 256

    # 图不平衡 & 多任务相关开关
    use_node_mil: bool = True         # 节点 MIL 弱监督定位
    lambda_node_mil: float = 0.2

    use_tair: bool = True             # 拓扑/中心不平衡正则（graph-center margin）
    lambda_tair: float = 0.05

    use_curriculum: bool = True       # Edge / 图级 curriculum
    use_pruning: bool = True          # 剪枝 + 稀疏化 (MoS-CPG 简化版)

    # 对比学习
    use_contrastive: bool = True
    contrastive_temp: float = 0.07
    contrastive_weight: float = 0.02  # 略小，避免不稳定

    warmup_epochs_node_mil: int = 10       # 前 10 个 epoch 不用 MIL
    warmup_epochs_tair: int = 10           # 前 10 个 epoch 不用 TAIR
    warmup_epochs_contrastive: int = 15    # 对比学习稍晚一点再开
    warmup_epochs_pruning: int = 5         # 前 5 epoch 不剪枝

    # 训练
    batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 200
    patience: int = 25
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # 损失（类不平衡）
    use_dice_loss: bool = True
    dice_weight: float = 0.7
    ce_weight: float = 0.3
    dice_smooth: float = 1.0

    use_focal_loss: bool = False
    focal_alpha: float = 0.75
    focal_gamma: float = 3.0
    max_class_weight: float = 10.0

    # SMOTETomek
    use_smote_tomek: bool = True
    smote_strategy: float = 0.8

    # 阈值搜索
    threshold_search_points: int = 100
    threshold_min_range: float = 0.25

    # 系统
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0
    use_cache: bool = True

    # 输出目录
    output_dir: str = "./experiments_reborn"
    save_model: bool = True
    save_plots: bool = True

    # 实验控制
    run_ablation: bool = True
    run_interpretability: bool = False
    num_interpretability_samples: int = 10

    # 多关系
    num_relations: int = 7

    # 图级 LogReg 后端
    use_graph_svm: bool = True

    # 自动子目录
    auto_subdir: bool = True

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{self.dataset_name}_{self.encoder_type}_reborn_{timestamp}"

        if self.auto_subdir:
            self.output_dir = os.path.join(self.output_dir, self.experiment_id)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.output_dir, "models")).mkdir(exist_ok=True)
        Path(os.path.join(self.output_dir, "plots")).mkdir(exist_ok=True)
        Path(os.path.join(self.output_dir, "logs")).mkdir(exist_ok=True)
        Path(os.path.join(self.output_dir, "results")).mkdir(exist_ok=True)

        self.cpg_path = os.path.join(self.data_root, self.dataset_name, "CPG")

        # 原始图缓存 & 编码器专属缓存
        self.cache_dir_shared = os.path.join("./cache", self.dataset_name)
        self.cache_dir_encoder = os.path.join("./cache", self.dataset_name, self.encoder_type)
        Path(self.cache_dir_shared).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir_encoder).mkdir(parents=True, exist_ok=True)


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(config: ExperimentConfig):
    log_file = os.path.join(config.output_dir, "logs", "experiment.log")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info(f"Experiment: {config.experiment_name} (OPTIMIZED VERSION + UnixCoder global encoding)")
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Encoder: {config.encoder_type.upper()}")
    logger.info(f"Device: {config.device}")
    logger.info("Output: %s", config.output_dir)
    logger.info("="*80)

    return logger


# ============================================================================
# 工具
# ============================================================================

def _canonical_dataset_name(name: str) -> str:
    name = name.strip()
    lower = name.lower()
    if lower in ["bigvul", "big-vul", "big_vul"]:
        return "SVulD"
    return name


# ============================================================================
# CPG 数据加载 & 编码（复用你原来的逻辑 + 新增 UnixCoder 函数级编码）
# ============================================================================

class CPGDataLoader:
    """CPG数据加载器"""

    def __init__(self, config: ExperimentConfig, logger):
        self.config = config
        self.logger = logger
        self.tokenizer = None
        self.encoder_model = None

        self.edge_types = ["CFG", "DDG", "CDG", "CALL", "DEF_USE", "AST", "OTHER"]
        self.config.num_relations = len(self.edge_types)

        encoder_prefix = self.config.encoder_type
        
        self.cache_graphs_file = os.path.join(config.cache_dir_shared, "step1_parsed_graphs.pkl")
        self.cache_onehot_file = os.path.join(config.cache_dir_shared, "step2_onehot_features.pkl")
        
        self.cache_features_file = os.path.join(config.cache_dir_encoder, f"step2_{encoder_prefix}_features.pkl")
        self.cache_concat_file = os.path.join(config.cache_dir_encoder, f"step2_{encoder_prefix}_concat_features.pkl")
        self.cache_dataset_file = os.path.join(config.cache_dir_encoder, f"step3_{encoder_prefix}_pyg_dataset.pkl")

        # ⭐ 新增：UnixCoder 函数级全局编码缓存（不改变原有缓存，仅新增一个文件）
        self.cache_func_emb_file = os.path.join(
            config.cache_dir_encoder,
            f"step2_{encoder_prefix}_func_emb.pkl"
        )

    # ------------------------------------------------------------------
    # 编码器加载：CodeBERT / UnixCoder
    # ------------------------------------------------------------------
    def _load_encoder(self):
        if self.tokenizer is None:
            if self.config.encoder_type == "codebert":
                self.logger.info("Loading CodeBERT model...")
                model_name = self.config.codebert_model
            elif self.config.encoder_type == "unixcoder":
                self.logger.info("Loading UnixCoder model...")
                model_name = self.config.unixcoder_model
            else:
                raise ValueError(f"Unsupported encoder type: {self.config.encoder_type}")
            
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.encoder_model = RobertaModel.from_pretrained(model_name)
            self.encoder_model.eval()
            self.encoder_model.to(self.config.device)
            self.logger.info(f"{self.config.encoder_type.upper()} loaded successfully")

    # ------------------------------------------------------------------
    # 顶层入口：返回 train/val/test DataLoader
    # ------------------------------------------------------------------
    def load_all_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        self.logger.info(f"Loading CPG data with {self.config.encoder_type.upper()} encoder...")

        expected_dim = 768 + (self.config.onehot_dim if self.config.use_onehot else 0)

        if self.config.use_cache and os.path.exists(self.cache_dataset_file):
            self.logger.info("✓ Found complete dataset cache, loading...")
            try:
                with open(self.cache_dataset_file, 'rb') as f:
                    cached = pickle.load(f)
                train_data = cached['train']
                val_data = cached['val']
                test_data = cached['test']

                need_rebuild = False

                # 1) x 维度不对（比如你改了 use_onehot）
                if len(train_data) > 0 and train_data[0].x.size(1) != expected_dim:
                    self.logger.warning("Cached dataset dim mismatch. Rebuilding...")
                    need_rebuild = True

                # 2) 对于 UnixCoder + 使用函数级编码，如果没有 func_emb 字段，说明是老缓存，也要重建一遍
                if (self.config.encoder_type == "unixcoder"
                        and self.config.use_global_func
                        and len(train_data) > 0
                        and not hasattr(train_data[0], "func_emb")):
                    self.logger.warning("Cached dataset has no func_emb. Rebuilding to inject UnixCoder function embeddings...")
                    need_rebuild = True

                if need_rebuild:
                    train_data, val_data, test_data = self._build_dataset_from_scratch()
                else:
                    self.logger.info(
                        f"✓ Loaded from cache: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}, rebuilding...")
                train_data, val_data, test_data = self._build_dataset_from_scratch()
        else:
            train_data, val_data, test_data = self._build_dataset_from_scratch()

        self._diagnose_label_distribution(train_data, val_data, test_data)

        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        test_loader = DataLoader(test_data, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

        self.logger.info("Data preparation completed:")
        self.logger.info(f"  Train: {len(train_data)} graphs")
        self.logger.info(f"  Val:   {len(val_data)} graphs")
        self.logger.info(f"  Test:  {len(test_data)} graphs")
        self.logger.info(f"  Node feature dim: {expected_dim}")

        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------
    # 标签诊断
    # ------------------------------------------------------------------
    def _diagnose_label_distribution(self, train_data, val_data, test_data):
        def count_labels(data_list):
            labels = [int(d.y.view(-1).item()) for d in data_list]
            neg_count = labels.count(0)
            pos_count = labels.count(1)
            total = len(labels)
            return neg_count, pos_count, total

        train_neg, train_pos, train_total = count_labels(train_data)
        val_neg, val_pos, val_total = count_labels(val_data)
        test_neg, test_pos, test_total = count_labels(test_data)

        self.logger.info("\n" + "="*70)
        self.logger.info("LABEL DISTRIBUTION DIAGNOSIS")
        self.logger.info("="*70)

        if train_total > 0:
            self.logger.info(f"Train Set:")
            self.logger.info(f"  Normal (0):     {train_neg:4d} ({train_neg/train_total*100:.1f}%)")
            self.logger.info(f"  Vulnerable (1): {train_pos:4d} ({train_pos/train_total*100:.1f}%)")
            self.logger.info(f"  Ratio (neg/pos): {train_neg/max(train_pos,1):.2f}:1")

        if val_total > 0:
            self.logger.info(f"\nValidation Set:")
            self.logger.info(f"  Normal (0):     {val_neg:4d} ({val_neg/val_total*100:.1f}%)")
            self.logger.info(f"  Vulnerable (1): {val_pos:4d} ({val_pos/val_total*100:.1f}%)")

        if test_total > 0:
            self.logger.info(f"\nTest Set:")
            self.logger.info(f"  Normal (0):     {test_neg:4d} ({test_neg/test_total*100:.1f}%)")
            self.logger.info(f"  Vulnerable (1): {test_pos:4d} ({test_pos/test_total*100:.1f}%)")

        if train_total > 0:
            train_ratio = train_neg / max(train_pos, 1)
            if train_ratio > 10:
                self.logger.warning(f"⚠ Severe class imbalance detected! Ratio: {train_ratio:.1f}:1")
            elif train_ratio > 5:
                self.logger.warning(f"⚠ Moderate class imbalance detected! Ratio: {train_ratio:.1f}:1")

        self.logger.info("="*70 + "\n")

    # ------------------------------------------------------------------
    # 构建数据集（如果没有缓存或需要重建）
    # ------------------------------------------------------------------
    def _build_dataset_from_scratch(self) -> Tuple[List[Data], List[Data], List[Data]]:
        graphs, file_infos = self._load_or_parse_graphs()
        encoder_data = self._extract_or_load_encoder_features(graphs, file_infos)

        if self.config.use_onehot:
            onehot_list = self._extract_or_load_onehot_features(graphs, file_infos)
        else:
            onehot_list = None

        all_data = self._build_or_load_concat_dataset(encoder_data, onehot_list)

        # ⭐ 新增：给每个 Data 挂 UnixCoder 函数级全局编码 data.func_emb
        all_data = self._attach_func_embeddings_to_data(all_data, file_infos)

        train_data, val_data, test_data = self._split_dataset(all_data)

        if self.config.use_cache:
            self.logger.info("Saving complete dataset cache with func_emb...")
            try:
                with open(self.cache_dataset_file, 'wb') as f:
                    pickle.dump({'train': train_data, 'val': val_data, 'test': test_data}, f)
                self.logger.info("✓ Dataset cache saved")
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")

        del graphs, file_infos, encoder_data, onehot_list, all_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return train_data, val_data, test_data

    # ------------------------------------------------------------------
    # Graph 解析 & 清洗（保留你原来的 min/max 节点数 + 最大连通分量逻辑）
    # ------------------------------------------------------------------
    def _scan_cpg_files(self) -> List[Dict]:
        pattern = os.path.join(self.config.cpg_path, "*.cpg.dot")
        files = glob.glob(pattern)

        file_list = []
        for filepath in files:
            filename = os.path.basename(filepath)
            stem = filename.replace('.cpg.dot', '')
            parts = stem.split('_', 1)
            if len(parts) == 2:
                try:
                    label = int(parts[0])
                    sample_id = parts[1]
                    file_list.append({'file': filepath, 'label': label, 'id': sample_id})
                except Exception:
                    continue
        return file_list

    def _parse_dot_file(self, filepath: str) -> nx.DiGraph:
        ag = pgv.AGraph(filepath)
        graph = nx.DiGraph()

        for n in ag.nodes():
            node_id = str(n)
            label = n.attr.get("label", "")
            label = label.strip()
            if label.startswith("<") and label.endswith(">"):
                label = label[1:-1]

            node_type = ""
            line_num = 0
            node_code = ""
            try:
                if "<BR/>" in label:
                    head, code_part = label.split("<BR/>", 1)
                    node_code = code_part.strip()
                else:
                    head = label
                    node_code = ""

                if "," in head:
                    type_part, line_part = head.split(",", 1)
                    node_type = type_part.strip()
                    line_num = int(line_part.strip())
                else:
                    node_type = head.strip()
                    line_num = 0
            except Exception:
                continue

            node_code = node_code.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")

            graph.add_node(node_id, type=node_type, code=node_code, line=line_num)

        for e in ag.edges():
            src = str(e[0])
            dst = str(e[1])
            edge_label = e.attr.get("label", "").strip()

            if not edge_label:
                etype = "OTHER"
            else:
                if ":" in edge_label:
                    etype = edge_label.split(":", 1)[0].strip()
                else:
                    etype = edge_label.strip()

            etype = self._normalize_edge_type(etype)

            if src in graph.nodes and dst in graph.nodes:
                graph.add_edge(src, dst, type=etype)

        return graph

    def _normalize_edge_type(self, etype: str) -> str:
        etype = etype.upper()
        if etype in ['CFG', 'FLOWS_TO']:
            return 'CFG'
        elif etype in ['DDG', 'REACHES', 'DEF_USE']:
            return 'DDG'
        elif etype in ['CDG', 'CONTROLS']:
            return 'CDG'
        elif etype in ['CALL', 'CALLS']:
            return 'CALL'
        elif etype in ['AST', 'CHILD']:
            return 'AST'
        else:
            return 'OTHER'

    def _load_or_parse_graphs(self) -> Tuple[List[nx.DiGraph], List[Dict]]:
        if self.config.use_cache and os.path.exists(self.cache_graphs_file):
            self.logger.info("✓ Found parsed graphs cache (SHARED), loading...")
            try:
                with open(self.cache_graphs_file, 'rb') as f:
                    cached = pickle.load(f)
                self.logger.info(f"✓ Loaded {len(cached['graphs'])} graphs from shared cache")
                return cached['graphs'], cached['file_infos']
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}, re-parsing...")

        self.logger.info("Parsing CPG files from scratch...")
        cpg_files = self._scan_cpg_files()
        self.logger.info(f"Found {len(cpg_files)} CPG files")

        graphs = []
        valid_file_infos = []

        for file_info in tqdm(cpg_files, desc="Parsing CPG files"):
            try:
                graph = self._parse_dot_file(file_info['file'])

                # ✅ 保持你原来的节点数清理逻辑
                if len(graph.nodes()) < self.config.min_nodes:
                    continue
                if len(graph.nodes()) > self.config.max_nodes:
                    continue

                # 只保留最大弱连通分量
                if len(list(nx.weakly_connected_components(graph))) > 1:
                    largest_cc = max(nx.weakly_connected_components(graph), key=len)
                    graph = graph.subgraph(largest_cc).copy()

                if len(graph.nodes()) > 0:
                    graphs.append(graph)
                    valid_file_infos.append(file_info)

            except Exception as e:
                self.logger.warning(f"Failed to parse {file_info['file']}: {e}")
                continue

        self.logger.info(f"✓ Successfully parsed {len(graphs)} valid graphs")

        if self.config.use_cache:
            self.logger.info("Saving parsed graphs cache (SHARED)...")
            try:
                with open(self.cache_graphs_file, 'wb') as f:
                    pickle.dump({'graphs': graphs, 'file_infos': valid_file_infos}, f)
                self.logger.info("✓ Graphs cache saved")
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")

        return graphs, valid_file_infos

    # ------------------------------------------------------------------
    # 节点级编码（CodeBERT / UnixCoder）
    # ------------------------------------------------------------------
    def _extract_node_features(self, graph: nx.DiGraph) -> torch.Tensor:
        node_list = list(graph.nodes())

        code_texts = []
        for node_id in node_list:
            node_data = graph.nodes[node_id]
            code = node_data.get('code', '')
            node_type = node_data.get('type', '')

            text = f"{node_type} {code}" if code else node_type
            code_texts.append(text[:512])

        with torch.no_grad():
            inputs = self.tokenizer(
                code_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.config.device)

            outputs = self.encoder_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings.cpu()

    def _build_edge_index(self, graph: nx.DiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        node_to_idx = {node: idx for idx, node in enumerate(graph.nodes())}

        edge_sources, edge_targets, edge_types = [], [], []
        for src, dst in graph.edges():
            edge_data = graph.edges[src, dst]
            etype = edge_data.get('type', 'OTHER')

            try:
                etype_idx = self.edge_types.index(etype)
            except ValueError:
                etype_idx = self.edge_types.index('OTHER')

            edge_sources.append(node_to_idx[src])
            edge_targets.append(node_to_idx[dst])
            edge_types.append(etype_idx)

        if len(edge_sources) == 0:
            n = len(graph.nodes())
            edge_index = torch.tensor([[i for i in range(n)], [i for i in range(n)]], dtype=torch.long)
            edge_type = torch.zeros(n, dtype=torch.long)
        else:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)

        return edge_index, edge_type

    def _extract_or_load_encoder_features(self, graphs: List[nx.DiGraph], file_infos: List[Dict]) -> List[Data]:
        if self.config.use_cache and os.path.exists(self.cache_features_file):
            self.logger.info(f"✓ Found {self.config.encoder_type.upper()} features cache, loading...")
            try:
                with open(self.cache_features_file, 'rb') as f:
                    all_data = pickle.load(f)
                self.logger.info(f"✓ Loaded {len(all_data)} graphs from cache")
                return all_data
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}, re-extracting...")

        self.logger.info(f"Extracting {self.config.encoder_type.upper()} features from scratch...")
        self._load_encoder()

        all_data = []
        for graph, file_info in tqdm(zip(graphs, file_infos), total=len(graphs), desc=f"Extracting features"):
            try:
                node_features = self._extract_node_features(graph)
                edge_index, edge_type = self._build_edge_index(graph)
                data = Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_type=edge_type,
                    y=torch.tensor([file_info['label']], dtype=torch.long),
                    num_nodes=len(graph.nodes())
                )
                all_data.append(data)
            except Exception as e:
                self.logger.warning(f"Failed to extract features: {e}")
                continue

        self.logger.info(f"✓ Extracted features for {len(all_data)} graphs")

        if self.config.use_cache:
            self.logger.info(f"Saving {self.config.encoder_type.upper()} features cache...")
            try:
                with open(self.cache_features_file, 'wb') as f:
                    pickle.dump(all_data, f)
                self.logger.info("✓ Features cache saved")
            except Exception as e:
                self.logger.warning(f"Failed to save cache: {e}")

        # 节点编码阶段暂时可以释放 encoder，后面构建 func_emb 时会重新用到
        if self.encoder_model is not None:
            del self.encoder_model
            self.encoder_model = None
            torch.cuda.empty_cache()
            gc.collect()

        return all_data

    # ------------------------------------------------------------------
    # 词袋 One-hot 特征（与你原来保持一致，可选）
    # ------------------------------------------------------------------
    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        text = text.lower()
        toks = re.findall(r"[a-zA-Z_]\w+|\d+", text)
        return toks

    def _hash_bow_onehot(self, tokens: List[str], dim: int) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        for tok in tokens:
            h = hashlib.md5(tok.encode("utf-8")).hexdigest()
            idx = int(h, 16) % dim
            vec[idx] = 1.0
        nrm = np.linalg.norm(vec)
        if nrm > 0:
            vec /= nrm
        return vec

    def _extract_onehot_for_graph(self, graph: nx.DiGraph) -> torch.Tensor:
        node_list = list(graph.nodes())
        feats = []

        for node_id in node_list:
            node_data = graph.nodes[node_id]
            code = node_data.get('code', '')
            node_type = node_data.get('type', '')
            text = f"{node_type} {code}" if code else node_type

            tokens = self._simple_tokenize(text)
            if self.config.onehot_max_tokens > 0:
                tokens = tokens[:self.config.onehot_max_tokens]

            vec = self._hash_bow_onehot(tokens, self.config.onehot_dim)
            feats.append(vec)

        feats = np.stack(feats, axis=0) if len(feats) > 0 else np.zeros((0, self.config.onehot_dim), dtype=np.float32)
        return torch.from_numpy(feats)

    def _extract_or_load_onehot_features(self, graphs: List[nx.DiGraph], file_infos: List[Dict]) -> List[torch.Tensor]:
        if self.config.use_cache and os.path.exists(self.cache_onehot_file):
            self.logger.info("✓ Found onehot features cache (SHARED), loading...")
            try:
                with open(self.cache_onehot_file, "rb") as f:
                    onehot_list = pickle.load(f)
                if len(onehot_list) == len(graphs):
                    self.logger.info(f"✓ Loaded onehot for {len(onehot_list)} graphs from cache")
                    return onehot_list
                else:
                    self.logger.warning("Onehot cache size mismatch. Rebuilding...")
            except Exception as e:
                self.logger.warning(f"Failed to load onehot cache: {e}, re-extracting...")

        self.logger.info("Extracting onehot features from scratch...")
        onehot_list = []
        for graph in tqdm(graphs, desc="Extracting onehot features"):
            try:
                feats = self._extract_onehot_for_graph(graph)
                onehot_list.append(feats)
            except Exception as e:
                self.logger.warning(f"Failed to extract onehot: {e}")
                onehot_list.append(torch.zeros((len(graph.nodes()), self.config.onehot_dim), dtype=torch.float32))

        if self.config.use_cache:
            self.logger.info("Saving onehot features cache (SHARED)...")
            try:
                with open(self.cache_onehot_file, "wb") as f:
                    pickle.dump(onehot_list, f)
                self.logger.info("✓ Onehot cache saved")
            except Exception as e:
                self.logger.warning(f"Failed to save onehot cache: {e}")

        return onehot_list

    def _build_or_load_concat_dataset(self, encoder_data: List[Data], onehot_list: Optional[List[torch.Tensor]]) -> List[Data]:
        if not self.config.use_onehot:
            self.logger.info("use_onehot=False, skip concat.")
            return encoder_data

        expected_dim = 768 + self.config.onehot_dim

        if self.config.use_cache and os.path.exists(self.cache_concat_file):
            self.logger.info(f"✓ Found concat features cache, loading...")
            try:
                with open(self.cache_concat_file, "rb") as f:
                    concat_data = pickle.load(f)
                if len(concat_data) == len(encoder_data) and concat_data[0].x.size(1) == expected_dim:
                    self.logger.info(f"✓ Loaded concat dataset for {len(concat_data)} graphs")
                    return concat_data
                else:
                    self.logger.warning("Concat cache mismatch. Rebuilding...")
            except Exception as e:
                self.logger.warning(f"Failed to load concat cache: {e}, rebuilding...")

        self.logger.info(f"Building concat dataset from scratch...")
        concat_data = []
        for i, data in tqdm(enumerate(encoder_data), total=len(encoder_data), desc="Concatenating features"):
            try:
                x_encoder = data.x
                x_onehot = onehot_list[i]

                if x_onehot.size(0) != x_encoder.size(0):
                    x_onehot = torch.zeros((x_encoder.size(0), self.config.onehot_dim), dtype=torch.float32)

                x_concat = torch.cat([x_encoder, x_onehot], dim=-1)

                new_data = Data(
                    x=x_concat,
                    edge_index=data.edge_index,
                    edge_type=data.edge_type,
                    y=data.y,
                    num_nodes=data.num_nodes
                )
                concat_data.append(new_data)
            except Exception as e:
                self.logger.warning(f"Failed to concat graph {i}: {e}")
                concat_data.append(data)

        if self.config.use_cache:
            self.logger.info(f"Saving concat features cache...")
            try:
                with open(self.cache_concat_file, "wb") as f:
                    pickle.dump(concat_data, f)
                self.logger.info("✓ Concat cache saved")
            except Exception as e:
                self.logger.warning(f"Failed to save concat cache: {e}")

        return concat_data

    # ------------------------------------------------------------------
    # ⭐ UnixCoder 函数级全局编码：从 cache_unixcoder_seq 生成 / 读取 CLS 并缓存
    # ------------------------------------------------------------------
    def _load_or_build_unixcoder_func_emb_cache(self) -> dict:
        """
        从 ./cache_unixcoder_seq 读取 token 缓存，跑一次 UnixCoder 得到函数级 CLS 向量，
        缓存在 self.cache_func_emb_file，之后所有实验直接读取。
        返回：id -> np.ndarray[global_func_dim]
        """
        if self.config.encoder_type != "unixcoder" or not self.config.use_global_func:
            return {}

        # 先看本地 func_emb 缓存
        if self.config.use_cache and os.path.exists(self.cache_func_emb_file):
            self.logger.info(f"✓ Found UnixCoder func_emb cache: {self.cache_func_emb_file}, loading...")
            try:
                with open(self.cache_func_emb_file, "rb") as f:
                    obj = pickle.load(f)
                id2emb = obj.get("id2emb", {})
                dim = obj.get("dim", self.config.global_func_dim)
                if len(id2emb) > 0 and dim == self.config.global_func_dim:
                    self.logger.info(f"✓ Loaded {len(id2emb)} func_emb vectors (dim={dim})")
                    return id2emb
                else:
                    self.logger.warning("func_emb cache mismatch or empty, will rebuild...")
            except Exception as e:
                self.logger.warning(f"Failed to load func_emb cache: {e}, will rebuild...")

        # 没有 func_emb 缓存：从 sequence 缓存里算一遍 CLS
        real_name = _canonical_dataset_name(self.config.dataset_name)
        encoder_tag = self.config.unixcoder_model.replace("/", "_")
        seq_cache_dir = os.path.join(
            self.config.unixcoder_seq_cache_root,
            real_name,
            encoder_tag
        )
        seq_cache_file = os.path.join(
            seq_cache_dir,
            f"all_maxlen{self.config.unixcoder_max_len}.pt"
        )

        if not os.path.exists(seq_cache_file):
            self.logger.warning(
                f"[UnixCoder func_emb] seq cache not found: {seq_cache_file}\n"
                f"请先用 unixcode.py 跑一遍，产生 cache_unixcoder_seq。"
            )
            return {}

        self.logger.info(f"[UnixCoder func_emb] Loading seq cache: {seq_cache_file}")
        buf = torch.load(seq_cache_file, map_location="cpu")
        input_ids = buf["input_ids"]           # [N, L]
        attention_mask = buf["attention_mask"]
        ids = list(buf["ids"])                # list[str] or list[int]

        N, L = input_ids.size()
        self.logger.info(f"[UnixCoder func_emb] seq cache: N={N}, L={L}")

        # 用已经存在的 UnixCoder encoder（如果没有就加载）
        self._load_encoder()
        self.encoder_model.eval()
        device = self.config.device if torch.cuda.is_available() else "cpu"
        self.encoder_model.to(device)

        all_cls = []
        batch_size = 64   # 可以根据显存调整

        for start in tqdm(range(0, N, batch_size), desc="Building func_emb from seq cache"):
            end = min(start + batch_size, N)
            batch_ids = input_ids[start:end].to(device)
            batch_mask = attention_mask[start:end].to(device)
            with torch.no_grad():
                outputs = self.encoder_model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask
                )
                cls = outputs.last_hidden_state[:, 0, :]   # [B, hidden]
                all_cls.append(cls.cpu())

        all_cls = torch.cat(all_cls, dim=0)    # [N, hidden]
        hidden_dim = all_cls.size(1)
        self.logger.info(f"[UnixCoder func_emb] built CLS matrix: {all_cls.shape}")

        id2emb = {}
        for i, idx in enumerate(ids):
            key = str(idx)
            id2emb[key] = all_cls[i].numpy()

        # 缓存到本地，后面直接读
        if self.config.use_cache:
            try:
                with open(self.cache_func_emb_file, "wb") as f:
                    pickle.dump(
                        {
                            "id2emb": id2emb,
                            "dim": int(hidden_dim),
                            "meta": {
                                "dataset": real_name,
                                "encoder": self.config.unixcoder_model,
                                "max_len": self.config.unixcoder_max_len,
                                "num_samples": int(N),
                            },
                        },
                        f
                    )
                self.logger.info(f"[UnixCoder func_emb] cache saved to {self.cache_func_emb_file}")
            except Exception as e:
                self.logger.warning(f"Failed to save func_emb cache: {e}")

        # 这个函数结束后可以保留 encoder_model（后续 node 编码不会再用到）

        return id2emb

    def _attach_func_embeddings_to_data(self, data_list: List[Data], file_infos: List[Dict]) -> List[Data]:
        """
        给每个 Data 挂一个 data.func_emb （UnixCoder 函数级全局编码）
        - 映射关系通过 CPG 文件名里的 sample_id（file_info['id']）和 jsonl 里的 idx 对齐
        - 如果找不到就用全零向量，并统计缺失数量
        """
        if self.config.encoder_type != "unixcoder" or not self.config.use_global_func:
            return data_list

        id2emb = self._load_or_build_unixcoder_func_emb_cache()
        if len(id2emb) == 0:
            self.logger.warning("No UnixCoder func_emb available, all func_emb will be zeros.")
            func_dim = self.config.global_func_dim
        else:
            # 随便拿一个看维度
            any_vec = next(iter(id2emb.values()))
            func_dim = int(len(any_vec))
            if func_dim != self.config.global_func_dim:
                self.logger.warning(
                    f"func_emb dim ({func_dim}) != config.global_func_dim ({self.config.global_func_dim}), "
                    f"将以缓存为准。"
                )
                self.config.global_func_dim = func_dim

        new_data_list = []
        miss = 0

        for data, info in zip(data_list, file_infos):
            sample_id = str(info.get("id"))
            vec = id2emb.get(sample_id, None)
            if vec is None:
                miss += 1
                func_tensor = torch.zeros(func_dim, dtype=torch.float32)
            else:
                func_tensor = torch.from_numpy(vec).float()
            # ⭐ 挂到 Data 上（注意是一维向量，Batch 之后会变成 [num_graphs, dim]）
            data.func_emb = func_tensor
            new_data_list.append(data)

        if miss > 0:
            self.logger.warning(
                f"[UnixCoder func_emb] {miss}/{len(data_list)} graphs have no matching function embedding, filled with zeros."
            )

        return new_data_list

    # ------------------------------------------------------------------
    # 数据集划分（仍然保持你原来的 0.8 / 0.1 / 0.1 逻辑）
    # ------------------------------------------------------------------
    def _split_dataset(self, all_data: List[Data]) -> Tuple[List[Data], List[Data], List[Data]]:
        labels = [int(data.y.view(-1).item()) for data in all_data]

        train_data, temp_data = train_test_split(
            all_data,
            test_size=(1 - self.config.train_ratio),
            random_state=self.config.random_seed,
            stratify=labels
        )

        temp_labels = [int(d.y.view(-1).item()) for d in temp_data]
        remain_ratio = 1 - self.config.train_ratio
        if remain_ratio <= 0:
            return train_data, [], []
        val_ratio_in_temp = self.config.val_ratio / remain_ratio

        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_ratio_in_temp),
            random_state=self.config.random_seed + 1,
            stratify=temp_labels
        )

        return train_data, val_data, test_data


# ============================================================================
# LNN 细胞（可选，目前没有接进主模型，但保留实现）
# ============================================================================

class LiquidTimeConstantCell(nn.Module):
    """
    单个液态神经元（LTC 风格）:
    dh/dt = -1/τ ⊙ h + f(W_in * x + W_rec * h + b)
    τ 通过 softplus 保证为正
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.rec_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.log_tau = nn.Parameter(torch.zeros(hidden_dim))
        self.activation = nn.Tanh()

    def forward(self, x, h, dt: float = 1.0):
        tau = F.softplus(self.log_tau) + 1e-3
        pre_act = self.in_layer(x) + self.rec_layer(h) + self.bias
        f = self.activation(pre_act)
        dh = (-1.0 / tau) * h + f
        h = h + dt * dh
        return h


# ============================================================================
# GNN 组件：Relational GAT + 注意力读出 + Edge 剪枝 / Curriculum
# ============================================================================

class RelationalGATBackbone(nn.Module):
    """
    边类型感知的多层 GATv2Conv + 残差 + LayerNorm
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_edge_types: int,
        dropout: float = 0.3,
        heads: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        edge_dim = hidden_dim // 4
        self.edge_type_embedding = nn.Embedding(num_edge_types, edge_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = input_dim
        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=in_dim,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                edge_dim=edge_dim
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        self.out_dim = hidden_dim

    def forward(self, x, edge_index, edge_type):
        if edge_index.numel() == 0:
            h = x
            for norm in self.norms:
                h = norm(F.relu(h))
            return h

        edge_attr = self.edge_type_embedding(edge_type)  # [E, edge_dim]

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr=edge_attr)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            if h_new.size(-1) == h.size(-1):
                h = norm(h + h_new)
            else:
                h = norm(h_new)
        return h


class AttentiveReadout(nn.Module):
    """
    图级注意力池化 + mean/max 池化拼接
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, batch):
        scores = self.gate_nn(x).squeeze(-1)  # [N]
        alpha = softmax(scores, batch)        # [N]

        num_graphs = int(batch.max().item()) + 1
        att_pool = torch.zeros(num_graphs, x.size(1), device=x.device)
        att_pool.index_add_(0, batch, alpha.unsqueeze(-1) * x)

        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)

        graph_feat = torch.cat([att_pool, mean_pool, max_pool], dim=-1)  # [G, 3F]
        return graph_feat, alpha


class EdgePruningCurriculum(nn.Module):
    """
    MoS-CPG 风格的边权重估计（importance score）+ curriculum + 剪枝
    """
    def __init__(self, node_dim: int, num_edge_types: int, hidden_dim: int = 128):
        super().__init__()
        edge_dim = hidden_dim // 2
        self.edge_type_embedding = nn.Embedding(num_edge_types, edge_dim)
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_type, progress: float):
        if edge_index.numel() == 0:
            return edge_index, edge_type, None

        src, dst = edge_index
        e_emb = self.edge_type_embedding(edge_type)          # [E, edge_emb]
        edge_feat = torch.cat([x[src], x[dst], e_emb], dim=-1)
        scores = self.mlp(edge_feat).squeeze(-1)             # [E], in (0,1)

        keep_ratio = 0.2 + 0.8 * float(progress)
        keep_ratio = max(0.2, min(1.0, keep_ratio))

        E = scores.size(0)
        k = int(max(1, keep_ratio * E))

        topk_scores, topk_idx = torch.topk(scores, k=k, largest=True, sorted=False)
        edge_index_new = edge_index[:, topk_idx]
        edge_type_new = edge_type[topk_idx]

        return edge_index_new, edge_type_new, topk_scores


class CausalVulModel(nn.Module):
    """
    主模型：Relational GAT + Attentive Readout + 节点MIL + Edge Pruning/Curriculum
    ⭐ 这里融合 UnixCoder 函数级全局编码（data.func_emb）
    """

    def __init__(self, config: ExperimentConfig, num_relations: int):
        super().__init__()
        self.config = config

        # 输入维度 = 节点编码(768) [+ 可选 onehot]
        input_dim = 768 + (config.onehot_dim if config.use_onehot else 0)
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)

        # GNN 主干
        self.backbone = RelationalGATBackbone(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_gnn_layers,
            num_edge_types=num_relations,
            dropout=config.dropout,
            heads=4
        )

        # Edge 剪枝 / curriculum
        if config.use_curriculum or config.use_pruning:
            self.edge_gate = EdgePruningCurriculum(
                node_dim=config.hidden_dim,
                num_edge_types=num_relations,
                hidden_dim=128
            )
        else:
            self.edge_gate = None

        # 图级读出：输出 3 * hidden_dim
        self.readout = AttentiveReadout(
            in_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim
        )

        # ⭐ UnixCoder 函数级全局编码
        self.use_global_func = getattr(config, "use_global_func", False)
        self.func_emb_dim = getattr(config, "global_func_dim", 0)

        if self.use_global_func and self.func_emb_dim > 0:
            # func_emb: [G, func_emb_dim] (比如 768) → [G, hidden_dim]
            self.func_proj = nn.Linear(self.func_emb_dim, config.hidden_dim)
            # graph_feat = [3H (GNN readout) + 1H (func_proj)] = 4H
            self.graph_dim = config.hidden_dim * 4
        else:
            self.func_proj = None
            # 只用 GNN 读出：3H
            self.graph_dim = config.hidden_dim * 3

        # 图级分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.graph_dim, self.graph_dim // 2),
            nn.LayerNorm(self.graph_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.graph_dim // 2, config.num_classes)
        )

        # 节点级 MIL 头
        if config.use_node_mil:
            self.node_classifier = nn.Linear(config.hidden_dim, config.num_classes)
        else:
            self.node_classifier = None

        # 对比学习投影
        if config.use_contrastive:
            self.projection = nn.Sequential(
                nn.Linear(self.graph_dim, self.graph_dim // 2),
                nn.ReLU(),
                nn.Linear(self.graph_dim // 2, 128)
            )
        else:
            self.projection = None

        self.curriculum_progress = 1.0  # [0,1]

        self.reset_parameters()

    def set_curriculum_progress(self, progress: float):
        self.curriculum_progress = float(max(0.0, min(1.0, progress)))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (GATv2Conv, SAGEConv, GINConv)):
                try:
                    m.reset_parameters()
                except Exception:
                    pass
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch):
        x = self.input_proj(batch.x)                     # [N, hidden_dim]
        edge_index = batch.edge_index
        edge_type = batch.edge_type

        # Edge 剪枝 + curriculum
        if self.edge_gate is not None and edge_index.numel() > 0:
            edge_index, edge_type, edge_weight = self.edge_gate(
                x, edge_index, edge_type, self.curriculum_progress
            )
        else:
            edge_weight = None

        # GNN 主干
        node_feat = self.backbone(x, edge_index, edge_type)

        # 图级读出：3H
        graph_feat, node_att = self.readout(node_feat, batch.batch)

        # ⭐ 融合 UnixCoder 函数级全局编码
        if self.func_proj is not None and hasattr(batch, "func_emb"):
            func_emb = batch.func_emb.to(graph_feat.device)

            # 按 PyG 的 Batch 规则，一般是 [G, func_dim]；保险起见做下 reshape 兜底
            if func_emb.dim() == 1:
                num_graphs = graph_feat.size(0)
                func_emb = func_emb.view(num_graphs, -1)
            elif func_emb.size(0) != graph_feat.size(0):
                num_graphs = graph_feat.size(0)
                func_emb = func_emb.view(num_graphs, -1)

            func_feat = self.func_proj(func_emb)         # [G, hidden_dim]
            graph_feat = torch.cat([graph_feat, func_feat], dim=-1)  # [G, 4H]

        # 图级 logits
        logits = self.classifier(graph_feat)

        # 节点 logits（MIL 用）
        if self.node_classifier is not None:
            node_logits = self.node_classifier(node_feat)
        else:
            node_logits = None

        # 对比学习投影
        if self.config.use_contrastive and self.projection is not None:
            proj_feat = self.projection(graph_feat)
        else:
            proj_feat = None

        return {
            "logits": logits,
            "graph_feat": graph_feat,
            "node_feat": node_feat,
            "node_logits": node_logits,
            "node_att": node_att,
            "proj_feat": proj_feat
        }

    def extract_multi_scale_graph_features(self, batch):
        """
        给 Graph LogReg 用的接口：返回图级特征（已经融合 UnixCoder 函数级编码）
        """
        self.eval()
        with torch.no_grad():
            x = self.input_proj(batch.x)
            edge_index = batch.edge_index
            edge_type = batch.edge_type

            if self.edge_gate is not None and edge_index.numel() > 0:
                edge_index, edge_type, _ = self.edge_gate(
                    x, edge_index, edge_type, progress=1.0
                )

            node_feat = self.backbone(x, edge_index, edge_type)
            graph_feat, _ = self.readout(node_feat, batch.batch)     # [G, 3H]

            if self.func_proj is not None and hasattr(batch, "func_emb"):
                func_emb = batch.func_emb.to(graph_feat.device)
                if func_emb.dim() == 1:
                    num_graphs = graph_feat.size(0)
                    func_emb = func_emb.view(num_graphs, -1)
                elif func_emb.size(0) != graph_feat.size(0):
                    num_graphs = graph_feat.size(0)
                    func_emb = func_emb.view(num_graphs, -1)

                func_feat = self.func_proj(func_emb)                  # [G, H]
                graph_feat = torch.cat([graph_feat, func_feat], dim=-1)  # [G, 4H]

            return graph_feat

# ========================================================================
# Trainer：多任务（图级 + 节点 MIL）+ 不平衡正则 + curriculum
# ========================================================================

class Trainer:
    def __init__(self, model, config: ExperimentConfig, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # 优化器 + Scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        self.class_weights = None
        self.use_focal_loss = config.use_focal_loss
        self.use_dice_loss = config.use_dice_loss

        if self.use_dice_loss:
            self.combined_loss = CombinedLoss(
                dice_weight=config.dice_weight,
                ce_weight=config.ce_weight,
                class_weights=None,
                smooth=config.dice_smooth
            )
        else:
            self.combined_loss = None

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_f1': [],
            'train_pos_count': [],
            'val_pos_count': [],
            'val_best_threshold': []
        }

        self.best_val_f1 = 0.0
        self.best_threshold = 0.5
        self.patience_counter = 0

    # ---------------------------
    # 工具函数
    # ---------------------------

    def _process_labels(self, labels):
        if labels.dim() > 1:
            labels = labels.view(-1)
        return labels.long()

    def _compute_class_weights(self, train_loader):
        """自动估计类别权重 & 更新 loss 配置"""
        neg_count = 0
        pos_count = 0
        for batch in train_loader:
            labels = self._process_labels(batch.y)
            neg_count += (labels == 0).sum().item()
            pos_count += (labels == 1).sum().item()

        total = neg_count + pos_count
        if total == 0:
            self.class_weights = torch.tensor([1.0, 1.0], device=self.device)
            return

        ratio = neg_count / max(pos_count, 1)

        self.logger.info("\n" + "="*70)
        self.logger.info("AUTO-CONFIGURATION FOR IMBALANCE")
        self.logger.info("="*70)
        self.logger.info(f"Total: {total}, Neg: {neg_count}, Pos: {pos_count}")
        self.logger.info(f"Imbalance ratio (neg:pos) = {ratio:.2f}:1")

        # 简化版策略：只调 class_weight 和一些开关
        if ratio < 1.5:
            self.use_focal_loss = False
            self.use_dice_loss = False
            self.logger.info("→ Near-balanced, using pure CE loss.")
        elif ratio < 4:
            self.use_dice_loss = True
            self.use_focal_loss = False
            if self.combined_loss is not None:
                self.combined_loss.dice_weight = 0.7
                self.combined_loss.ce_weight = 0.3
            self.logger.info("→ Moderate imbalance, CE + Dice (0.3/0.7).")
        else:
            self.use_dice_loss = True
            self.use_focal_loss = True
            if self.combined_loss is not None:
                self.combined_loss.dice_weight = 0.4
                self.combined_loss.ce_weight = 0.6
            self.logger.info("→ Severe imbalance, Focal + CE + Dice.")

        # class_weight：给正例更高权重，但有上限
        if ratio >= 4:
            pos_w = min(np.sqrt(ratio) * 2, self.config.max_class_weight)
        else:
            pos_w = min(ratio, self.config.max_class_weight)

        self.class_weights = torch.tensor(
            [1.0, pos_w],
            device=self.device,
            dtype=torch.float32
        )
        if self.combined_loss is not None:
            self.combined_loss.class_weights = self.class_weights

        # SMOTETomek 策略
        if pos_count < 50 or ratio < 1.5:
            self.config.use_smote_tomek = False
            self.logger.info("→ Too few positives or nearly balanced, disabling SMOTETomek.")
        elif pos_count < 200:
            self.config.smote_strategy = 0.5
            self.logger.info("→ Limited positives, SMOTE strategy=0.5.")
        else:
            self.config.smote_strategy = 0.8
            self.logger.info("→ Sufficient positives, SMOTE strategy=0.8.")

        self.logger.info(f"→ Class weights: [1.0, {pos_w:.2f}]")
        self.logger.info("="*70 + "\n")

    def _focal_loss(self, logits, labels, alpha=None, gamma=None):
        if alpha is None:
            alpha = self.config.focal_alpha
        if gamma is None:
            gamma = self.config.focal_gamma

        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def _contrastive_loss(self, features, labels):
        """简单 supervised contrastive：同 label 为正 pair"""
        if features is None:
            return torch.tensor(0.0, device=self.device)

        features = F.normalize(features, dim=1)
        batch_size = features.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=self.device)

        sim_matrix = torch.matmul(features, features.T) / self.config.contrastive_temp
        labels = labels.view(-1, 1)
        label_eq = (labels == labels.T).float()
        mask = torch.eye(batch_size, device=self.device).bool()
        label_eq = label_eq.masked_fill(mask, 0)

        exp_sim = torch.exp(sim_matrix)
        positive_sim = exp_sim * label_eq

        loss = 0.0
        num_pos = 0
        for i in range(batch_size):
            pos_sum = positive_sim[i].sum()
            if pos_sum > 0:
                all_sum = exp_sim[i].sum() - exp_sim[i, i]
                loss += -torch.log(pos_sum / (all_sum + 1e-8))
                num_pos += 1

        if num_pos > 0:
            loss = loss / num_pos
        else:
            loss = torch.tensor(0.0, device=self.device)
        return loss

    def _node_mil_loss(self, node_logits, batch, graph_labels):
        """
        Multiple Instance Learning 弱监督：
        - 每个图的节点正类概率 p_i
        - 图级 MIL 概率：p_graph = max_i p_i
        - 用 BCE(p_graph, graph_label) 训练一个额外监督
        """
        if node_logits is None:
            return torch.tensor(0.0, device=self.device)

        probs = F.softmax(node_logits, dim=1)[:, 1]   # [N]
        batch_idx = batch.batch                       # [N]
        num_graphs = int(batch_idx.max().item()) + 1

        mil_probs = []
        mil_labels = []
        for g in range(num_graphs):
            mask = (batch_idx == g)
            if mask.sum() == 0:
                continue
            node_probs_g = probs[mask]
            p_graph = node_probs_g.max()              # max pooling MIL
            mil_probs.append(p_graph)
            mil_labels.append(graph_labels[g])

        if len(mil_probs) == 0:
            return torch.tensor(0.0, device=self.device)

        mil_probs = torch.stack(mil_probs)            # [G']
        mil_labels = torch.tensor(mil_labels, device=self.device, dtype=torch.float32)
        loss = F.binary_cross_entropy(mil_probs, mil_labels)
        return loss

    def _tair_loss(self, graph_feat, graph_labels):
        """
        简化版拓扑/中心正则：让正类图 embedding 更集中（cluster）
        """
        if not self.config.use_tair:
            return torch.tensor(0.0, device=self.device)

        labels = graph_labels.view(-1)
        mask_pos = (labels == 1)
        if mask_pos.sum() <= 1:
            return torch.tensor(0.0, device=self.device)

        pos_feats = graph_feat[mask_pos]
        mu_pos = pos_feats.mean(dim=0)
        diff = pos_feats - mu_pos
        loss = (diff ** 2).sum(dim=1).mean()
        return loss

    @staticmethod
    def _scan_best_threshold(probs: np.ndarray, labels: np.ndarray):
        search_points = np.linspace(0.05, 0.95, 100)
        best_score = -1.0
        best_t = 0.5
        for t in search_points:
            preds = (probs >= t).astype(np.int64)
            f1 = f1_score(labels, preds, zero_division=0)
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)
            score = 0.4 * f1 + 0.3 * recall + 0.3 * precision
            if score > best_score:
                best_score = score
                best_t = float(t)
        preds = (probs >= best_t).astype(np.int64)
        best_f1 = f1_score(labels, preds, zero_division=0)
        return best_t, best_f1

    # ---------------------------
    # 训练 / 验证
    # ---------------------------

    def train_epoch(self, train_loader, epoch_idx: int, num_epochs: int):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        # ---------- 读 warmup 配置（兼容旧 config，没有就当 0） ----------
        warmup_pruning = getattr(self.config, "warmup_epochs_pruning", 0)
        warmup_node_mil = getattr(self.config, "warmup_epochs_node_mil", 0)
        warmup_tair = getattr(self.config, "warmup_epochs_tair", 0)
        warmup_contrastive = getattr(self.config, "warmup_epochs_contrastive", 0)

        # ---------- curriculum / 剪枝进度 ----------
        # 前 warmup_pruning 个 epoch 不剪枝（keep_ratio=1.0），之后再线性进入原有进度
        if epoch_idx + 1 <= warmup_pruning:
            effective_progress = 1.0  # keep_ratio = 1.0 → 不剪枝
        else:
            remain = max(num_epochs - warmup_pruning, 1)
            effective_progress = (epoch_idx + 1 - warmup_pruning) / remain

        if hasattr(self.model, "set_curriculum_progress"):
            self.model.set_curriculum_progress(effective_progress)

        # 当前 epoch 各模块是否开启（受 warmup 控制）
        use_node_mil = (
            self.config.use_node_mil and
            (epoch_idx + 1) > warmup_node_mil
        )
        use_tair = (
            self.config.use_tair and
            (epoch_idx + 1) > warmup_tair
        )
        use_contrastive = (
            self.config.use_contrastive and
            (epoch_idx + 1) > warmup_contrastive
        )

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch_idx+1}", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(batch)
            logits = outputs['logits']
            labels = self._process_labels(batch.y)

            # ---------- 图级主分类损失 ----------
            if self.use_dice_loss and self.combined_loss is not None:
                loss_cls = self.combined_loss(logits, labels)
            elif self.use_focal_loss:
                loss_cls = self._focal_loss(logits, labels)
            else:
                if self.class_weights is not None:
                    loss_cls = F.cross_entropy(logits, labels, weight=self.class_weights)
                else:
                    loss_cls = F.cross_entropy(logits, labels)

            loss = loss_cls

            # ---------- 节点 MIL（预热后才打开） ----------
            if use_node_mil and outputs.get('node_logits') is not None:
                loss_node = self._node_mil_loss(outputs['node_logits'], batch, labels)
                loss = loss + self.config.lambda_node_mil * loss_node

            # ---------- TAIR 图中心正则（预热后才打开） ----------
            if use_tair:
                tair = self._tair_loss(outputs['graph_feat'], labels)
                loss = loss + self.config.lambda_tair * tair

            # ---------- 对比学习（预热后才打开） ----------
            if use_contrastive and outputs.get('proj_feat') is not None:
                contrastive = self._contrastive_loss(outputs['proj_feat'], labels)
                loss = loss + self.config.contrastive_weight * contrastive

            # ---------- L2 正则 ----------
            l2_reg = torch.tensor(0.0, device=self.device)
            for p in self.model.parameters():
                l2_reg += torch.norm(p, p=2)
            loss = loss + 1e-5 * l2_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / max(len(train_loader), 1)
        if len(all_labels) > 0:
            acc = accuracy_score(all_labels, all_preds)
            pos_pred_count = int(np.sum(all_preds))
            total_count = len(all_preds)
        else:
            acc = 0.0
            pos_pred_count = 0
            total_count = 0

        return avg_loss, acc, pos_pred_count, total_count

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                batch = batch.to(self.device)
                outputs = self.model(batch)
                logits = outputs['logits']
                labels = self._process_labels(batch.y)

                if self.use_dice_loss and self.combined_loss is not None:
                    loss_cls = self.combined_loss(logits, labels)
                elif self.use_focal_loss:
                    loss_cls = self._focal_loss(logits, labels)
                else:
                    if self.class_weights is not None:
                        loss_cls = F.cross_entropy(logits, labels, weight=self.class_weights)
                    else:
                        loss_cls = F.cross_entropy(logits, labels)

                loss = loss_cls

                # 验证阶段也可以附带 MIL / TAIR，但不算对比 / L2
                if self.config.use_node_mil and outputs.get('node_logits') is not None:
                    loss_node = self._node_mil_loss(outputs['node_logits'], batch, labels)
                    loss = loss + self.config.lambda_node_mil * loss_node

                if self.config.use_tair:
                    tair = self._tair_loss(outputs['graph_feat'], labels)
                    loss = loss + self.config.lambda_tair * tair

                total_loss += loss.item()

                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / max(len(val_loader), 1)

        if len(all_labels) == 0:
            return {
                'loss': avg_loss, 'accuracy': 0.0, 'precision': 0.0,
                'recall': 0.0, 'f1': 0.0, 'auc': 0.0,
                'pos_pred_count': 0, 'total_count': 0,
                'best_threshold': 0.5, 'best_f1_raw': 0.0
            }

        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)

        best_t, best_f1 = self._scan_best_threshold(all_probs_np, all_labels_np)
        preds_thr = (all_probs_np >= best_t).astype(np.int64)

        acc = accuracy_score(all_labels_np, preds_thr)
        precision = precision_score(all_labels_np, preds_thr, zero_division=0)
        recall = recall_score(all_labels_np, preds_thr, zero_division=0)
        f1 = f1_score(all_labels_np, preds_thr, zero_division=0)
        try:
            auc = roc_auc_score(all_labels_np, all_probs_np)
        except Exception:
            auc = 0.0

        pos_pred_count = int(preds_thr.sum())
        total_count = len(preds_thr)

        return {
            'loss': avg_loss,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'pos_pred_count': pos_pred_count,
            'total_count': total_count,
            'best_threshold': best_t,
            'best_f1_raw': best_f1
        }

    # ---------------------------
    # 总训练流程（含早停 + 模型保存）
    # ---------------------------

    def train(self, train_loader, val_loader):
        self.logger.info("Starting training (Reborn Model with UnixCoder global func_emb)...")
        self._compute_class_weights(train_loader)

        for epoch in range(self.config.num_epochs):
            start_time = time.time()

            train_loss, train_acc, train_pos_pred, train_total = self.train_epoch(
                train_loader, epoch_idx=epoch, num_epochs=self.config.num_epochs
            )
            val_metrics = self.validate(val_loader)

            self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['train_pos_count'].append(train_pos_pred)
            self.history['val_pos_count'].append(val_metrics['pos_pred_count'])
            self.history['val_best_threshold'].append(val_metrics['best_threshold'])

            epoch_time = time.time() - start_time

            self.logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
                f"Prec: {val_metrics['precision']:.4f} Rec: {val_metrics['recall']:.4f} "
                f"F1: {val_metrics['f1']:.4f} | BestT: {val_metrics['best_threshold']:.2f}"
            )

            if val_metrics['best_f1_raw'] > self.best_val_f1 + 1e-4:
                self.best_val_f1 = val_metrics['best_f1_raw']
                self.best_threshold = val_metrics['best_threshold']
                self.patience_counter = 0

                if self.config.save_model:
                    model_path = os.path.join(self.config.output_dir, "models", "best_model.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_f1': self.best_val_f1,
                        'best_threshold': self.best_threshold,
                        'config': asdict(self.config)
                    }, model_path)
                    self.logger.info(
                        f"✓ Saved best model (Val F1: {self.best_val_f1:.4f}, "
                        f"T={self.best_threshold:.2f})"
                    )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self.logger.info(
            f"Training completed. Best Val F1: {self.best_val_f1:.4f} | "
            f"Best T: {self.best_threshold:.2f}"
        )
        return self.history


# ========================================================================
# Evaluator：图级测试 + 保存预测 & 混淆矩阵
# ========================================================================

class Evaluator:
    def __init__(self, model, config: ExperimentConfig, logger, threshold: float = 0.5):
        self.model = model
        self.config = config
        self.logger = logger
        self.threshold = threshold
        self.device = torch.device(config.device)

    def _process_labels(self, labels):
        if labels.dim() > 1:
            labels = labels.view(-1)
        return labels.long()

    def evaluate(self, test_loader) -> dict:
        self.model.eval()

        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = batch.to(self.device)
                outputs = self.model(batch)
                logits = outputs['logits']
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = (probs >= self.threshold).long()
                labels = self._process_labels(batch.y)

                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())

        labels_arr = np.array(all_labels)
        preds_arr = np.array(all_preds)
        probs_arr = np.array(all_probs)

        results = self._compute_metrics(labels_arr, preds_arr, probs_arr)
        self._save_predictions_and_cm(labels_arr, preds_arr, probs_arr, results, prefix="mlp")
        self._print_results(results)

        del all_preds, all_labels, all_probs, labels_arr, preds_arr, probs_arr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _compute_metrics(self, labels, preds, probs) -> dict:
        labels = np.array(labels)
        preds = np.array(preds)
        probs = np.array(probs)

        if len(labels) == 0:
            return {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
                'specificity': 0.0, 'fpr': 0.0, 'fnr': 0.0, 'auc': 0.0
            }

        if len(set(labels)) == 1:
            tn = fp = fn = tp = 0
            if labels[0] == 0:
                tn = int(np.sum((labels == 0) & (preds == 0)))
                fp = int(np.sum((labels == 0) & (preds == 1)))
            else:
                tp = int(np.sum((labels == 1) & (preds == 1)))
                fn = int(np.sum((labels == 1) & (preds == 0)))
        else:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, zero_division=0),
            'recall': recall_score(labels, preds, zero_division=0),
            'f1': f1_score(labels, preds, zero_division=0),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0
        }

        try:
            metrics['auc'] = roc_auc_score(labels, probs)
        except Exception:
            metrics['auc'] = 0.0

        return metrics

    def _save_predictions_and_cm(self, labels, preds, probs, results: dict, prefix: str = "mlp"):
        results_dir = os.path.join(self.config.output_dir, "results")
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        df_pred = pd.DataFrame({
            "idx": np.arange(len(labels)),
            "label": labels,
            "pred": preds,
            "prob_pos": probs
        })
        df_pred.to_csv(os.path.join(results_dir, f"test_predictions_{prefix}.csv"), index=False)

        cm_df = pd.DataFrame(
            [
                [results["tn"], results["fp"]],
                [results["fn"], results["tp"]],
            ],
            index=["Actual_0", "Actual_1"],
            columns=["Pred_0", "Pred_1"],
        )
        cm_df.to_csv(os.path.join(results_dir, f"confusion_matrix_{prefix}.csv"), index=False)

    def _print_results(self, results: dict):
        self.logger.info("\n" + "="*60)
        self.logger.info("TEST RESULTS (MLP Classifier)")
        self.logger.info("="*60)
        self.logger.info(f"Threshold:   {self.threshold:.2f}")
        self.logger.info(f"Accuracy:    {results['accuracy']:.4f}")
        self.logger.info(f"Precision:   {results['precision']:.4f}")
        self.logger.info(f"Recall:      {results['recall']:.4f}")
        self.logger.info(f"F1-Score:    {results['f1']:.4f}")
        self.logger.info(f"AUC:         {results['auc']:.4f}")
        self.logger.info(f"TP: {results['tp']:4d}  FP: {results['fp']:4d}")
        self.logger.info(f"FN: {results['fn']:4d}  TN: {results['tn']:4d}")
        self.logger.info("="*60 + "\n")


# ========================================================================
# GraphSVM：图级特征 + Logistic Regression 后端
# ========================================================================

class GraphSVMClassifier:
    """图级特征 + Logistic Regression 分类器"""

    def __init__(self, config: ExperimentConfig, logger, device='cpu', output_dir: Optional[str] = None):
        self.config = config
        self.logger = logger
        self.device = torch.device(device)
        self.clf = None
        self.best_params_ = None
        self.best_threshold_ = 0.5
        self.output_dir = output_dir

    def _collect_features(self, model, loader):
        model.eval()
        X_list, y_list = [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting graph features", leave=False):
                batch = batch.to(self.device)

                feats = model.extract_multi_scale_graph_features(batch)
                if feats is None:
                    continue

                labels = batch.y.view(-1).long()
                X_list.append(feats.cpu().numpy())
                y_list.append(labels.cpu().numpy())

        if len(X_list) == 0:
            return None, None

        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        return X, y

    @staticmethod
    def _find_best_threshold(prob: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """🔴 优化版阈值搜索：多指标综合优化"""
        search_points = np.linspace(0.05, 0.95, 100)

        best_score = -1.0
        best_t = 0.5

        for t in search_points:
            preds = (prob >= t).astype(np.int64)
            f1 = f1_score(labels, preds, zero_division=0)
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)

            # 🔴 综合得分（优先Recall，兼顾Precision）
            score = 0.4 * f1 + 0.3 * recall + 0.3 * precision

            if score > best_score:
                best_score = score
                best_t = float(t)

        preds = (prob >= best_t).astype(np.int64)
        best_f1 = f1_score(labels, preds, zero_division=0)
        return best_t, best_f1

    def fit(self, model, train_loader, val_loader):
        self.logger.info("Collecting features for LogReg training...")
        X_train, y_train = self._collect_features(model, train_loader)

        if X_train is None:
            self.logger.warning("No features for LogReg. Skip.")
            return

        X_val, y_val = self._collect_features(model, val_loader)
        if X_val is None:
            self.logger.warning("No val features for LogReg. Skip.")
            del X_train, y_train
            gc.collect()
            return

        # ========== SMOTE-Tomek ==========
        if self.config.use_smote_tomek:
            try:
                from imblearn.combine import SMOTETomek

                neg_count_orig = int(np.sum(y_train == 0))
                pos_count_orig = int(np.sum(y_train == 1))
                self.logger.info(f"Before SMOTETomek: Neg={neg_count_orig}, Pos={pos_count_orig}")

                smt = SMOTETomek(
                    sampling_strategy=self.config.smote_strategy,
                    random_state=self.config.random_seed
                )
                X_train, y_train = smt.fit_resample(X_train, y_train)

                neg_count_new = int(np.sum(y_train == 0))
                pos_count_new = int(np.sum(y_train == 1))
                self.logger.info(f"After SMOTETomek: Neg={neg_count_new}, Pos={pos_count_new}")

            except ImportError:
                self.logger.warning("imblearn not installed. Skip SMOTETomek.")
            except Exception as e:
                self.logger.warning(f"SMOTETomek failed: {e}")

        self.logger.info(
            f"LogReg feature dim: {X_train.shape[1]}, "
            f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}"
        )

        # ========== Logistic Regression Search ==========
        C_list = [1.0]
        best_f1 = -1.0
        best_clf = None
        best_params = None
        best_threshold = 0.5

        for C in C_list:
            self.logger.info(f"Training LogReg (C={C})...")

            clf = Pipeline([
                ('scaler', StandardScaler()),
                ('logreg', LogisticRegression(
                    C=C,
                    class_weight='balanced',
                    max_iter=1000,
                    n_jobs=-1,
                    solver='lbfgs'
                ))
            ])

            clf.fit(X_train, y_train)

            val_prob = clf.predict_proba(X_val)[:, 1]
            t, f1_raw = self._find_best_threshold(val_prob, y_val)

            self.logger.info(f"  Val F1={f1_raw:.4f} at T={t:.2f}")

            if f1_raw > best_f1 + 1e-4:
                best_f1 = f1_raw
                best_clf = clf
                best_params = {'C': C}
                best_threshold = t

        if best_clf is None:
            self.logger.warning("LogReg training failed.")
            del X_train, y_train, X_val, y_val
            gc.collect()
            return

        self.clf = best_clf
        self.best_params_ = best_params
        self.best_threshold_ = best_threshold

        self.logger.info(
            f"✓ Best LogReg: {best_params}, Val F1={best_f1:.4f}, T={best_threshold:.2f}"
        )

        del X_train, y_train, X_val, y_val
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def evaluate(self, model, test_loader):
        if self.clf is None:
            self.logger.warning("LogReg not trained. Skip.")
            return {}

        self.logger.info("Collecting features for LogReg testing...")
        X_test, y_test = self._collect_features(model, test_loader)

        if X_test is None:
            self.logger.warning("No test features for LogReg.")
            return {}

        prob = self.clf.predict_proba(X_test)[:, 1]
        threshold = getattr(self, 'best_threshold_', 0.5)
        preds = (prob >= threshold).astype(np.int64)
        labels = y_test

        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        except ValueError:
            tn = fp = fn = tp = 0

        metrics = {
            'accuracy': accuracy_score(labels, preds) if len(labels) > 0 else 0.0,
            'precision': precision_score(labels, preds, zero_division=0) if len(labels) > 0 else 0.0,
            'recall': recall_score(labels, preds, zero_division=0) if len(labels) > 0 else 0.0,
            'f1': f1_score(labels, preds, zero_division=0) if len(labels) > 0 else 0.0,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        }

        try:
            metrics['auc'] = roc_auc_score(labels, prob)
        except Exception:
            metrics['auc'] = 0.0

        self._save_predictions_and_cm(labels, preds, prob, metrics, prefix="logreg")

        # 打印结果
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TEST RESULTS (Graph + LogReg)")
        self.logger.info("=" * 60)
        self.logger.info(f"Params: {self.best_params_}, Threshold: {threshold:.2f}")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"F1-Score: {metrics['f1']:.4f}")
        self.logger.info(f"AUC: {metrics['auc']:.4f}")
        self.logger.info(f"TP: {metrics['tp']:4d}  FP: {metrics['fp']:4d}")
        self.logger.info(f"FN: {metrics['fn']:4d}  TN: {metrics['tn']:4d}")
        self.logger.info("=" * 60 + "\n")

        del X_test, y_test, prob, preds, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    def _save_predictions_and_cm(self, labels, preds, prob, metrics: Dict, prefix: str = "logreg"):
        if self.output_dir is None:
            return

        results_dir = os.path.join(self.output_dir, "results")
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # prediction csv
        df_pred = pd.DataFrame({
            "idx": np.arange(len(labels)),
            "label": labels,
            "pred": preds,
            "prob_pos": prob
        })
        pred_path = os.path.join(results_dir, f"test_predictions_{prefix}.csv")
        df_pred.to_csv(pred_path, index=False)

        # confusion matrix csv
        cm_df = pd.DataFrame(
            [
                [metrics["tn"], metrics["fp"]],
                [metrics["fn"], metrics["tp"]],
            ],
            index=["Actual_0", "Actual_1"],
            columns=["Pred_0", "Pred_1"],
        )
        cm_path = os.path.join(results_dir, f"confusion_matrix_{prefix}.csv")
        cm_df.to_csv(cm_path, index=False)


# ========================================================================
# ExperimentManager：消融 + 结果汇总 + 图像绘制
# ========================================================================

class ExperimentManager:
    def __init__(self, config: ExperimentConfig, logger):
        self.config = config
        self.logger = logger
        self.all_results = {}

    def run_all_experiments(self):
        self.logger.info("\n" + "="*80)
        self.logger.info("STARTING ALL EXPERIMENTS (REBORN VERSION + UnixCoder global func_emb)")
        self.logger.info("="*80 + "\n")

        self.logger.info("STEP 1: Loading Data")
        data_loader = CPGDataLoader(self.config, self.logger)
        train_loader, val_loader, test_loader = data_loader.load_all_data()

        num_relations = self.config.num_relations

        # 主模型
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 2: Main Experiment (Full Reborn Model)")
        self.logger.info("="*80 + "\n")
        main_results = self._run_single_experiment(
            "full_model",
            train_loader, val_loader, test_loader,
            num_relations=num_relations,
            use_node_mil=True,
            use_tair=True,
            use_curriculum=True,
            use_pruning=True,
            use_contrastive=True
        )
        self.all_results['full_model'] = main_results

        # 消融
        if self.config.run_ablation:
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 3: Ablation Studies")
            self.logger.info("="*80 + "\n")
            self._run_ablation_studies(train_loader, val_loader, test_loader, num_relations)

        # 汇总报告
        self.logger.info("\n" + "="*80)
        self.logger.info("STEP 4: Generating Reports")
        self.logger.info("="*80 + "\n")
        self._generate_all_reports()

        self.logger.info("\n" + "="*80)
        self.logger.info("ALL EXPERIMENTS COMPLETED!")
        self.logger.info("="*80 + "\n")

        del train_loader, val_loader, test_loader, data_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.all_results

    def _run_single_experiment(
        self, exp_name: str,
        train_loader, val_loader, test_loader,
        num_relations: int,
        use_node_mil=True,
        use_tair=True,
        use_curriculum=True,
        use_pruning=True,
        use_contrastive=True
    ):
        self.logger.info(f"Running experiment: {exp_name}")
        self.logger.info(
            f"  node_mil={use_node_mil}, tair={use_tair}, "
            f"curriculum={use_curriculum}, pruning={use_pruning}, "
            f"contrastive={use_contrastive}"
        )

        sub_output_dir = os.path.join(self.config.output_dir, exp_name)

        exp_config = ExperimentConfig(
            dataset_name=self.config.dataset_name,
            data_root=self.config.data_root,
            random_seed=self.config.random_seed,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_epochs,
            patience=self.config.patience,
            output_dir=sub_output_dir,
            auto_subdir=False,
            encoder_type=self.config.encoder_type,

            use_onehot=self.config.use_onehot,
            onehot_dim=self.config.onehot_dim,
            onehot_max_tokens=self.config.onehot_max_tokens,

            use_node_mil=use_node_mil,
            use_tair=use_tair,
            use_curriculum=use_curriculum,
            use_pruning=use_pruning,
            use_contrastive=use_contrastive,
            contrastive_temp=self.config.contrastive_temp,
            contrastive_weight=self.config.contrastive_weight,

            use_dice_loss=self.config.use_dice_loss,
            dice_weight=self.config.dice_weight,
            ce_weight=self.config.ce_weight,
            dice_smooth=self.config.dice_smooth,
            use_focal_loss=self.config.use_focal_loss,
            focal_alpha=self.config.focal_alpha,
            focal_gamma=self.config.focal_gamma,
            max_class_weight=self.config.max_class_weight,
            use_smote_tomek=self.config.use_smote_tomek,
            smote_strategy=self.config.smote_strategy,

            num_relations=num_relations,
            run_ablation=False,
            run_interpretability=False,
            use_graph_svm=self.config.use_graph_svm,

            # 关键：继承 UnixCoder 函数级配置
            use_global_func=self.config.use_global_func,
            global_func_dim=self.config.global_func_dim,
            unixcoder_seq_cache_root=self.config.unixcoder_seq_cache_root,
            unixcoder_max_len=self.config.unixcoder_max_len,
        )

        Path(exp_config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(exp_config.output_dir, "models")).mkdir(exist_ok=True)

        model = CausalVulModel(exp_config, num_relations=num_relations)

        trainer = Trainer(model, exp_config, self.logger)
        history = trainer.train(train_loader, val_loader)

        self._save_history_csv(history, exp_config, tag="mlp")

        # 载入最佳模型
        model_path = os.path.join(exp_config.output_dir, "models", "best_model.pth")
        best_threshold = 0.5
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=exp_config.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_threshold = float(checkpoint.get('best_threshold', 0.5))

        evaluator = Evaluator(model, exp_config, self.logger, threshold=best_threshold)
        test_results = evaluator.evaluate(test_loader)

        svm_test_results = {}
        if exp_config.use_graph_svm:
            self.logger.info("===== Training & Evaluating Graph LogReg =====")
            graph_svm = GraphSVMClassifier(exp_config, self.logger, device=exp_config.device, output_dir=exp_config.output_dir)
            graph_svm.fit(model, train_loader, val_loader)
            svm_test_results = graph_svm.evaluate(model, test_loader)

        del trainer, evaluator, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'config': {
                'use_node_mil': use_node_mil,
                'use_tair': use_tair,
                'use_curriculum': use_curriculum,
                'use_pruning': use_pruning,
                'use_contrastive': use_contrastive,
                'best_threshold': best_threshold,
                'use_graph_svm': exp_config.use_graph_svm,
                'encoder_type': exp_config.encoder_type
            },
            'history': history,
            'test_results': test_results,
            'svm_test_results': svm_test_results
        }

    def _save_history_csv(self, history: dict, exp_config: ExperimentConfig, tag: str = "mlp"):
        if not history or 'train_loss' not in history:
            return
        num_epochs = len(history['train_loss'])
        data = {'epoch': list(range(1, num_epochs + 1))}
        for k, v in history.items():
            if isinstance(v, list) and len(v) == num_epochs:
                data[k] = v
        df = pd.DataFrame(data)
        results_dir = os.path.join(exp_config.output_dir, "results")
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(results_dir, f"training_history_{tag}.csv")
        df.to_csv(out_file, index=False)

    def _run_ablation_studies(self, train_loader, val_loader, test_loader, num_relations):
        ablations = [
            ("wo_node_mil", False, True, True, True, True),
            ("wo_tair", True, False, True, True, True),
            ("wo_curriculum", True, True, False, False, True),  # 剪枝也一起关
            ("wo_contrastive", True, True, True, True, False),
            ("baseline", False, False, False, False, False)
        ]
        for name, use_node, use_tair, use_curri, use_prune, use_contra in ablations:
            self.logger.info(f"\nAblation: {name}")
            res = self._run_single_experiment(
                name,
                train_loader, val_loader, test_loader,
                num_relations=num_relations,
                use_node_mil=use_node,
                use_tair=use_tair,
                use_curriculum=use_curri,
                use_pruning=use_prune,
                use_contrastive=use_contra
            )
            self.all_results[name] = res

    def _generate_all_reports(self):
        # 保存汇总 JSON
        results_file = os.path.join(self.config.output_dir, "results", "all_results.json")
        Path(os.path.dirname(results_file)).mkdir(parents=True, exist_ok=True)

        clean_results = {}
        for exp_name, exp_results in self.all_results.items():
            clean_results[exp_name] = {
                'config': exp_results.get('config', {}),
                'test_results': exp_results.get('test_results', {}),
                'svm_test_results': exp_results.get('svm_test_results', {})
            }

        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2)

        self.logger.info(f"✓ Results saved to: {results_file}")

        self._generate_comparison_table()

        if self.config.save_plots:
            plots_dir = os.path.join(self.config.output_dir, "plots")
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
            try:
                self._plot_performance_comparison(plots_dir)
                self._plot_training_curves(plots_dir)
                self._plot_confusion_matrices(plots_dir)
                self.logger.info(f"✓ All plots saved to: {plots_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to generate plots: {e}")

    def _generate_comparison_table(self):
        rows = []
        for exp_name, exp_results in self.all_results.items():
            t = exp_results.get('test_results', {})
            s = exp_results.get('svm_test_results', {}) or {}
            rows.append({
                'Experiment': exp_name,
                'Acc_MLP': f"{t.get('accuracy', 0):.4f}",
                'F1_MLP': f"{t.get('f1', 0):.4f}",
                'AUC_MLP': f"{t.get('auc', 0):.4f}",
                'Acc_LogReg': f"{s.get('accuracy', 0):.4f}" if s else "",
                'F1_LogReg': f"{s.get('f1', 0):.4f}" if s else "",
                'AUC_LogReg': f"{s.get('auc', 0):.4f}" if s else "",
            })
        df = pd.DataFrame(rows)
        results_dir = os.path.join(self.config.output_dir, "results")
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        csv_file = os.path.join(results_dir, "comparison_table.csv")
        df.to_csv(csv_file, index=False)
        latex_file = os.path.join(results_dir, "comparison_table.tex")
        df.to_latex(latex_file, index=False)
        self.logger.info("\nComparison Table:")
        self.logger.info("\n" + df.to_string(index=False))
        self.logger.info(f"✓ Table saved to: {csv_file}")

    def _plot_performance_comparison(self, plots_dir):
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        data = {m: [] for m in metrics}
        exp_names = list(self.all_results.keys())

        for exp_name in exp_names:
            t = self.all_results[exp_name].get('test_results', {})
            for m in metrics:
                data[m].append(t.get(m, 0))

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(range(len(exp_names)), data[metric], alpha=0.8)
            ax.set_xticks(range(len(exp_names)))
            ax.set_xticklabels(exp_names, rotation=45, ha='right')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., h,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "performance_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_training_curves(self, plots_dir):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        for exp_name, exp_results in self.all_results.items():
            history = exp_results.get('history', {})
            if 'train_loss' in history and len(history['train_loss']) > 0:
                axes[0].plot(history['train_loss'], label=f'{exp_name} (train)', alpha=0.7)
                axes[0].plot(history['val_loss'], label=f'{exp_name} (val)', linestyle='--', alpha=0.7)
                axes[1].plot(history['val_f1'], label=exp_name, alpha=0.7)

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('Validation F1-Score')
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrices(self, plots_dir):
        n_experiments = len(self.all_results)
        ncols = min(3, n_experiments)
        nrows = (n_experiments + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        if n_experiments == 1:
            axes = [axes]
        elif nrows == 1:
            axes = list(axes)
        else:
            axes = axes.ravel()

        for idx, (exp_name, exp_results) in enumerate(self.all_results.items()):
            t = exp_results.get('test_results', {})
            cm = np.array([
                [t.get('tn', 0), t.get('fp', 0)],
                [t.get('fn', 0), t.get('tp', 0)]
            ])

            ax = axes[idx]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Normal', 'Vulnerable'],
                        yticklabels=['Normal', 'Vulnerable'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{exp_name}\n(F1: {t.get("f1", 0):.4f})')

        for idx in range(n_experiments, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "confusion_matrices.png"), dpi=300, bbox_inches='tight')
        plt.close()


# ========================================================================
# main 函数：命令行入口
# ========================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CausalVulDetect Reborn - Graph Imbalance & Curriculum GNN + UnixCoder global func_emb')
    parser.add_argument('--dataset', type=str, default='Reveal',
                        choices=['Reveal', 'FFMPeg', 'SVulD', 'Devign', 'DiverseVul', 'VDISC'],
                        help='Dataset name')
    parser.add_argument('--encoder', type=str, default='unixcoder',
                        choices=['codebert', 'unixcoder'],
                        help='Encoder: codebert or unixcoder')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--no-ablation', action='store_true', help='Skip ablation')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache (for this encoder)')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = ExperimentConfig(
        dataset_name=args.dataset,
        encoder_type=args.encoder,
        random_seed=args.seed,
        num_epochs=args.epochs,
        run_ablation=not args.no_ablation,
        run_interpretability=False,
        # 默认使用 UnixCoder 函数级编码。如果你想关闭，可以改成 False
        use_global_func=(args.encoder == 'unixcoder')
    )

    if args.clear_cache:
        import shutil
        if os.path.exists(config.cache_dir_encoder):
            shutil.rmtree(config.cache_dir_encoder)
            Path(config.cache_dir_encoder).mkdir(parents=True, exist_ok=True)
            print(f"✓ Cache cleared: {config.cache_dir_encoder}")

    logger = setup_logging(config)
    set_seed(config.random_seed)

    # 保存配置
    config_file = os.path.join(config.output_dir, "config.json")
    Path(os.path.dirname(config_file)).mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)

    try:
        exp_manager = ExperimentManager(config, logger)
        results = exp_manager.run_all_experiments()

        logger.info("\n" + "="*80)
        logger.info("🎉 SUCCESS! All experiments completed.")
        logger.info(f"Results saved to: {config.output_dir}")
        logger.info("="*80 + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return 0

    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("❌ FAILED! Experiment crashed.")
        logger.error(f"Error: {str(e)}")
        logger.error("="*80 + "\n")
        import traceback
        logger.error(traceback.format_exc())

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return 1


if __name__ == "__main__":
    sys.exit(main())
