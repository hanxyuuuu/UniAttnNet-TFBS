#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-modal TF–TFBS model with MSTC + Cross-Attention + Avg+Max pooling (V2 Pro Modified)

修改说明：
在原 V2 Pro 基础上增加了 --external_csv 参数。
如果指定了该参数，训练结束后会自动加载外部数据集，使用最佳模型权重进行推理，
并保存外部测试集的详细评估结果 (Overall + Per-TF)。
"""

import os
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from torch.optim.lr_scheduler import CosineAnnealingLR


# ===================== 编码工具 =====================

BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")  # 20 aa
AA2IDX = {aa: i for i, aa in enumerate(AA_LIST)}
UNK_AA_IDX = 20
AA_VOCAB_SIZE = 21  # 20 aa + 1 UNK/Pad


def one_hot_encode_dna(seq: str, max_len: int = 101) -> np.ndarray:
    """将 101bp DNA 序列 one-hot 成 (4, max_len)"""
    seq = seq.upper()
    L = len(seq)
    if L != max_len:
        if L > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + "N" * (max_len - L)

    arr = np.zeros((4, max_len), dtype=np.float32)
    for i, ch in enumerate(seq):
        idx = BASE2IDX.get(ch, None)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


def one_hot_encode_protein(seq: str, max_len: int = 800) -> np.ndarray:
    """将蛋白质氨基酸序列 one-hot 成 (21, max_len)"""
    seq = seq.upper().replace(" ", "")
    L = len(seq)
    if L > max_len:
        seq = seq[:max_len]

    arr = np.zeros((AA_VOCAB_SIZE, max_len), dtype=np.float32)
    for i in range(max_len):
        if i < len(seq):
            aa = seq[i]
            idx = AA2IDX.get(aa, UNK_AA_IDX)
        else:
            idx = UNK_AA_IDX
        arr[idx, i] = 1.0
    return arr


def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===================== Dataset =====================

class TFBSMultiModalDataset(Dataset):
    """
    从 DataFrame 中读取:
        TF_symbol: 转录因子名称（可用于 per-TF 统计）
        seq_101bp: DNA 序列 (长度 101)
        Protein_seq: 氨基酸序列
        label: 0/1
    返回:
        dna_x: (4, 101) float32
        prot_x: (21, max_prot_len) float32
        y: scalar float32
        tf_symbol: str
    """
    def __init__(self, df: pd.DataFrame, max_prot_len: int = 800):
        self.seqs = df["seq_101bp"].astype(str).tolist()
        self.prots = df["Protein_seq"].fillna("").astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.max_prot_len = max_prot_len

        if "TF_symbol" in df.columns:
            self.tf_symbols = df["TF_symbol"].astype(str).tolist()
        else:
            # 若没有 TF_symbol 列，用占位符
            self.tf_symbols = ["UNK"] * len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dna_seq = self.seqs[idx]
        prot_seq = self.prots[idx]
        label = self.labels[idx]
        tf_symbol = self.tf_symbols[idx]

        dna_x = one_hot_encode_dna(dna_seq, max_len=101)
        prot_x = one_hot_encode_protein(prot_seq, max_len=self.max_prot_len)

        dna_x = torch.from_numpy(dna_x)        # (4, 101)
        prot_x = torch.from_numpy(prot_x)      # (21, L)
        y = torch.tensor(label, dtype=torch.float32)
        return dna_x, prot_x, y, tf_symbol


# ===================== 多尺度卷积模块 =====================

class MultiScaleConv1D(nn.Module):
    """
    多尺度卷积块:
    - 并联多个 Conv1d(kernel_size 不同)，padding 使得长度不变
    - 每个分支: Conv1d -> BN -> ReLU
    - 输出在通道维 concat: (B, C*len(kernels), L)
    """
    def __init__(self, in_channels, branch_channels=64,
                 kernel_sizes=(3, 7, 11), dilation=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for k in kernel_sizes:
            padding = (k // 2) * dilation
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=k,
                padding=padding,
                dilation=dilation,
            )
            bn = nn.BatchNorm1d(branch_channels)
            self.convs.append(conv)
            self.bns.append(bn)

        self.out_channels = branch_channels * len(kernel_sizes)

    def forward(self, x):
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            y = conv(x)
            y = bn(y)
            y = F.relu(y)
            outs.append(y)
        return torch.cat(outs, dim=1)  # (B, out_channels, L)


# ===================== 多模态模型：MSTC + Cross-Attn =====================

class MultiModalMSTC_CrossAttn(nn.Module):
    """
    - DNA: one-hot -> MSTC -> Conv1x1(128) -> BN -> ReLU
           -> x_d (B,128,L)
           -> H = x_d^T (B,L,128) 作为 cross-attn 的 K/V
           -> GlobalAvgPool(x_d) + GlobalMaxPool(x_d) -> x1 ∈ R^256

    - Protein: one-hot -> MSTC -> Conv1x1(128) -> BN -> ReLU -> x_p(B,128,Lp)
           -> GlobalAvgPool + GlobalMaxPool -> x2_global ∈ R^256
           -> 其中 x2_query = x2_max ∈ R^128 用作 cross-attn 的 query

    - Cross-attention:
           Q = x2_query, K=H, V=H
           -> z ∈ R^128 (TF-aware DNA summary)

    - 融合:
           fused = concat(z, x1, x2_global) ∈ R^640
           -> FC -> logit
    """
    def __init__(self,
                 dna_channels: int = 4,
                 prot_channels: int = AA_VOCAB_SIZE,
                 dna_branch_channels: int = 64,
                 prot_branch_channels: int = 64,
                 dna_kernels=(5, 9, 13),
                 prot_kernels=(9, 15, 21),
                 attn_heads: int = 4):
        super().__init__()

        # DNA 分支
        self.dna_mstc = MultiScaleConv1D(
            in_channels=dna_channels,
            branch_channels=dna_branch_channels,
            kernel_sizes=dna_kernels,
            dilation=1,
        )
        dna_mstc_out = self.dna_mstc.out_channels  # 64*3=192

        self.dna_conv1x1 = nn.Conv1d(dna_mstc_out, 128, kernel_size=1)
        self.dna_bn = nn.BatchNorm1d(128)
        self.dna_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dna_max_pool = nn.AdaptiveMaxPool1d(1)

        # Protein 分支
        self.prot_mstc = MultiScaleConv1D(
            in_channels=prot_channels,
            branch_channels=prot_branch_channels,
            kernel_sizes=prot_kernels,
            dilation=1,
        )
        prot_mstc_out = self.prot_mstc.out_channels

        self.prot_conv1x1 = nn.Conv1d(prot_mstc_out, 128, kernel_size=1)
        self.prot_bn = nn.BatchNorm1d(128)
        self.prot_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.prot_max_pool = nn.AdaptiveMaxPool1d(1)

        # Cross-attention: TF embedding (query) 与 DNA features (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=attn_heads,
            dropout=0.2,
            batch_first=True,  # (B,L,E)
        )

        # 分类头: 输入维度 = z(128) + x1(256) + x2_global(256) = 640
        self.fc = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, dna_x, prot_x):
        # dna_x:  (B,4,101)
        # prot_x: (B,21,Lp)

        # ----- DNA 分支 -----
        x_d = self.dna_mstc(dna_x)          # (B, C_mstc, L)
        x_d = self.dna_conv1x1(x_d)         # (B,128,L)
        x_d = self.dna_bn(x_d)
        x_d = F.relu(x_d)

        H = x_d.transpose(1, 2)             # (B,L,128) for attention

        x1_avg = self.dna_avg_pool(x_d).squeeze(-1)  # (B,128)
        x1_max = self.dna_max_pool(x_d).squeeze(-1)  # (B,128)
        x1 = torch.cat([x1_avg, x1_max], dim=1)      # (B,256)

        # ----- Protein 分支 -----
        x_p = self.prot_mstc(prot_x)        # (B,C_mstc,Lp)
        x_p = self.prot_conv1x1(x_p)        # (B,128,Lp)
        x_p = self.prot_bn(x_p)
        x_p = F.relu(x_p)

        x2_avg = self.prot_avg_pool(x_p).squeeze(-1)     # (B,128)
        x2_max = self.prot_max_pool(x_p).squeeze(-1)     # (B,128)
        x2_global = torch.cat([x2_avg, x2_max], dim=1)   # (B,256)

        # 用 x2_max 作为 query（更偏向“激活”模式）
        x2_query = x2_max                           # (B,128)

        # ----- Cross-attention: Q = TF, K/V = DNA -----
        q = x2_query.unsqueeze(1)                   # (B,1,128)
        k = H                                       # (B,L,128)
        v = H

        z, _ = self.cross_attn(q, k, v)             # (B,1,128)
        z = z.squeeze(1)                            # (B,128)

        # ----- 融合 & 分类 -----
        fused = torch.cat([z, x1, x2_global], dim=1)  # (B,640)
        logit = self.fc(fused).squeeze(-1)           # (B,)
        return logit


# ===================== Metrics =====================

def compute_metrics(y_true, y_prob):
    """
    统一计算指标:
      - ACC
      - F1
      - MCC
      - AUROC
      - PRAUC
      - Recall (敏感度/召回)
      - Specificity (特异性)
      - Precision (预测精度/精确率)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}

    try:
        metrics["ACC"] = accuracy_score(y_true, y_pred)
    except Exception:
        metrics["ACC"] = np.nan

    try:
        metrics["F1"] = f1_score(y_true, y_pred)
    except Exception:
        metrics["F1"] = np.nan

    try:
        metrics["MCC"] = matthews_corrcoef(y_true, y_pred)
    except Exception:
        metrics["MCC"] = np.nan

    try:
        metrics["AUROC"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["AUROC"] = np.nan

    try:
        metrics["PRAUC"] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics["PRAUC"] = np.nan

    # Precision / Recall / Specificity
    try:
        metrics["Precision"] = precision_score(y_true, y_pred)
    except Exception:
        metrics["Precision"] = np.nan

    try:
        metrics["Recall"] = recall_score(y_true, y_pred)
    except Exception:
        metrics["Recall"] = np.nan

    try:
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        metrics["Specificity"] = spec
    except Exception:
        metrics["Specificity"] = np.nan

    return metrics


# ===================== Train / Eval =====================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx):
    model.train()
    running_loss = 0.0
    total = 0

    all_true = []
    all_prob = []

    pbar = tqdm(loader, desc=f"Train epoch {epoch_idx}", leave=False)
    for batch in pbar:
        # 解包 batch（包含 tf_symbol，但训练阶段不用）
        dna_x, prot_x, y, _ = batch

        dna_x = dna_x.to(device)
        prot_x = prot_x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(dna_x, prot_x)
        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        all_prob.append(prob)
        all_true.append(y.detach().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / total
    all_true = np.concatenate(all_true)
    all_prob = np.concatenate(all_prob)

    metrics = compute_metrics(all_true, all_prob)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, epoch_idx, split_name="Val"):
    model.eval()
    running_loss = 0.0
    total = 0

    all_true = []
    all_prob = []

    pbar = tqdm(loader, desc=f"{split_name} epoch {epoch_idx}", leave=False)
    for batch in pbar:
        dna_x, prot_x, y, _ = batch

        dna_x = dna_x.to(device)
        prot_x = prot_x.to(device)
        y = y.to(device)

        logits = model(dna_x, prot_x)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        all_prob.append(prob)
        all_true.append(y.detach().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / total
    all_true = np.concatenate(all_true)
    all_prob = np.concatenate(all_prob)

    metrics = compute_metrics(all_true, all_prob)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def eval_test_per_tf(model, loader, device):
    """
    使用 best 模型在 Test 集上按 TF_symbol 统计指标。
    返回一个 DataFrame，每行一个 TF。
    """
    model.eval()

    all_tf = []
    all_true = []
    all_prob = []
    all_loss = []

    pbar = tqdm(loader, desc="Per-TF Analysis", leave=False)
    for batch in pbar:
        dna_x, prot_x, y, tf_symbol = batch

        dna_x = dna_x.to(device)
        prot_x = prot_x.to(device)
        y = y.to(device)

        logits = model(dna_x, prot_x)
        prob = torch.sigmoid(logits)

        loss_vec = F.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        )

        all_tf.extend(list(tf_symbol))
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_prob.extend(prob.detach().cpu().numpy().tolist())
        all_loss.extend(loss_vec.detach().cpu().numpy().tolist())

    df = pd.DataFrame(
        {
            "TF_symbol": all_tf,
            "y_true": all_true,
            "y_prob": all_prob,
            "loss_sample": all_loss,
        }
    )

    rows = []
    for tf_name, g in df.groupby("TF_symbol"):
        y_true_tf = g["y_true"].values
        y_prob_tf = g["y_prob"].values
        loss_tf = g["loss_sample"].mean()

        m = compute_metrics(y_true_tf, y_prob_tf)

        rows.append(
            {
                "TF_symbol": tf_name,
                "n_samples": len(g),
                "loss": loss_tf,
                "ACC": m["ACC"],
                "F1": m["F1"],
                "MCC": m["MCC"],
                "PRAUC": m["PRAUC"],
                "AUROC": m["AUROC"],
                "Recall": m["Recall"],
                "Specificity": m["Specificity"],
                "Precision": m["Precision"],
            }
        )

    tf_df = pd.DataFrame(rows)
    return tf_df


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=r"D:\project\NT\AnimalTFDB\tfbs_101bp_pos_neg_4cols.csv",
        help="数据集 CSV 路径 (需要包含: TF_symbol, seq_101bp, Protein_seq, label)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=-1,
        help="从大文件读取前多少行；<=0 表示用全数据集",
    )
    parser.add_argument(
        "--max_prot_len",
        type=int,
        default=800,
        help="蛋白序列最大长度，太长截断，太短 padding",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="EarlyStopping patience（Val AUROC 无提升的 epoch 数）",
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="multimodal_mstc_crossattn_metrics_avgmax_tfavgmax_es.csv",
        help="保存每个 epoch 指标的 CSV 文件名",
    )
    # ===== 新增参数：外部测试集 =====
    parser.add_argument(
        "--external_csv",
        type=str,
        default=None,
        help="外部测试集 CSV 路径 (可选，格式需与训练集一致)",
    )
    args = parser.parse_args()

    # Windows 下强制 num_workers=0 避免多进程问题
    if os.name == "nt":
        args.num_workers = 0

    set_seed(2025)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # ---------- 读取数据 ----------
    if args.sample_size is not None and args.sample_size > 0:
        print(f"读取数据子集 (前 {args.sample_size} 行)...")
        df = pd.read_csv(args.csv, nrows=args.sample_size)
    else:
        print("读取全数据集...")
        df = pd.read_csv(args.csv)

    print("总行数：", df.shape[0])
    print("示例几行：")
    print(df.head())

    # 打乱 & 8:1:1 划分
    df = df.sample(frac=1.0, random_state=2025).reset_index(drop=True)

    n = len(df)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train : n_train + n_val]
    df_test = df.iloc[n_train + n_val :]

    print(f"划分：train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    train_dataset = TFBSMultiModalDataset(df_train, max_prot_len=args.max_prot_len)
    val_dataset = TFBSMultiModalDataset(df_val, max_prot_len=args.max_prot_len)
    test_dataset = TFBSMultiModalDataset(df_test, max_prot_len=args.max_prot_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---------- 构建模型 ----------
    model = MultiModalMSTC_CrossAttn(
        dna_channels=4,
        prot_channels=AA_VOCAB_SIZE,
        dna_branch_channels=64,
        prot_branch_channels=64,
        dna_kernels=(5, 9, 13),
        prot_kernels=(9, 15, 21),
        attn_heads=4,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    metrics_history = []
    best_val_auroc = -1.0
    best_state = None
    best_epoch = -1
    no_improve_epochs = 0

    # ---------- 训练循环 ----------
    for epoch in range(1, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(
            "Train: "
            f"loss={train_metrics['loss']:.4f}, "
            f"ACC={train_metrics['ACC']:.4f}, "
            f"F1={train_metrics['F1']:.4f}, "
            f"MCC={train_metrics['MCC']:.4f}, "
            f"AUROC={train_metrics['AUROC']:.4f}, "
            f"PRAUC={train_metrics['PRAUC']:.4f}, "
            f"Recall={train_metrics['Recall']:.4f}, "
            f"Spec={train_metrics['Specificity']:.4f}, "
            f"Prec={train_metrics['Precision']:.4f}"
        )

        val_metrics = eval_one_epoch(
            model, val_loader, criterion, device, epoch, split_name="Val"
        )
        print(
            "Val  : "
            f"loss={val_metrics['loss']:.4f}, "
            f"ACC={val_metrics['ACC']:.4f}, "
            f"F1={val_metrics['F1']:.4f}, "
            f"MCC={val_metrics['MCC']:.4f}, "
            f"AUROC={val_metrics['AUROC']:.4f}, "
            f"PRAUC={val_metrics['PRAUC']:.4f}, "
            f"Recall={val_metrics['Recall']:.4f}, "
            f"Spec={val_metrics['Specificity']:.4f}, "
            f"Prec={val_metrics['Precision']:.4f}"
        )

        # 记录 metrics
        for split_name, m in [("train", train_metrics), ("val", val_metrics)]:
            metrics_history.append(
                {
                    "epoch": epoch,
                    "split": split_name,
                    "loss": m["loss"],
                    "ACC": m["ACC"],
                    "F1": m["F1"],
                    "MCC": m["MCC"],
                    "AUROC": m["AUROC"],
                    "PRAUC": m["PRAUC"],
                    "Recall": m["Recall"],
                    "Specificity": m["Specificity"],
                    "Precision": m["Precision"],
                }
            )

        # 更新 best (根据 Val AUROC)
        cur_auroc = val_metrics["AUROC"]
        improved = False
        if (
            cur_auroc is not None
            and not np.isnan(cur_auroc)
            and cur_auroc > best_val_auroc
        ):
            best_val_auroc = cur_auroc
            best_state = model.state_dict()
            best_epoch = epoch
            no_improve_epochs = 0
            improved = True
        else:
            no_improve_epochs += 1
            print(f"Val AUROC 无提升次数: {no_improve_epochs}/{args.patience}")

        scheduler.step()

        if (not improved) and (no_improve_epochs >= args.patience):
            print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
            break

# ---------- 用 best 模型评估 Test ----------
    if best_state is not None:
        model.load_state_dict(best_state)
        final_test_metrics = eval_one_epoch(
            model,
            test_loader,
            criterion,
            device,
            best_epoch,
            split_name="Test(best)",
        )
        print(
            f"\n>>> 最优 epoch = {best_epoch}, "
            f"best Val AUROC = {best_val_auroc:.4f}"
        )
        print(">>> 对应模型在 Test 上的性能：")
        print(
            "Test(best): "
            f"loss={final_test_metrics['loss']:.4f}, "
            f"ACC={final_test_metrics['ACC']:.4f}, "
            f"F1={final_test_metrics['F1']:.4f}, "
            f"MCC={final_test_metrics['MCC']:.4f}, "
            f"AUROC={final_test_metrics['AUROC']:.4f}, "
            f"PRAUC={final_test_metrics['PRAUC']:.4f}, "
            f"Recall={final_test_metrics['Recall']:.4f}, "
            f"Spec={final_test_metrics['Specificity']:.4f}, "
            f"Prec={final_test_metrics['Precision']:.4f}"
        )

        metrics_history.append(
            {
                "epoch": best_epoch,
                "split": "test_best",
                "loss": final_test_metrics["loss"],
                "ACC": final_test_metrics["ACC"],
                "F1": final_test_metrics["F1"],
                "MCC": final_test_metrics["MCC"],
                "AUROC": final_test_metrics["AUROC"],
                "PRAUC": final_test_metrics["PRAUC"],
                "Recall": final_test_metrics["Recall"],
                "Specificity": final_test_metrics["Specificity"],
                "Precision": final_test_metrics["Precision"],
            }
        )

        # ===== 按 TF 统计 Test(best) 指标 =====
        tf_metrics_df = eval_test_per_tf(model, test_loader, device)
        tf_metrics_df = tf_metrics_df.sort_values("TF_symbol")

        print("\n>>> 按 TF_symbol 统计 Test(best) 指标：")
        print(tf_metrics_df.to_string(index=False))

        # 保存 per-TF 指标表
        per_tf_out = os.path.join(
            os.path.dirname(args.csv),
            "multimodal_mstc_crossattn_test_best_per_TF_metrics.csv",
        )
        tf_metrics_df.to_csv(per_tf_out, index=False)
        print(">>> 已保存每个 TF 的 Test(best) 指标到：", per_tf_out)

        # =========================================================
        #            【修改点】在此处保存最佳模型
        # =========================================================
        model_save_path = os.path.join(
            os.path.dirname(args.csv), 
            "best_model_v2_pro.pth"
        )
        torch.save(best_state, model_save_path)
        print(f"\n>>> [Success] 最佳模型权重已保存至: {model_save_path}")
        # =========================================================

    # ---------- 保存每个 epoch 的指标 ----------
    metrics_df = pd.DataFrame(metrics_history)
    out_path = os.path.join(os.path.dirname(args.csv), args.metrics_out)
    metrics_df.to_csv(out_path, index=False)
    print("\n已保存每个 epoch 的指标到：", out_path)

    # =========================================================
    #            新增：外部测试集评估 (External Test)
    # =========================================================
    if args.external_csv is not None and os.path.exists(args.external_csv):
        print(f"\n========== 开始外部测试集评估: {args.external_csv} ==========")

        # 1. 读取外部数据
        df_ext = pd.read_csv(args.external_csv)
        print(f"外部数据集行数: {len(df_ext)}")

        # 2. 构建 Dataset 和 DataLoader (复用相同的预处理逻辑)
        ext_dataset = TFBSMultiModalDataset(df_ext, max_prot_len=args.max_prot_len)
        ext_loader = DataLoader(
            ext_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # 测试集不需要打乱
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # 3. 计算整体指标 (Overall Metrics)
        ext_metrics = eval_one_epoch(
            model, ext_loader, criterion, device, best_epoch, split_name="External"
        )
        print(
            "External Overall: "
            f"loss={ext_metrics['loss']:.4f}, "
            f"ACC={ext_metrics['ACC']:.4f}, "
            f"F1={ext_metrics['F1']:.4f}, "
            f"MCC={ext_metrics['MCC']:.4f}, "
            f"AUROC={ext_metrics['AUROC']:.4f}, "
            f"PRAUC={ext_metrics['PRAUC']:.4f}, "
            f"Recall={ext_metrics['Recall']:.4f}, "
            f"Spec={ext_metrics['Specificity']:.4f}, "
            f"Prec={ext_metrics['Precision']:.4f}"
        )

        # 4. 按 TF 分组统计详细指标 (Per-TF Metrics)
        ext_tf_df = eval_test_per_tf(model, ext_loader, device)
        ext_tf_df = ext_tf_df.sort_values("TF_symbol")

        # 5. 保存结果
        ext_out_path = os.path.join(
            os.path.dirname(args.csv),
            "multimodal_mstc_crossattn_EXTERNAL_per_TF_metrics.csv"
        )
        ext_tf_df.to_csv(ext_out_path, index=False)
        print(f"\n>>> 外部测试集 per-TF 结果已保存至: {ext_out_path}")


if __name__ == "__main__":
    main()