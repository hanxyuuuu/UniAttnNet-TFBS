#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone prediction / evaluation script for UniAttnNet-TFBS.

Features
--------
1. Load a trained PyTorch state_dict (.pth).
2. Run inference on a CSV file containing DNA sequence and protein sequence.
3. Save per-sample probabilities and binary predictions.
4. If labels are present, compute overall and per-TF metrics.

Required CSV columns
--------------------
- seq_101bp: DNA sequence (length 101 is recommended)
- Protein_seq: transcription factor amino-acid sequence

Optional CSV columns
--------------------
- TF_symbol: used for per-TF statistics; if missing, UNK will be used
- label: if present, overall / per-TF metrics will be computed
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm


BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA2IDX = {aa: i for i, aa in enumerate(AA_LIST)}
UNK_AA_IDX = 20
AA_VOCAB_SIZE = 21


def one_hot_encode_dna(seq: str, max_len: int = 101) -> np.ndarray:
    seq = str(seq).upper()
    if len(seq) > max_len:
        seq = seq[:max_len]
    elif len(seq) < max_len:
        seq = seq + "N" * (max_len - len(seq))

    arr = np.zeros((4, max_len), dtype=np.float32)
    for i, ch in enumerate(seq):
        idx = BASE2IDX.get(ch)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr



def one_hot_encode_protein(seq: str, max_len: int = 800) -> np.ndarray:
    seq = str(seq).upper().replace(" ", "")
    if len(seq) > max_len:
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


class TFBSInferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_prot_len: int = 800):
        required = {"seq_101bp", "Protein_seq"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.df = df.reset_index(drop=True).copy()
        self.max_prot_len = max_prot_len

        if "TF_symbol" not in self.df.columns:
            self.df["TF_symbol"] = "UNK"

        self.has_label = "label" in self.df.columns
        if not self.has_label:
            self.df["label"] = np.nan

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dna_x = torch.from_numpy(one_hot_encode_dna(row["seq_101bp"], max_len=101))
        prot_x = torch.from_numpy(one_hot_encode_protein(row["Protein_seq"], max_len=self.max_prot_len))
        tf_symbol = str(row["TF_symbol"])

        label = row["label"]
        if pd.isna(label):
            y = torch.tensor(-1.0, dtype=torch.float32)
        else:
            y = torch.tensor(float(label), dtype=torch.float32)

        return dna_x, prot_x, y, tf_symbol, idx


class MultiScaleConv1D(nn.Module):
    def __init__(self, in_channels, branch_channels=64, kernel_sizes=(3, 7, 11), dilation=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for k in kernel_sizes:
            padding = (k // 2) * dilation
            self.convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=branch_channels,
                    kernel_size=k,
                    padding=padding,
                    dilation=dilation,
                )
            )
            self.bns.append(nn.BatchNorm1d(branch_channels))
        self.out_channels = branch_channels * len(kernel_sizes)

    def forward(self, x):
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            y = F.relu(bn(conv(x)))
            outs.append(y)
        return torch.cat(outs, dim=1)


class MultiModalMSTC_CrossAttn(nn.Module):
    def __init__(
        self,
        dna_channels: int = 4,
        prot_channels: int = AA_VOCAB_SIZE,
        dna_branch_channels: int = 64,
        prot_branch_channels: int = 64,
        dna_kernels=(5, 9, 13),
        prot_kernels=(9, 15, 21),
        attn_heads: int = 4,
    ):
        super().__init__()

        self.dna_mstc = MultiScaleConv1D(
            in_channels=dna_channels,
            branch_channels=dna_branch_channels,
            kernel_sizes=dna_kernels,
            dilation=1,
        )
        self.dna_conv1x1 = nn.Conv1d(self.dna_mstc.out_channels, 128, kernel_size=1)
        self.dna_bn = nn.BatchNorm1d(128)
        self.dna_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dna_max_pool = nn.AdaptiveMaxPool1d(1)

        self.prot_mstc = MultiScaleConv1D(
            in_channels=prot_channels,
            branch_channels=prot_branch_channels,
            kernel_sizes=prot_kernels,
            dilation=1,
        )
        self.prot_conv1x1 = nn.Conv1d(self.prot_mstc.out_channels, 128, kernel_size=1)
        self.prot_bn = nn.BatchNorm1d(128)
        self.prot_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.prot_max_pool = nn.AdaptiveMaxPool1d(1)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=attn_heads,
            dropout=0.2,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, dna_x, prot_x):
        x_d = F.relu(self.dna_bn(self.dna_conv1x1(self.dna_mstc(dna_x))))
        H = x_d.transpose(1, 2)

        x1_avg = self.dna_avg_pool(x_d).squeeze(-1)
        x1_max = self.dna_max_pool(x_d).squeeze(-1)
        x1 = torch.cat([x1_avg, x1_max], dim=1)

        x_p = F.relu(self.prot_bn(self.prot_conv1x1(self.prot_mstc(prot_x))))
        x2_avg = self.prot_avg_pool(x_p).squeeze(-1)
        x2_max = self.prot_max_pool(x_p).squeeze(-1)
        x2_global = torch.cat([x2_avg, x2_max], dim=1)

        q = x2_max.unsqueeze(1)
        z, _ = self.cross_attn(q, H, H)
        z = z.squeeze(1)

        fused = torch.cat([z, x1, x2_global], dim=1)
        return self.fc(fused).squeeze(-1)



def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

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
    try:
        metrics["Precision"] = precision_score(y_true, y_pred)
    except Exception:
        metrics["Precision"] = np.nan
    try:
        metrics["Recall"] = recall_score(y_true, y_pred)
    except Exception:
        metrics["Recall"] = np.nan
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except Exception:
        metrics["Specificity"] = np.nan
    return metrics


@torch.no_grad()
def run_inference(model, loader, device, threshold=0.5):
    model.eval()
    all_probs, all_labels, all_tf, all_idx = [], [], [], []

    for dna_x, prot_x, y, tf_symbol, idx in tqdm(loader, desc="Inference", leave=False):
        dna_x = dna_x.to(device)
        prot_x = prot_x.to(device)

        logits = model(dna_x, prot_x)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(y.cpu().numpy().tolist())
        all_tf.extend(list(tf_symbol))
        all_idx.extend(idx.cpu().numpy().tolist())

    pred_df = pd.DataFrame(
        {
            "row_index": all_idx,
            "TF_symbol": all_tf,
            "y_prob": all_probs,
            "y_pred": (np.asarray(all_probs) >= threshold).astype(int),
            "label": all_labels,
        }
    ).sort_values("row_index").reset_index(drop=True)

    has_valid_label = np.isfinite(pred_df["label"]).all() and ((pred_df["label"] == 0) | (pred_df["label"] == 1)).all()
    overall_df = None
    per_tf_df = None

    if has_valid_label:
        y_true = pred_df["label"].astype(int).to_numpy()
        y_prob = pred_df["y_prob"].to_numpy()
        overall = compute_metrics(y_true, y_prob, threshold=threshold)
        overall["n_samples"] = len(pred_df)
        overall_df = pd.DataFrame([overall])

        rows = []
        for tf_name, g in pred_df.groupby("TF_symbol"):
            if g["label"].nunique(dropna=True) < 2:
                # Some TF-specific subsets may be single-class.
                m = compute_metrics(g["label"].astype(int).to_numpy(), g["y_prob"].to_numpy(), threshold=threshold)
            else:
                m = compute_metrics(g["label"].astype(int).to_numpy(), g["y_prob"].to_numpy(), threshold=threshold)
            rows.append(
                {
                    "TF_symbol": tf_name,
                    "n_samples": len(g),
                    **m,
                }
            )
        per_tf_df = pd.DataFrame(rows).sort_values("TF_symbol").reset_index(drop=True)

    return pred_df, overall_df, per_tf_df



def main():
    parser = argparse.ArgumentParser(description="Standalone prediction for UniAttnNet-TFBS")
    parser.add_argument("--input_csv", required=True, type=str, help="Input CSV file")
    parser.add_argument("--model_path", required=True, type=str, help="Path to .pth state_dict")
    parser.add_argument("--output_dir", default="outputs/predict", type=str, help="Directory for outputs")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--max_prot_len", default=800, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    dataset = TFBSInferenceDataset(df, max_prot_len=args.max_prot_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = MultiModalMSTC_CrossAttn().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=True)

    pred_df, overall_df, per_tf_df = run_inference(model, loader, device, threshold=args.threshold)

    merged_df = df.copy().reset_index(drop=True)
    merged_df["y_prob"] = pred_df["y_prob"]
    merged_df["y_pred"] = pred_df["y_pred"]

    pred_path = out_dir / "predictions.csv"
    merged_df.to_csv(pred_path, index=False)
    print(f"[Saved] predictions: {pred_path}")

    if overall_df is not None:
        overall_path = out_dir / "overall_metrics.csv"
        overall_df.to_csv(overall_path, index=False)
        print(f"[Saved] overall metrics: {overall_path}")

    if per_tf_df is not None:
        per_tf_path = out_dir / "per_tf_metrics.csv"
        per_tf_df.to_csv(per_tf_path, index=False)
        print(f"[Saved] per-TF metrics: {per_tf_path}")

    print("Done.")


if __name__ == "__main__":
    main()
