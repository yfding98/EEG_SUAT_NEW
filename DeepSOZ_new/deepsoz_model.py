#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSOZ 模型 —— 忠实复现官方代码 (ctg_11_8 + txlstm_szpool)

参考：
  github.com/deeksha-ms/DeepSOZ
  code/train/szloc.py       → ctg_11_8  (Stage-2 SOZ 定位器)
  code/train/txlstm_szpool.py → txlstm_szpool (Stage-1 检测器 + Stage-2 包装)

两阶段训练：
  Stage-1 (预训练 / pretrain)
    模型  : transformer_lstm  (Transformer + BiLSTM)
    任务  : 时序癫痫检测  (每 1s 窗口 二分类: 0=非发作, 1=发作)
    损失  : CrossEntropyLoss(weight=[0.2, 0.8])
    监督  : sz_labels  [B, Nsz, T]  (每窗口 0/1)

  Stage-2 (SOZ 定位 / szloc finetune)
    模型  : ctg_11_8  (CNN + Transformer + BiGRU)
    任务  : 通道级 SOZ 定位 (19 通道多标签)
    输入  : 同 Stage-1，但 loader 截取窗口时以发作 onset 为中心
    损失  : CrossEntropy(sz) + MapLoss(attn_onset_map) + MapLoss(chn_onset_map)
    监督  : onset_map [B, 19]  (0/1 通道标签) + sz_labels [B, T]

数据格式 (官方 npy 预处理结果)：
  X       : [Nsz, T, C, L]   Nsz 次发作 × T 个 1s 窗口 × 19 通道 × 200 采样点
  Y       : [Nsz, T]         每窗口发作标签 0/1
  soz     : [19]             通道级 SOZ 二进���标签
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 基础卷积块（与官方 szloc.py 完全一致）
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    1D 卷积残差块

    官方实现：residual=True 时 h = conv(x) + x，再 LeakyReLU
    注意：官方的 forward 无论 residual 参数如何都加了 return h + x，
         这里保持与原代码完全一致。
    """

    def __init__(self, channels: int, nlayers: int,
                 kernel_size: int = 7, stride: int = 1,
                 padding: int = 3, residual: bool = True,
                 batch_norm: bool = False):
        super().__init__()
        self.residual   = residual
        self.batch_norm = batch_norm

        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=kernel_size,
                      stride=stride, padding=padding)
            for _ in range(nlayers)
        ])
        self.bns  = nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(nlayers)])
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.convs[0](x)
        if self.batch_norm:
            h = self.bns[0](h)
        for ii, conv in enumerate(self.convs[1:]):
            h = self.relu(h)
            h = conv(h)
            if self.batch_norm:
                h = self.bns[ii](h)
        if self.residual:
            h = h + x
        h = self.relu(h)
        return h + x     # 与官方代码保持一致（注意官方也 return h+x）


# ─────────────────────────────────────────────────────────────────────────────
# Stage-1：Transformer + BiLSTM 发作检测器
# ─────────────────────────────────────────────────────────────────────────────

class TransformerLSTM(nn.Module):
    """
    官方 transformer_lstm（Stage-1 预训练检测器）

    输入 : x [B, Nsz, T, C, L]
           L=200（预处理后的特征维度，官方代码直接把 200 维作为 embedding）
    输出 :
        proba [B, Nsz, T, 2]   每窗口发作 logits
        h_c   [B, Nsz, T, C, 200]  通道级特征（供 Stage-2 使用）
        sat   None（推理时）
    """

    def __init__(self, n_channels: int = 19,
                 transformer_dropout: float = 0.15,
                 device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.n_channels = n_channels

        # 通道位置嵌入：通道 0~(C-1) + 1 个多通道 token (index C)
        self.pos_encoder = nn.Embedding(n_channels + 1, 200)

        # Transformer Encoder（单层，d_model=200，8 头）
        self.tx_encoder = nn.TransformerEncoderLayer(
            200, nhead=8, dim_feedforward=256,
            batch_first=True, dropout=transformer_dropout
        )

        # 全局双向 LSTM
        self.nhidden_sz = 100
        self.multi_lstm = nn.LSTM(
            input_size=200, hidden_size=self.nhidden_sz,
            batch_first=True, bidirectional=True,
            num_layers=1, dropout=0.0
        )
        self.multi_linear = nn.Linear(2 * self.nhidden_sz, 2)

    def forward(self, x: torch.Tensor):
        """
        x: [B, Nsz, T, C, L]  L=200
        """
        B, Nsz, T, C, L = x.size()

        # 通道位置编码
        chn_pos = torch.arange(C, device=x.device)
        pos_emb = self.pos_encoder(chn_pos)[None, None, :, :]   # [1,1,C,200]
        h_c = x + pos_emb                                        # [B,Nsz,T,C,200]

        # 多通道 token 位置编码
        h_m = self.pos_encoder(
            torch.full((B * Nsz * T, 1), C, dtype=torch.long, device=x.device)
        )                                                         # [B*Nsz*T, 1, 200]

        # Transformer：将 19 通道 + 1 个全局 token 拼接后送入
        h_c_flat = h_c.reshape(B * Nsz * T, C, 200)
        tx_input = torch.cat([h_c_flat, h_m], dim=1)            # [B*T*Nsz, C+1, 200]
        tx_input = self.tx_encoder(tx_input)

        h_c = tx_input[:, :-1, :].view(B * Nsz, T, C, 200)     # 通道特征
        h_m = tx_input[:, -1, :].view(B * Nsz, T, 200)         # 全局特征

        # 全局双向 LSTM
        self.multi_lstm.flatten_parameters()
        h_m, _ = self.multi_lstm(h_m)                            # [B*Nsz, T, 200]

        logits = self.multi_linear(
            h_m.reshape(B * Nsz * T, -1)
        ).reshape(B, Nsz, T, 2)

        h_c = h_c.reshape(B, Nsz, T, C, 200)
        return logits, h_c, None


# ─────────────────────────────────────────────────────────────────────────────
# Stage-2：ctg_11_8  SOZ 定位器
# ─────────────────────────────────────────────────────────────────────────────

class DeepSOZLocator(nn.Module):
    """
    官方 ctg_11_8（Stage-2 SOZ 定位主模型）

    Pipeline（与官方 szloc.py 完全一致）：
      1. Per-channel CNN  → h_c [B,T,C,80]
      2. Multi-channel CNN → h_m [B,T,80]
      3. Transformer (src=h_m‖h_c, tgt=h_c) → h_c [B,T,C,80]
      4. Per-channel BiGRU → h_c [B,T,C,80]
      5. Global BiGRU    → h_m [B,T,80]
      6. Linear heads：
           channel_linear : h_c → [B,T,C,2]  通道发作 logits
           multi_linear   : h_m → [B,T,2]    全局发作 logits
           onset_linear   : h_m → [B,T,1]    时间注意力 (softmax over T)

    输出 (forward)：
      channel_sz_logits [B,T,2]   最活跃通道的发作 logits
      h_m               [B,T,2]   全局发作 logits
      attn_onset_map    [B,C]     注意力加权 SOZ 图
      chn_onset_map     [B,C]     通道梯度 SOZ 图
    """

    def __init__(self, n_channels: int = 19,
                 cnn_dropout: float = 0.15,
                 gru_dropout: float = 0.0,
                 transformer_dropout: float = 0.15):
        super().__init__()
        self.n_channels = n_channels

        # ── Per-channel CNN ─────────────────────────────────────────
        self.nchn_c = 80
        self.ConvEmbeddingC = nn.Conv1d(1,  10,  kernel_size=7, stride=1, padding=3)
        self.ConvC1  = ConvBlock(10,  1, residual=True, kernel_size=7, padding=3)
        self.ProjC1  = nn.Conv1d(10, 20, kernel_size=1, stride=2, padding=0)
        self.ConvC2  = ConvBlock(20,  1, residual=True, kernel_size=7, padding=3)
        self.ProjC2  = nn.Conv1d(20, 40, kernel_size=1, stride=2, padding=0)
        self.ConvC3  = ConvBlock(40,  1, residual=True, kernel_size=7, padding=3)
        self.ProjC3  = nn.Conv1d(40, 80, kernel_size=1, stride=2, padding=0)
        self.ConvC4  = ConvBlock(80,  1, residual=True, kernel_size=7, padding=3)
        self.cnn_dropout = nn.Dropout(cnn_dropout)

        # ── Multi-channel CNN ────────────────────────────────────────
        self.nchn_m = 80
        self.ConvEmbeddingM = nn.Conv1d(n_channels, 40, kernel_size=7, stride=1, padding=3)
        self.Conv1  = ConvBlock(40,  2, residual=True, kernel_size=7, padding=3)
        self.ProjM1 = nn.Conv1d(40, 80, kernel_size=1, stride=2, padding=0)
        self.Conv2  = ConvBlock(80,  2, residual=True, kernel_size=7, padding=3)
        self.ProjM2 = nn.Conv1d(80, 80, kernel_size=1, stride=2, padding=0)
        self.Conv3  = ConvBlock(80,  2, residual=True, kernel_size=7, padding=3)

        # ── Transformer（Encoder-Decoder）────────────────────────────
        # src = [h_m‖h_c]  (多通道全局 + 各通道)
        # tgt = h_c         (各通道)
        self.channel_transformer = nn.Transformer(
            80, num_encoder_layers=1, num_decoder_layers=1,
            dim_feedforward=128, batch_first=True,
            dropout=transformer_dropout
        )

        # ── Per-channel BiGRU ────────────────────────────────────────
        self.nhidden_c = 40
        self.channel_gru = nn.GRU(
            input_size=80, hidden_size=self.nhidden_c,
            batch_first=True, bidirectional=True,
            num_layers=2, dropout=gru_dropout
        )

        # ── Global BiGRU ─────────────────────────────────────────────
        self.nhidden_sz = 40
        self.multi_gru = nn.GRU(
            input_size=80, hidden_size=self.nhidden_sz,
            batch_first=True, bidirectional=True,
            num_layers=1, dropout=0.0
        )

        # ── 输出头 ───────────────────────────────────────────────────
        self.channel_linear = nn.Linear(2 * self.nhidden_c, 2)   # 通道发作
        self.multi_linear   = nn.Linear(2 * self.nhidden_sz, 2)  # 全局发作
        self.onset_linear   = nn.Linear(2 * self.nhidden_sz, 1)  # 时间注意力
        self.sig            = nn.Sigmoid()

    # ── 子编码器 ─────────────────────────────────────────────────────────────

    def _channel_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,T,C,L]  →  [B,T,C,80]"""
        B, T, C, L = x.size()
        h = self.ConvEmbeddingC(x.view(B * T * C, 1, L))
        h = self.ConvC1(h)
        h = self.ProjC1(h)
        h = self.ConvC2(h)
        h = self.ProjC2(h)
        h = self.ConvC3(h)
        h = self.ProjC3(h)
        h = self.ConvC4(h)
        # 全局��均池化（时间维度）
        h = torch.mean(h.view(B, T, C, self.nchn_c, -1), dim=4)
        return h  # [B,T,C,80]

    def _multichannel_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,T,C,L]  →  [B,T,80]"""
        B, T, C, L = x.size()
        h = x.view(B * T, C, L)
        h = self.ConvEmbeddingM(h)
        h = self.Conv1(h)
        h = self.ProjM1(h)
        h = self.Conv2(h)
        h = self.ProjM2(h)
        h = self.Conv3(h)
        h = torch.mean(h.view(B, T, self.nchn_m, -1), dim=3)
        return h  # [B,T,80]

    # ── onset map 计算（与官方一致）───────────────────────────────────────────

    def _attn_onset_map(self, h: torch.Tensor,
                        a: torch.Tensor) -> torch.Tensor:
        """
        注意力加权 onset map

        h : [B,T,C,2]  通道发作 logits
        a : [B,T,1]    时间注意力权重（已经 softmax）
        →   [B,C]
        """
        B, T, C, _ = h.shape
        probs = F.softmax(h, dim=3)                          # [B,T,C,2]
        onset_map = torch.sum(
            a.view(B, T, 1) * probs[:, :, :, 1], dim=1
        )                                                     # [B,C]
        return onset_map

    def _channel_onset_map(self, h_c: torch.Tensor) -> torch.Tensor:
        """
        通道梯度 onset map（最活跃通道的概率前向差分）

        h_c : [B,T,C,2]
        →    [B,C]
        """
        channel_probs = F.softmax(h_c, dim=3)
        max_chn_probs, _ = torch.max(channel_probs[:, :, :, 1], dim=2)  # [B,T]
        attn = F.relu(max_chn_probs[:, 1:] - max_chn_probs[:, :-1])    # [B,T-1]
        onset_map = torch.sum(
            attn.unsqueeze(2) * channel_probs[:, 1:, :, 1], dim=1
        )                                                                  # [B,C]
        return onset_map

    def _max_channel_logits(self, h: torch.Tensor) -> torch.Tensor:
        """
        取每时间步预测概率最高通道的 logits，作为全局时序判断依据

        h : [B,T,C,2]
        →  [B,T,2]
        """
        B, T, C, _ = h.shape
        probs = F.softmax(h, dim=3)
        device = h.device if h.device.type != 'cpu' else None
        max_logits = torch.zeros(B, T, 2, device=device)
        for bb in range(B):
            max_chns = torch.argmax(probs[bb, :, :, 1], dim=1)  # [T]
            for tt in range(T):
                max_logits[bb, tt, :] = h[bb, tt, max_chns[tt], :]
        return max_logits

    # ── 前向传播 ─────────────────────────────────────────────────────────────

    def forward_pass(self, x: torch.Tensor):
        """
        x: [B, Nsz, T, C, L]  或  [B, T, C, L]

        返回 (h_c, h_m, a)：
          h_c : [B,T,C,2]    通道发作 logits
          h_m : [B,T,2]      全局发作 logits
          a   : [B,T,1]      时间注意力
        """
        if x.dim() == 5:
            B, Nsz, T, C, L = x.size()
            x = x.reshape(B * Nsz, T, C, L)
        else:
            B, T, C, L = x.size()
            Nsz = 1

        BN = x.shape[0]   # B * Nsz

        # CNN 编码
        h_c = self.cnn_dropout(self._channel_encoder(x))      # [BN,T,C,80]
        h_m = self.cnn_dropout(self._multichannel_encoder(x)) # [BN,T,80]

        # Transformer：src=[h_m‖h_c], tgt=h_c
        h_c_flat = h_c.reshape(BN * T, C, 80)
        src = torch.cat([h_c_flat, h_m.reshape(BN * T, 1, 80)], dim=1)  # [BN*T, C+1, 80]
        h_c_flat = self.channel_transformer(src, h_c_flat)               # [BN*T, C, 80]
        h_c = h_c_flat.view(BN, T, C, 80)

        # Per-channel BiGRU（在时间维度建模）
        h_c = h_c.transpose(1, 2)                 # [BN, C, T, 80]
        h_c = h_c.reshape(BN * C, T, 80)
        self.channel_gru.flatten_parameters()
        h_c, _ = self.channel_gru(h_c)            # [BN*C, T, 80]
        h_c = h_c.view(BN, C, T, 2 * self.nhidden_c).transpose(1, 2)  # [BN,T,C,80]

        # Global BiGRU
        self.multi_gru.flatten_parameters()
        h_m, _ = self.multi_gru(h_m)              # [BN, T, 80]

        # Linear heads
        h_c = self.channel_linear(h_c)            # [BN,T,C,2]
        a   = torch.softmax(self.onset_linear(h_m), dim=1)  # [BN,T,1]
        h_m = self.multi_linear(h_m)              # [BN,T,2]

        return (h_c.reshape(B, Nsz, T, C, 2)[:, 0],   # [B,T,C,2]
                h_m.reshape(B, Nsz, T, 2)[:, 0],       # [B,T,2]
                a.reshape(B, Nsz, T, 1)[:, 0])          # [B,T,1]

    def forward(self, x: torch.Tensor):
        """
        x : [B, Nsz, T, C, L]

        返回：
          channel_sz_logits [B,T,2]   最活跃通道的发作 logits
          h_m               [B,T,2]   全局发作 logits
          attn_onset_map    [B,C]     注意力加权 SOZ 图
          chn_onset_map     [B,C]     通道梯度 SOZ 图
        """
        h_c, h_m, a = self.forward_pass(x)
        channel_sz_logits = self._max_channel_logits(h_c)
        attn_onset_map    = self._attn_onset_map(h_c, a)
        chn_onset_map     = self._channel_onset_map(h_c)
        return channel_sz_logits, h_m, attn_onset_map, chn_onset_map


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────��───────────────────────────────────────────────────────────────────

def build_stage1_model(n_channels: int = 19,
                       transformer_dropout: float = 0.15,
                       device: str = 'cpu') -> TransformerLSTM:
    """Stage-1 检测器（官方 transformer_lstm）"""
    return TransformerLSTM(n_channels=n_channels,
                           transformer_dropout=transformer_dropout,
                           device=device)


def build_stage2_model(n_channels: int = 19,
                       cnn_dropout: float = 0.15,
                       gru_dropout: float = 0.0,
                       transformer_dropout: float = 0.15) -> DeepSOZLocator:
    """Stage-2 SOZ 定位器（官方 ctg_11_8）"""
    return DeepSOZLocator(
        n_channels=n_channels,
        cnn_dropout=cnn_dropout,
        gru_dropout=gru_dropout,
        transformer_dropout=transformer_dropout,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 快速冒烟测试
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B, Nsz, T, C, L = 2, 3, 45, 19, 200

    # Stage-1
    m1 = build_stage1_model()
    x  = torch.randn(B, Nsz, T, C, L)
    logits, hc, _ = m1(x)
    print(f'[Stage-1] logits={logits.shape}  h_c={hc.shape}')
    # → [2,3,45,2]  [2,3,45,19,200]

    # Stage-2
    m2 = build_stage2_model()
    ch_logits, h_m, attn_map, chn_map = m2(x)
    print(f'[Stage-2] channel_sz={ch_logits.shape}  h_m={h_m.shape}'
          f'  attn_map={attn_map.shape}  chn_map={chn_map.shape}')
    # → [2,45,2]  [2,45,2]  [2,19]  [2,19]

    p1 = sum(p.numel() for p in m1.parameters())
    p2 = sum(p.numel() for p in m2.parameters())
    print(f'Stage-1 参数: {p1:,}   Stage-2 参数: {p2:,}')
