#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimeFilter_LaBraM_BrainNetwork_Integration

Full SOZ detection pipeline integrating:
  - SeizureAlignedAdaptivePatching     (Step 1)
  - LaBraM + TimeFilter                (Branch A: temporal)
  - BrainNetworkExtractor + Evolution  (Branch B: network)
  - Gated cross-modal fusion           (Step 3)
  - SOZ localization head              (Step 4)

Input:  [B, 22, window_samples]  +  onset/start metadata
Output: [B, 19] SOZ probabilities  +  auxiliary outputs
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ── Local imports (sibling modules) ──
try:
    from .seizure_aligned_patching import SeizureAlignedAdaptivePatching
    from .brain_network_extractor import MultiScaleBrainNetworkExtractor
    from .dynamic_network_evolution import DynamicNetworkEvolutionModel
    from .labram_timefilter_soz import (
        LaBraMBackbone, TimeFilterBackbone,
        GATBlock, ModelConfig as LaBraMModelConfig,
    )
    from .directed_brain_timefilter import (
        DirectedBrainTimeFilter, DirectedTimeFilterConfig,
    )
    from .bipolar_to_monopolar import BipolarToMonopolarMapper
except ImportError:
    from seizure_aligned_patching import SeizureAlignedAdaptivePatching
    from brain_network_extractor import MultiScaleBrainNetworkExtractor
    from dynamic_network_evolution import DynamicNetworkEvolutionModel
    from labram_timefilter_soz import (
        LaBraMBackbone, TimeFilterBackbone,
        GATBlock, ModelConfig as LaBraMModelConfig,
    )
    from directed_brain_timefilter import (
        DirectedBrainTimeFilter, DirectedTimeFilterConfig,
    )
    from bipolar_to_monopolar import BipolarToMonopolarMapper


# =====================================================================
# Config
# =====================================================================

@dataclass
class IntegrationConfig:
    """Full pipeline configuration."""
    # patching
    n_channels: int = 22
    patch_len: int = 200               # 对齐LaBraM patch_size
    n_pre_patches: int = 5
    n_post_patches: int = 5
    fs: float = 200.0

    # LaBraM backbone
    embed_dim: int = 200               # 对齐LaBraM-base
    n_transformer_layers: int = 12     # 对齐LaBraM-base
    n_heads_transformer: int = 10      # 对齐LaBraM-base
    n_frozen_layers: int = 10
    labram_checkpoint: str = ''
    checkpoint_type: str = 'labram-base'

    # TimeFilter (Branch A)
    tf_alpha: float = 0.15
    tf_n_heads: int = 4
    top_p: float = 0.5
    n_timefilter_blocks: int = 2
    temporal_k: int = 3
    moe_loss_weight: float = 0.01

    # GAT (kept for backward compat, unused in main forward)
    gat_layers: int = 2
    gat_heads: int = 4
    gat_dropout: float = 0.1

    # brain network
    gc_order: int = 20
    te_n_bins: int = 8

    # DirectedBrainTimeFilter (Branch B)
    brain_tf_n_blocks: int = 1
    brain_tf_n_heads: int = 4
    brain_tf_hidden: int = 64

    # evolution
    gru_hidden: int = 128
    gru_layers: int = 2
    gcn_hidden: int = 64

    # fusion
    fusion_dropout: float = 0.1

    # output
    task_mode: str = 'soz'            # 'soz' or 'stage_pretrain'
    output_mode: str = 'monopolar'     # 'monopolar' (19) or 'bipolar' (22)
    n_monopolar: int = 19
    n_regions: int = 6
    n_hemisphere_classes: int = 3
    n_stage_classes: int = 2
    stage_ignore_index: int = -100

    # loss weights
    w_transition: float = 0.3
    w_pattern: float = 0.2
    w_contrast: float = 0.1
    w_moe: float = 0.01               # MoE辅助损失权重
    w_region: float = 0.5
    w_hemisphere: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    # training strategy
    use_checkpoint: bool = False


# =====================================================================
# Focal Loss
# =====================================================================

class FocalLoss(nn.Module):
    """
    Multi-label Binary Focal Loss with proper alpha balancing.

    For each element:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where alpha_t = alpha for positives, (1-alpha) for negatives.

    Also supports dynamic pos_weight computed from label statistics.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Use raw sigmoid probabilities for pt; exp(-weighted_bce) is incorrect when pos_weight != 1.
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=self.pos_weight,
        )
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        # alpha_t: alpha for positive, (1-alpha) for negative
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - pt) ** self.gamma * bce
        
        if sample_weight is not None:
            focal = focal * sample_weight.unsqueeze(1)
            
        return focal.mean()


# =====================================================================
# Gated Cross-Modal Fusion
# =====================================================================

class GatedFusion(nn.Module):
    """gate = sigmoid(MLP(concat(temporal, network))) => weighted blend."""

    def __init__(self, temporal_dim: int, network_dim: int, dropout: float = 0.1):
        super().__init__()
        self.temporal_norm = nn.LayerNorm(temporal_dim)
        self.network_norm = nn.LayerNorm(network_dim)
        self.gate = nn.Sequential(
            nn.Linear(temporal_dim + network_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_dim, temporal_dim),
            nn.Dropout(0.2),
            nn.Sigmoid(),
        )
        self.net_proj = nn.Linear(network_dim, temporal_dim)
        self.out_norm = nn.LayerNorm(temporal_dim)

    def forward(
        self, temporal: torch.Tensor, network: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        temporal : [B, N, D_t]
        network  : [B, D_n]   (broadcast to N nodes)
        Returns  : fused [B, N, D_t],  gate_weights [B, N]
        """
        temporal = self.temporal_norm(temporal)
        network = self.network_norm(network)

        N = temporal.size(1)
        net_exp = network.unsqueeze(1).expand(-1, N, -1)     # [B, N, D_n]
        cat = torch.cat([temporal, net_exp], dim=-1)          # [B, N, D_t+D_n]
        g = self.gate(cat)                                    # [B, N, D_t]
        net_proj = self.net_proj(net_exp)                     # [B, N, D_t]
        fused = g * temporal + (1 - g) * net_proj
        gate_scalar = g.mean(dim=-1)                          # [B, N]
        return self.out_norm(fused), gate_scalar


# =====================================================================
# SOZ Head (channel pool + bipolar-to-monopolar)
# =====================================================================

class SOZHead(nn.Module):
    """Channel-level pooling + bipolar→monopolar mapping → logits."""

    def __init__(self, embed_dim: int, n_channels: int = 22,
                 n_patches: int = 10, output_mode: str = 'monopolar',
                 dropout: float = 0.3):
        super().__init__()
        self.n_channels = n_channels
        self.n_patches = n_patches
        self.output_mode = output_mode
        hidden = embed_dim
        self.pool_norm = nn.LayerNorm(embed_dim * 2)
        self.channel_fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        if output_mode == 'monopolar':
            self.b2m = BipolarToMonopolarMapper()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, N, D]  where N = n_channels * n_patches
        Returns: logits [B, n_output], bipolar_logits [B, 22]
        """
        B, N, D = x.shape
        P = N // self.n_channels
        x_4d = x.view(B, self.n_channels, P, D)              # [B, 22, P, D]
        # max pool + mean pool over patches
        max_p = x_4d.max(dim=2).values                        # [B, 22, D]
        mean_p = x_4d.mean(dim=2)                             # [B, 22, D]
        cat = torch.cat([max_p, mean_p], dim=-1)               # [B, 22, 2D]
        cat = self.pool_norm(cat)                               # normalize before MLP
        bipolar_logits = self.channel_fc(cat).squeeze(-1)      # [B, 22]

        if self.output_mode == 'monopolar':
            logits = self.b2m.forward_logits(bipolar_logits)   # [B, 19]
        else:
            logits = bipolar_logits
        return logits, bipolar_logits


class GlobalPoolHead(nn.Module):
    """Global mean/max pooling over fused tokens for auxiliary classification."""

    def __init__(self, embed_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        hidden = embed_dim
        self.pool_norm = nn.LayerNorm(embed_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_pool = x.max(dim=1).values
        mean_pool = x.mean(dim=1)
        pooled = torch.cat([max_pool, mean_pool], dim=-1)
        pooled = self.pool_norm(pooled)
        return self.mlp(pooled)


class PatchStageHead(nn.Module):
    """Predict a binary stage label for each patch."""

    def __init__(
        self,
        embed_dim: int,
        n_channels: int = 22,
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_channels = n_channels
        hidden = embed_dim
        self.pool_norm = nn.LayerNorm(embed_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D], where N = n_channels * n_patches
        returns: [B, n_patches, n_classes]
        """
        B, N, D = x.shape
        P = N // self.n_channels
        x_4d = x.view(B, self.n_channels, P, D).permute(0, 2, 1, 3)
        max_pool = x_4d.max(dim=2).values
        mean_pool = x_4d.mean(dim=2)
        pooled = torch.cat([max_pool, mean_pool], dim=-1)
        pooled = self.pool_norm(pooled)
        return self.mlp(pooled)


# =====================================================================
# Main Integration Model
# =====================================================================

class TimeFilter_LaBraM_BrainNetwork_Integration(nn.Module):
    """
    Full SOZ detection pipeline.

    Input:  raw EEG [B, 22, T] + seizure metadata
    Output: SOZ probabilities [B, 19] + auxiliary signals
    """

    def __init__(self, cfg: IntegrationConfig = None):
        super().__init__()
        self.cfg = cfg or IntegrationConfig()
        c = self.cfg
        max_patches = c.n_pre_patches + c.n_post_patches

        # ── Step 1: Adaptive patching ──
        self.patching = SeizureAlignedAdaptivePatching(
            n_channels=c.n_channels, patch_len=c.patch_len,
            n_pre_patches=c.n_pre_patches,
            n_post_patches=c.n_post_patches, fs=c.fs,
        )

        # ── Branch A: LaBraM Backbone + TimeFilter ──
        labram_cfg = LaBraMModelConfig(
            n_channels=c.n_channels, n_patches=max_patches,
            patch_len=c.patch_len, embed_dim=c.embed_dim,
            n_nodes=c.n_channels * max_patches,
            n_transformer_layers=c.n_transformer_layers,
            n_heads_transformer=c.n_heads_transformer,
            n_frozen_layers=c.n_frozen_layers,
            labram_checkpoint=c.labram_checkpoint,
            checkpoint_type=c.checkpoint_type,
            tf_alpha=c.tf_alpha,
            tf_n_heads=c.tf_n_heads,
            top_p=c.top_p,
            n_timefilter_blocks=c.n_timefilter_blocks,
            temporal_k=c.temporal_k,
            gat_dropout=c.gat_dropout,
        )
        self.backbone = LaBraMBackbone(labram_cfg)
        self.timefilter = TimeFilterBackbone(labram_cfg)

        # ── Branch B: Brain network + DirectedBrainTimeFilter ──
        self.net_extractor = MultiScaleBrainNetworkExtractor(
            n_channels=c.n_channels, patch_len=c.patch_len,
            fs=c.fs, gc_order=c.gc_order, te_n_bins=c.te_n_bins,
        )
        brain_tf_cfg = DirectedTimeFilterConfig(
            n_channels=c.n_channels, n_patches=max_patches,
            n_blocks=c.brain_tf_n_blocks, n_heads=c.brain_tf_n_heads,
            hidden_dim=c.brain_tf_hidden, dropout=c.gat_dropout,
            temporal_k=c.temporal_k,
        )
        self.brain_timefilter = DirectedBrainTimeFilter(brain_tf_cfg)
        self.net_evolution = DynamicNetworkEvolutionModel(
            n_channels=c.n_channels, max_patches=max_patches,
            gcn_hidden=c.gcn_hidden, gru_hidden=c.gru_hidden,
            gru_layers=c.gru_layers,
            use_checkpoint=c.use_checkpoint,
        )

        # ── Step 3: Gated fusion ──
        self.fusion = GatedFusion(
            temporal_dim=c.embed_dim, network_dim=c.gru_hidden * 2,
            dropout=c.fusion_dropout,
        )

        # ── Step 4: SOZ head ──
        self.soz_head = SOZHead(
            embed_dim=c.embed_dim, n_channels=c.n_channels,
            n_patches=max_patches, output_mode=c.output_mode,
        )
        self.region_head = GlobalPoolHead(
            embed_dim=c.embed_dim,
            out_dim=c.n_regions,
        )
        self.hemisphere_head = GlobalPoolHead(
            embed_dim=c.embed_dim,
            out_dim=c.n_hemisphere_classes,
        )
        self.stage_head = PatchStageHead(
            embed_dim=c.embed_dim,
            n_channels=c.n_channels,
            n_classes=c.n_stage_classes,
        )

        # ── Loss functions ──
        self.focal_loss = FocalLoss(gamma=c.focal_gamma, alpha=c.focal_alpha)
        self.region_bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.hemisphere_ce = nn.CrossEntropyLoss(
            reduction='mean',
            ignore_index=-100,
        )
        self.transition_bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.pattern_ce = nn.CrossEntropyLoss(reduction='mean')
        self.stage_ce = nn.CrossEntropyLoss(
            reduction='mean',
            ignore_index=c.stage_ignore_index,
        )
        self._last: Optional[Dict] = None

    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set pos_weight for focal loss from dataset label statistics."""
        self.focal_loss = FocalLoss(
            gamma=self.cfg.focal_gamma, alpha=self.cfg.focal_alpha,
            pos_weight=pos_weight,
        )

    def set_stage_class_weight(self, class_weight: torch.Tensor):
        """Set class weights for stage-pretraining CrossEntropy loss."""
        self.stage_ce = nn.CrossEntropyLoss(
            weight=class_weight,
            reduction='mean',
            ignore_index=self.cfg.stage_ignore_index,
        )

    def configure_stage_pretraining(self, train_backbone: bool = True):
        """
        Enable only the modules used in stage pretraining.

        This keeps branch-B parameters frozen so the optimizer only updates
        LaBraM, TimeFilter, and the patch-level stage head.
        """
        for p in self.parameters():
            p.requires_grad = False

        if train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = True
        for p in self.timefilter.parameters():
            p.requires_grad = True
        for p in self.stage_head.parameters():
            p.requires_grad = True

    # -----------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        seizure_onset_sec: Optional[torch.Tensor] = None,
        window_start_sec: Optional[torch.Tensor] = None,
        valid_patch_counts: Optional[torch.Tensor] = None,
        brain_networks: Optional[torch.Tensor] = None,
        rel_time: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args
        ----
        x                 : [B, 22, T]  raw EEG (200 Hz)
        seizure_onset_sec : [B]         absolute onset time (s)
        window_start_sec  : [B]         absolute window start (s)
        valid_patch_counts: [B] (opt)   override patching counts
        brain_networks    : [B, P, 22, 22, 4] (opt) precomputed brain networks
        rel_time          : [B, P] (opt) precomputed rel time

        Returns
        -------
        dict with SOZ, region, hemisphere, and sequence auxiliary outputs
        """
        c = self.cfg
        B = x.size(0)

        if c.task_mode == 'stage_pretrain':
            if x.dim() == 4:
                patches = x.permute(0, 2, 1, 3)
            elif x.dim() == 3:
                expected = c.patch_len * (c.n_pre_patches + c.n_post_patches)
                if x.size(-1) != expected:
                    raise ValueError(
                        f"Stage pretrain expects {expected} samples per window, got {x.size(-1)}"
                    )
                patches = x.view(B, c.n_channels, -1, c.patch_len).permute(0, 2, 1, 3)
            else:
                raise ValueError(f"Unsupported stage-pretrain input shape: {tuple(x.shape)}")

            P = patches.size(1)
            vp_counts = torch.full(
                (B,),
                fill_value=P,
                dtype=torch.long,
                device=patches.device,
            )
            if rel_time is None:
                rel_time = torch.arange(
                    P, device=patches.device, dtype=torch.float32,
                ) * (c.patch_len / c.fs)
                rel_time = rel_time.unsqueeze(0).expand(B, -1)

            patches_a = patches.permute(0, 2, 1, 3)
            if c.use_checkpoint and self.training:
                h = checkpoint(self.backbone, patches_a, use_reentrant=False)
            else:
                h = self.backbone(patches_a)
            h, moe_loss_a = self.timefilter(h, is_training=self.training)

            stage_logits = self.stage_head(h)
            outputs = {
                'stage_logits': stage_logits,
                'stage_probs': torch.softmax(stage_logits, dim=-1),
                'valid_patch_counts': vp_counts,
                'seizure_relative_time': rel_time,
                'moe_loss': moe_loss_a,
            }
            self._last = {
                k: (v.detach() if isinstance(v, torch.Tensor) else v)
                for k, v in outputs.items()
            }
            return outputs

        # ── Step 1: Seizure-aligned patching ──
        if seizure_onset_sec is None or window_start_sec is None:
            raise ValueError("SOZ mode requires seizure_onset_sec and window_start_sec")

        patches, vp_counts_patched, rel_time_patched = self.patching(
            x, seizure_onset_sec, window_start_sec,
        )  # patches [B, P, 22, patch_len]
        
        vp_counts = valid_patch_counts if valid_patch_counts is not None else vp_counts_patched
        rel_time = rel_time if rel_time is not None else rel_time_patched
        P = patches.size(1)

        # ── Branch A: Temporal (LaBraM Backbone + TimeFilter) ──
        # LaBraMBackbone expects [B, 22, P, patch_len]
        patches_a = patches.permute(0, 2, 1, 3)              # [B, 22, P, patch_len]
        if c.use_checkpoint and self.training:
            h = checkpoint(self.backbone, patches_a, use_reentrant=False)
        else:
            h = self.backbone(patches_a)                      # [B, N, D]  N=22*P
        h, moe_loss_a = self.timefilter(h, is_training=self.training)  # [B, N, D]
        temporal_feat = h                                     # [B, N, D]

        # ── Branch B: Brain network + DirectedBrainTimeFilter ──
        if brain_networks is None:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
                net_result = self.net_extractor(patches.float())  # dict
            brain_nets = net_result['all']                        # [B, P, 22, 22, 4]
        else:
            brain_nets = brain_networks

        # 有向图过滤
        brain_nets_filtered, moe_loss_b = self.brain_timefilter(
            brain_nets, is_training=self.training,
        )  # [B, P, 22, 22, 4], scalar

        evo_out = self.net_evolution(brain_nets_filtered, vp_counts, rel_time)
        network_feat = evo_out['network_features']            # [B, 256]
        transition_probs = evo_out['transition_probs']        # [B, P]
        transition_logits = evo_out['transition_logits']      # [B, P]
        pattern_logits = evo_out['pattern_logits']            # [B, 3]
        branch_weights = evo_out['branch_weights']            # [B, P, 4]

        # ── Step 3: Gated fusion ──
        fused, gate_w = self.fusion(temporal_feat, network_feat)
        # fused: [B, N, D],  gate_w: [B, N]

        # ── Step 4: SOZ localization ──
        soz_logits, bipolar_logits = self.soz_head(fused)     # [B, 19], [B, 22]
        soz_probs = torch.sigmoid(soz_logits)
        region_logits = self.region_head(fused)               # [B, 6]
        hemisphere_logits = self.hemisphere_head(fused)       # [B, 3]

        # MoE辅助损失合并
        moe_loss = moe_loss_a + moe_loss_b

        outputs = {
            'soz_probs': soz_probs,
            'soz_logits': soz_logits,
            'bipolar_logits': bipolar_logits,
            'region_logits': region_logits,
            'region_probs': torch.sigmoid(region_logits),
            'hemisphere_logits': hemisphere_logits,
            'hemisphere_probs': torch.softmax(hemisphere_logits, dim=-1),
            'transition_probs': transition_probs,
            'transition_logits': transition_logits,
            'pattern_logits': pattern_logits,
            'gate_weights': gate_w,
            'brain_networks': brain_nets.detach(),
            'brain_networks_filtered': brain_nets_filtered.detach(),
            'branch_weights': branch_weights,
            'valid_patch_counts': vp_counts,
            'seizure_relative_time': rel_time,
            'moe_loss': moe_loss,
        }
        self._last = {k: (v.detach() if isinstance(v, torch.Tensor) else v)
                      for k, v in outputs.items()}
        return outputs

    # -----------------------------------------------------------------
    # Multi-task loss
    # -----------------------------------------------------------------

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        soz_targets: torch.Tensor,
        region_targets: Optional[torch.Tensor] = None,
        hemisphere_targets: Optional[torch.Tensor] = None,
        transition_targets: Optional[torch.Tensor] = None,
        pattern_targets: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args
        ----
        soz_targets        : [B, 19] or [B, 22]
        region_targets     : [B, 6]  (optional, multi-label)
        hemisphere_targets : [B]     (optional, 0=L,1=R,2=B,-100=ignore)
        transition_targets : [B, P]  (optional)
        pattern_targets    : [B]     (optional)

        Returns: total_loss, loss_dict
        """
        c = self.cfg
        losses = {}

        # primary: SOZ focal loss
        losses['soz'] = self.focal_loss(
            outputs['soz_logits'], soz_targets, sample_weight=sample_weight
        )
        total = losses['soz']

        if region_targets is not None:
            region_loss = F.binary_cross_entropy_with_logits(
                outputs['region_logits'],
                region_targets,
                reduction='none',
            )
            if sample_weight is not None:
                region_loss = region_loss * sample_weight.unsqueeze(1)
            losses['region'] = region_loss.mean()
            total = total + c.w_region * losses['region']

        if hemisphere_targets is not None:
            valid_hemisphere = hemisphere_targets != -100
            if valid_hemisphere.any():
                hemisphere_loss = F.cross_entropy(
                    outputs['hemisphere_logits'],
                    hemisphere_targets,
                    reduction='none',
                    ignore_index=-100,
                )
                hemisphere_loss = hemisphere_loss[valid_hemisphere]
                if sample_weight is not None:
                    hemisphere_loss = hemisphere_loss * sample_weight[valid_hemisphere]
                losses['hemisphere'] = hemisphere_loss.mean()
                total = total + c.w_hemisphere * losses['hemisphere']
            else:
                losses['hemisphere'] = outputs['hemisphere_logits'].new_zeros(())

        # auxiliary 1: transition detection
        if transition_targets is not None:
            tp = outputs['transition_logits']
            transition_loss = F.binary_cross_entropy_with_logits(
                tp, transition_targets, reduction='none',
            )
            if sample_weight is not None:
                transition_loss = transition_loss * sample_weight.unsqueeze(1)
            losses['transition'] = transition_loss.mean()
            total = total + c.w_transition * losses['transition']

        # auxiliary 2: pattern classification
        if pattern_targets is not None:
            pattern_loss = F.cross_entropy(
                outputs['pattern_logits'], pattern_targets, reduction='none',
            )
            if sample_weight is not None:
                pattern_loss = pattern_loss * sample_weight
            losses['pattern'] = pattern_loss.mean()
            total = total + c.w_pattern * losses['pattern']

        # MoE auxiliary loss (both branches)
        if 'moe_loss' in outputs:
            losses['moe'] = outputs['moe_loss']
            total = total + c.w_moe * outputs['moe_loss']

        losses['total'] = total
        return total, losses

    def compute_stage_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        stage_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute patch-level seizure/non-seizure loss."""
        losses: Dict[str, torch.Tensor] = {}
        flat_targets = stage_targets.reshape(-1)
        valid_mask = flat_targets != self.cfg.stage_ignore_index
        if not valid_mask.any():
            zero = outputs['stage_logits'].new_zeros(())
            losses['stage'] = zero
            losses['total'] = zero
            return zero, losses

        flat_logits = outputs['stage_logits'].reshape(-1, outputs['stage_logits'].size(-1))
        losses['stage'] = self.stage_ce(flat_logits, flat_targets)
        total = losses['stage']

        if 'moe_loss' in outputs:
            losses['moe'] = outputs['moe_loss']
            total = total + self.cfg.w_moe * outputs['moe_loss']

        losses['total'] = total
        return total, losses

    # -----------------------------------------------------------------
    # Training phases
    # -----------------------------------------------------------------

    def freeze_backbone(self):
        """Phase 1: freeze LaBraM backbone, train only heads."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_timefilter(self):
        """Phase 2: unfreeze TimeFilter + network modules."""
        for p in self.timefilter.parameters():
            p.requires_grad = True
        for p in self.brain_timefilter.parameters():
            p.requires_grad = True
        for p in self.net_evolution.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Phase 3: unfreeze everything."""
        for p in self.parameters():
            p.requires_grad = True

    def get_param_groups(self, lr: float = 1e-4) -> List[Dict]:
        """Differential LR: backbone 0.1x, timefilter 0.5x, heads 1x."""
        backbone_ids = set(id(p) for p in self.backbone.parameters())
        tf_ids = set(id(p) for p in self.timefilter.parameters())
        tf_ids |= set(id(p) for p in self.brain_timefilter.parameters())

        backbone_params, tf_params, head_params = [], [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in backbone_ids:
                backbone_params.append(p)
            elif pid in tf_ids:
                tf_params.append(p)
            else:
                head_params.append(p)

        return [
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': tf_params,       'lr': lr * 0.5},
            {'params': head_params,     'lr': lr},
        ]

    # -----------------------------------------------------------------
    # Checkpoint save / load
    # -----------------------------------------------------------------

    def save_checkpoint(self, path: str, extra: Dict = None):
        ckpt = {'model_state': self.state_dict(), 'config': self.cfg}
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location)
        cfg = ckpt['config']
        if isinstance(cfg, dict):
            cfg = IntegrationConfig(**cfg)
        model = cls(cfg)
        model.load_state_dict(ckpt['model_state'])
        return model, ckpt

    def load_a_branch_weights(self, path: str, map_location='cpu') -> Dict[str, List[str]]:
        """Load only backbone + timefilter weights from a checkpoint."""
        ckpt = torch.load(path, map_location=map_location)
        state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
        branch_state = {}
        for key, value in state.items():
            clean_key = key[7:] if key.startswith('module.') else key
            if clean_key.startswith('backbone.') or clean_key.startswith('timefilter.'):
                branch_state[clean_key] = value
        own_state = self.state_dict()
        loaded_keys: List[str] = []
        unexpected_keys: List[str] = []
        for key, value in branch_state.items():
            if key not in own_state or own_state[key].shape != value.shape:
                unexpected_keys.append(key)
                continue
            own_state[key].copy_(value)
            loaded_keys.append(key)

        missing_keys = [
            key for key in own_state.keys()
            if (key.startswith('backbone.') or key.startswith('timefilter.'))
            and key not in branch_state
        ]
        return {
            'loaded_keys': sorted(loaded_keys),
            'missing_keys': sorted(missing_keys),
            'unexpected_keys': sorted(unexpected_keys),
        }

    # -----------------------------------------------------------------
    # Interpretability
    # -----------------------------------------------------------------

    def visualize_gate_weights(self, batch_idx: int = 0, save_path=None):
        """Visualize gate weights showing temporal vs network contribution."""
        import matplotlib.pyplot as plt
        d = self._last
        gw = d['gate_weights'][batch_idx].cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(gw, 'o-', markersize=3)
        ax.set_xlabel('Node index (channel x patch)')
        ax.set_ylabel('Gate weight (1=temporal, 0=network)')
        ax.set_title(f'Gate weights (sample {batch_idx})')
        ax.axhline(0.5, color='gray', ls='--', alpha=0.5)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        plt.show()
        return fig

    def highlight_soz_channels(self, batch_idx: int = 0, threshold: float = 0.5):
        """Return predicted SOZ channels above threshold."""
        from labram_timefilter_soz import STANDARD_19
        probs = self._last['soz_probs'][batch_idx].cpu()
        channels = []
        for i, p in enumerate(probs):
            if p > threshold:
                name = STANDARD_19[i] if i < len(STANDARD_19) else f'ch{i}'
                channels.append((name, p.item()))
        channels.sort(key=lambda x: -x[1])
        return channels

    def summary(self) -> str:
        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"TimeFilter_LaBraM_BrainNetwork_Integration\n"
            f"  Task mode:        {self.cfg.task_mode}\n"
            f"  Total params:     {n_total:,}\n"
            f"  Trainable params: {n_train:,}\n"
            f"  Output mode:      {self.cfg.output_mode}\n"
            f"  Max patches:      {self.patching.max_patches}\n"
        )


# =====================================================================
# Self-test
# =====================================================================

def _test():
    torch.manual_seed(0)

    cfg = IntegrationConfig(
        labram_checkpoint='',
        n_transformer_layers=2,
        n_frozen_layers=0,
        embed_dim=200,             # 对齐LaBraM-base
        gru_hidden=64,
        gcn_hidden=32,
        n_pre_patches=4,
        n_post_patches=6,
        n_timefilter_blocks=1,
        brain_tf_n_blocks=1,
        brain_tf_hidden=32,
        patch_len=200,
    )
    max_patches = cfg.n_pre_patches + cfg.n_post_patches  # 10

    B, C, T = 2, 22, 2400  # 12 seconds @ 200 Hz
    x = torch.randn(B, C, T)
    onset = torch.tensor([105.0, 107.0])
    start = torch.tensor([100.0, 100.0])

    model = TimeFilter_LaBraM_BrainNetwork_Integration(cfg)
    print(model.summary())

    # forward
    out = model(x, onset, start)

    n_out = 19 if cfg.output_mode == 'monopolar' else 22
    assert out['soz_probs'].shape == (B, n_out), \
        f"soz_probs: expected [B,{n_out}], got {list(out['soz_probs'].shape)}"
    assert out['region_logits'].shape == (B, cfg.n_regions)
    assert out['hemisphere_logits'].shape == (B, cfg.n_hemisphere_classes)
    assert out['transition_probs'].shape == (B, max_patches)
    assert out['pattern_logits'].shape == (B, 3)
    assert out['brain_networks'].shape == (B, max_patches, C, C, 4)
    assert out['gate_weights'].shape[0] == B

    print(f"soz_probs         : {list(out['soz_probs'].shape)}")
    print(f"region_logits     : {list(out['region_logits'].shape)}")
    print(f"hemisphere_logits : {list(out['hemisphere_logits'].shape)}")
    print(f"transition_probs  : {list(out['transition_probs'].shape)}")
    print(f"pattern_logits    : {list(out['pattern_logits'].shape)}")
    print(f"gate_weights      : {list(out['gate_weights'].shape)}")
    print(f"brain_networks    : {list(out['brain_networks'].shape)}")

    # loss
    soz_target = torch.zeros(B, n_out)
    soz_target[:, 3] = 1.0   # simulate one SOZ channel
    region_target = torch.zeros(B, cfg.n_regions)
    region_target[:, 0] = 1.0
    region_target[:, 3] = 1.0
    hemisphere_target = torch.tensor([0, 2], dtype=torch.long)
    vm = model.patching._valid_mask
    if vm is None:
        vm = torch.ones(B, max_patches, dtype=torch.bool)
    aux_targets = DynamicNetworkEvolutionModel.compute_auxiliary_targets(
        out['seizure_relative_time'], vm,
    )
    total, losses = model.compute_loss(
        out, soz_target,
        region_targets=region_target,
        hemisphere_targets=hemisphere_target,
        transition_targets=aux_targets['transition_targets'],
        pattern_targets=aux_targets['pattern_targets'],
    )
    print(f"\nLosses: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))
    total.backward()

    n_grad = sum(1 for p in model.parameters()
                 if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0)
    n_req = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Gradient: {n_grad}/{n_req} params have grad")

    # freeze / unfreeze
    model.freeze_backbone()
    n_after = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"After freeze_backbone: {n_after}/{n_req} trainable")
    model.unfreeze_all()

    # param groups
    groups = model.get_param_groups(lr=1e-4)
    for i, g in enumerate(groups):
        print(f"  group {i}: {len(g['params'])} params, lr={g['lr']}")

    print("\n[PASS] All tests passed!")


if __name__ == '__main__':
    _test()
