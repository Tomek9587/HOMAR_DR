"""
RETFound 纯基线 (P1 消融对照):
  模型 = RETFound ViT-L/16 + GAP + Linear-Proj + CORAL Ordinal Head + Classifier
  对比 HOMAR 去掉:
    - AnatomicalRouter
    - MambaCapsuleExpert × 3
    - HierarchicalMamba
    - CrossLevelConsensus
  保留 HOMAR 的训练策略 (两阶段微调 + EMA + TTA + CORAL + Soft-QWK + CE),
  这样 ablation 差异只来源于"架构模块", 训练 trick 全一致, 符合论文严谨要求.

CLI 用法 (四源联合训练):
  python retfound_baseline.py \
    --aptos_root /root/autodl-tmp/longfei/colored_images \
    --m2_root /root/autodl-tmp/longfei/colored_images_messidor \
    --idrid_train /root/autodl-tmp/longfei/iDRiD/organized/train \
    --idrid_test  /root/autodl-tmp/longfei/iDRiD/organized/test \
    --ddr_train /root/autodl-tmp/longfei/colored_images_ddr/train \
    --ddr_test  /root/autodl-tmp/longfei/colored_images_ddr/test \
    --save_dir classification/runs/baseline_retfound_joint4 \
    --epochs 100 --patience 999
"""
import os
import sys
import argparse
import importlib.util

import torch
import torch.nn as nn

# ------------------------------------------------------------
# 动态加载 HOMAR-retfound-v3 (文件名带 '-' 不能 import)
# ------------------------------------------------------------
_V3_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'HOMAR-retfound-v3.py')
_spec = importlib.util.spec_from_file_location("homar_v3", _V3_PATH)
homar_v3 = importlib.util.module_from_spec(_spec)
sys.modules["homar_v3"] = homar_v3
_spec.loader.exec_module(homar_v3)

HOMARTrainerV3 = homar_v3.HOMARTrainerV3
get_loaders_multi = homar_v3.get_loaders_multi
_parse_sources = homar_v3._parse_sources

# 通过 v3 拿到 v2 的基础模块
homar_v2 = sys.modules["homar_v2"]
RETFoundBackbone = homar_v2.RETFoundBackbone
CORALOrdinalHead = homar_v2.CORALOrdinalHead
DEFAULT_RETFOUND_WEIGHTS = homar_v2.DEFAULT_RETFOUND_WEIGHTS
compute_model_efficiency = homar_v2.compute_model_efficiency


# ============================================================
# 1. RETFound 纯基线模型
# ============================================================
class RETFoundBaseline(nn.Module):
    """
    最精简基线: RETFound ViT-L/16 → GAP → Linear-Proj → CORAL + Classifier
    forward 返回的 dict 接口与 HOMAR 对齐 (logits/severity/prob_dist/ord_logits),
    可以直接喂给 HOMARTrainerV3 和 tta_forward / evaluate_external.
    """
    def __init__(self, dim: int = 384, num_classes: int = 5,
                 pretrained: bool = True,
                 retfound_weights: str = None,
                 drop_path: float = 0.1):
        super().__init__()
        weights = retfound_weights if pretrained else None
        self.backbone = RETFoundBackbone(weights_path=weights, drop_path=drop_path)

        # 投影: 1024 -> dim (与 HOMAR 的 proj 对齐接口)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.backbone.embed_dim),
            nn.Linear(self.backbone.embed_dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.feat_norm = nn.LayerNorm(dim)

        self.ordinal_head = CORALOrdinalHead(dim, num_classes=num_classes)
        self.classifier = nn.Linear(dim, num_classes)

        # 消融对齐标记 (给 Trainer 一个稳定的 flag, 但不会影响 forward)
        self.use_routing = False
        self.use_hierarchy = False
        self.use_consensus = False

    def forward(self, img: torch.Tensor, return_all: bool = False, patch_tokens=None):
        if patch_tokens is None:
            patch_tokens = self.backbone(img)          # (B, 196, 1024)
        tokens = self.proj(patch_tokens)               # (B, 196, dim)
        feat = tokens.mean(dim=1)                      # GAP -> (B, dim)
        feat = self.feat_norm(feat)

        severity, prob_dist, ord_logits = self.ordinal_head(feat, None)
        cls_logits = self.classifier(feat)

        if return_all:
            return {
                'logits': cls_logits,
                'severity': severity,
                'prob_dist': prob_dist,
                'ord_logits': ord_logits,
                'consensus_feat': feat,
            }
        return cls_logits


# ============================================================
# 2. Baseline Trainer: 只重写两个 optimizer builder,
#    去掉 experts/router/hierarchy/consensus 参数组.
# ============================================================
class BaselineTrainerV3(HOMARTrainerV3):
    def _build_stage1_optimizer(self):
        model = self.model
        head_groups = [
            {'params': model.proj.parameters(),          'lr': 5e-4},
            {'params': model.feat_norm.parameters(),     'lr': 5e-4},
            {'params': model.ordinal_head.parameters(),  'lr': 5e-4},
            {'params': model.classifier.parameters(),    'lr': 5e-4},
        ]
        self.optimizer = torch.optim.AdamW(head_groups, weight_decay=0.01)

        warmup_ep = 2
        warmup = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda e: (e + 1) / warmup_ep
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.freeze_epochs - warmup_ep, 1),
            eta_min=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[warmup_ep]
        )

    def _build_stage2_optimizer(self):
        model = self.model
        for p in model.backbone.parameters():
            p.requires_grad_(True)

        param_groups = [
            {'params': model.backbone.parameters(),      'lr': 1e-5, 'weight_decay': 0.01},
            {'params': model.proj.parameters(),          'lr': 5e-5},
            {'params': model.feat_norm.parameters(),     'lr': 5e-5},
            {'params': model.ordinal_head.parameters(),  'lr': 5e-5},
            {'params': model.classifier.parameters(),    'lr': 5e-5},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        stage2_ep = max(self.total_epochs - self.freeze_epochs, 1)
        warmup_ep = 2
        warmup = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda e: (e + 1) / warmup_ep
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(stage2_ep - warmup_ep, 1),
            eta_min=1e-7
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, [warmup, cosine], milestones=[warmup_ep]
        )


# ============================================================
# 3. main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aptos_root', type=str, default=None)
    parser.add_argument('--m2_root', type=str, default=None)
    parser.add_argument('--idrid_train', type=str, default=None)
    parser.add_argument('--idrid_test', type=str, default=None)
    parser.add_argument('--ddr_train', type=str, default=None)
    parser.add_argument('--ddr_test', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--freeze_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=999)
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--disable_tta', action='store_true')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--retfound_weights', type=str, default=DEFAULT_RETFOUND_WEIGHTS)
    parser.add_argument('--strong_aug', action='store_true')
    parser.add_argument('--full_ckpt', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"RETFound-Baseline (multi-source) on {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sources = _parse_sources(args)
    print(f"Sources: {list(sources.keys())}")

    train_loader, test_loader, test_sources, source_keys, num_classes, classes = \
        get_loaders_multi(sources, batch_size=args.batch_size, img_size=args.img_size,
                          strong_aug=args.strong_aug, num_workers=args.num_workers)
    print(f"num_classes={num_classes}, classes={classes}")

    model = RETFoundBaseline(
        dim=args.dim, num_classes=num_classes, pretrained=True,
        retfound_weights=args.retfound_weights,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Baseline] Total params: {total/1e6:.2f}M, trainable(before-stage1-freeze): {trainable/1e6:.2f}M")
    print("[Baseline] 架构: RETFound ViT-L/16 + GAP + Proj + CORAL Head + Linear Cls")
    print("[Baseline] 消融基准: NO Routing / NO Experts / NO Hierarchy / NO Consensus")

    trainer = BaselineTrainerV3(
        model, device,
        save_dir=args.save_dir,
        freeze_epochs=args.freeze_epochs,
        total_epochs=args.epochs,
        ema_decay=args.ema_decay,
        patience=args.patience,
        use_tta=not args.disable_tta,
        test_sources=test_sources,
        source_keys=source_keys,
        full_ckpt=args.full_ckpt,
    )
    trainer.train(train_loader, test_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
