"""
HOMAR-V3 (多源扩展版): 支持 APTOS / Messidor-2 / IDRiD / DDR 四源联合训练, 按源分组评估。

核心设计:
  1. 通过 importlib 动态加载 V2 (文件名含 `-`, 不能直接 import), 继承其所有类与函数
  2. HOMARTrainerV3: 按 test_sources 列表的每个来源分别计算 Acc/QWK/AUROC/AUPR
     history / best_ckpt / final_summary 动态支持 N 个数据源
  3. get_loaders_multi(): 不再合并目录, 每个源独立 ImageFolder, 用 ConcatDataset 拼接
     - APTOS / M2 (无官方划分): 7:3 stratified random split
     - IDRiD / DDR (有官方划分): 直接用官方 train/test 目录
     - test_sources 数组按 dataset 区间生成 (不再依赖文件名前缀)
  4. 支持任意组合: 1 源 (单域基线, 等价 V2.1) / 2 源 / 3 源 / 4 源
  5. 默认 patience=999 (不触发早停), epochs=100

CLI 示例 (四源联合训练):
  python HOMAR-retfound-v3.py \
    --aptos_root /root/autodl-tmp/longfei/colored_images \
    --m2_root /root/autodl-tmp/longfei/colored_images_messidor \
    --idrid_train /root/autodl-tmp/longfei/iDRiD/organized/train \
    --idrid_test  /root/autodl-tmp/longfei/iDRiD/organized/test \
    --ddr_train /root/autodl-tmp/longfei/colored_images_ddr/train \
    --ddr_test  /root/autodl-tmp/longfei/colored_images_ddr/test \
    --save_dir runs/homar_v3_joint4 --epochs 100 --patience 999
"""

import os
import sys
import argparse
import json
import copy
import importlib.util
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset, Subset
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------
# 动态加载 HOMAR-retfound-v2
# ------------------------------------------------------------
_V2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'HOMAR-retfound-v2.py')
_spec = importlib.util.spec_from_file_location("homar_v2", _V2_PATH)
homar_v2 = importlib.util.module_from_spec(_spec)
sys.modules["homar_v2"] = homar_v2
_spec.loader.exec_module(homar_v2)

HOMAR = homar_v2.HOMAR
HOMARTrainer = homar_v2.HOMARTrainer
ModelEMA = homar_v2.ModelEMA
tta_forward = homar_v2.tta_forward
plot_training_curves = homar_v2.plot_training_curves
plot_confusion_matrix = homar_v2.plot_confusion_matrix
compute_model_efficiency = homar_v2.compute_model_efficiency
GradCAM = homar_v2.GradCAM
DEFAULT_RETFOUND_WEIGHTS = homar_v2.DEFAULT_RETFOUND_WEIGHTS
BenGrahamTransform = homar_v2.BenGrahamTransform
CLAHETransform = homar_v2.CLAHETransform


# ------------------------------------------------------------
# V3 Trainer: 支持按源 (aptos/m2/idrid/ddr) 拆分测试集指标
# ------------------------------------------------------------
class HOMARTrainerV3(HOMARTrainer):
    def __init__(self, *args, test_sources=None, source_keys: Optional[List[str]] = None,
                 full_ckpt: bool = False, no_vis: bool = False, **kwargs):
        """
        test_sources: np.array of shape (N_test,), 元素取自 source_keys
        source_keys:  实际参与训练的源名列表, 例如 ['aptos','m2','idrid','ddr']
        full_ckpt:    默认 False (light_ckpt), 只存 EMA 权重和指标; True 时额外存 model + history
        no_vis:       True 时跳过所有可视化输出 (vis/curves/cm/gradcam), 仅保留 JSON 和 best_model.pth
        """
        super().__init__(*args, **kwargs)
        self.test_sources = test_sources
        self.source_keys = list(source_keys) if source_keys else []
        self.full_ckpt = full_ckpt
        self.no_vis = no_vis

        # 动态构建 history: all + 每个源 4 指标
        self.history = {
            'train_loss': [],
            'val_acc': [], 'val_qwk': [], 'val_auroc': [], 'val_aupr': [],
        }
        for sk in self.source_keys:
            for m in ('acc', 'qwk', 'auroc', 'aupr'):
                self.history[f'val_{m}_{sk}'] = []

        if self.test_sources is not None:
            print("[TrainerV3] test_sources distribution:")
            for sk in self.source_keys:
                n = int((self.test_sources == sk).sum())
                print(f"  {sk}: {n}")
            print(f"  total: {len(self.test_sources)}")

    @staticmethod
    def _compute_subset_metrics(targets, preds, probs, nc):
        """对一个 (targets, preds, probs) 子集算 Acc/QWK/AUROC/AUPR。"""
        if len(targets) == 0:
            return {'accuracy': 0.0, 'qwk': 0.0, 'auroc': 0.0, 'aupr': 0.0,
                    'confusion_matrix': np.zeros((nc, nc), dtype=int), 'n': 0}
        acc = accuracy_score(targets, preds)
        try:
            qwk = cohen_kappa_score(targets, preds, weights='quadratic')
        except Exception:
            qwk = 0.0
        try:
            from sklearn.metrics import roc_auc_score
            present = np.unique(targets)
            if len(present) < 2:
                auroc = 0.0
            else:
                sub_probs = probs[:, present] / probs[:, present].sum(axis=1, keepdims=True).clip(1e-12)
                auroc = roc_auc_score(targets, sub_probs, multi_class='ovr', average='macro',
                                      labels=present)
        except Exception:
            auroc = 0.0
        try:
            from sklearn.metrics import average_precision_score
            onehot = np.eye(nc)[targets]
            aupr = average_precision_score(onehot, probs, average='macro')
        except Exception:
            aupr = 0.0
        cm = confusion_matrix(targets, preds, labels=list(range(nc)))
        return {'accuracy': acc, 'qwk': qwk, 'auroc': auroc, 'aupr': aupr,
                'confusion_matrix': cm, 'n': int(len(targets))}

    @torch.no_grad()
    def validate(self, test_loader, use_ema=True, use_tta=None):
        """V3 验证: 用全量搜索最优阈值, 再对 all 及每个源分别计算指标。"""
        if use_tta is None:
            use_tta = self.use_tta

        if use_ema:
            backup = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict(), strict=False)

        self.model.eval()
        all_targets, all_sev, all_logits = [], [], []
        nc = self.model.ordinal_head.num_classes

        desc = 'Validating' + (' [EMA+TTA]' if use_ema and use_tta else ' [EMA]' if use_ema else '')
        for images, labels in tqdm(test_loader, desc=desc):
            images = images.to(self.device)
            if use_tta:
                out = tta_forward(self.model, images)
            else:
                out = self.model(images, return_all=True)
            all_targets.extend(labels.numpy())
            all_sev.extend(out['severity'].cpu().numpy())
            all_logits.append(out['logits'].cpu().numpy())

        if use_ema:
            self.model.load_state_dict(backup)

        all_targets = np.array(all_targets)
        all_sev = np.array(all_sev)
        all_logits = np.concatenate(all_logits, axis=0)
        all_probs = F.softmax(torch.from_numpy(all_logits), dim=1).numpy()

        best_thresholds = self._search_optimal_thresholds(all_sev, all_targets, nc)
        all_preds = np.digitize(all_sev, best_thresholds).clip(0, nc - 1)

        full_metrics = self._compute_subset_metrics(all_targets, all_preds, all_probs, nc)
        full_metrics['thresholds'] = best_thresholds.tolist()
        full_metrics['n'] = int(len(all_targets))

        # 按源子集 (动态)
        result = {
            'accuracy': full_metrics['accuracy'],
            'qwk': full_metrics['qwk'],
            'auroc': full_metrics['auroc'],
            'aupr': full_metrics['aupr'],
            'confusion_matrix': full_metrics['confusion_matrix'],
            'thresholds': best_thresholds.tolist(),
            'round_qwk': 0.0,
            'all': full_metrics,
        }
        if self.test_sources is not None and len(self.test_sources) == len(all_targets):
            for sk in self.source_keys:
                mask = self.test_sources == sk
                if mask.any():
                    result[sk] = self._compute_subset_metrics(
                        all_targets[mask], all_preds[mask], all_probs[mask], nc)
                else:
                    result[sk] = {'accuracy': 0.0, 'qwk': 0.0, 'auroc': 0.0, 'aupr': 0.0,
                                  'confusion_matrix': np.zeros((nc, nc), dtype=int), 'n': 0}
        else:
            for sk in self.source_keys:
                result[sk] = {'accuracy': 0.0, 'qwk': 0.0, 'auroc': 0.0, 'aupr': 0.0,
                              'confusion_matrix': np.zeros((nc, nc), dtype=int), 'n': 0}
        return result

    def train(self, train_loader, test_loader, epochs=100):
        for epoch in range(epochs):
            if self.stage == 1 and epoch == self.freeze_epochs:
                self.unfreeze_backbone()
            train_loss = self.train_epoch(train_loader)
            metrics = self.validate(test_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(metrics['accuracy'])
            self.history['val_qwk'].append(metrics['qwk'])
            self.history['val_auroc'].append(metrics['auroc'])
            self.history['val_aupr'].append(metrics['aupr'])
            for sk in self.source_keys:
                sm = metrics[sk]
                self.history[f'val_acc_{sk}'].append(sm['accuracy'])
                self.history[f'val_qwk_{sk}'].append(sm['qwk'])
                self.history[f'val_auroc_{sk}'].append(sm['auroc'])
                self.history[f'val_aupr_{sk}'].append(sm['aupr'])

            improved = metrics['qwk'] > self.best_qwk
            if improved:
                self.best_qwk = metrics['qwk']
                self.best_epoch = epoch
                self.patience_counter = 0
                ckpt = {
                    'epoch': epoch,
                    'ema_state_dict': self.ema.state_dict(),  # light_ckpt: 仅保留 EMA (推理就够)
                    'metrics_all': {k: v for k, v in metrics['all'].items() if k != 'confusion_matrix'},
                    'thresholds': metrics['thresholds'],
                    'source_keys': self.source_keys,
                }
                if getattr(self, 'full_ckpt', False):
                    # 反向开关: 同时存 model_state_dict + history (体积翻倍)
                    ckpt['model_state_dict'] = self.model.state_dict()
                    ckpt['history'] = self.history
                for sk in self.source_keys:
                    ckpt[f'metrics_{sk}'] = {k: v for k, v in metrics[sk].items() if k != 'confusion_matrix'}
                torch.save(ckpt, os.path.join(self.save_dir, 'best_model.pth'))
                if epoch > 0 and not self.no_vis:
                    try:
                        self.model.visualize(next(iter(test_loader))[0].to(self.device),
                                             os.path.join(self.save_dir, f'vis_epoch{epoch}.png'))
                    except Exception as e:
                        print(f"[vis] skip: {e}")
            else:
                self.patience_counter += 1

            if (epoch + 1) % 10 == 0 and not self.no_vis:
                plot_training_curves(self.history, os.path.join(self.save_dir, f'curves_epoch{epoch+1}.png'))
                if metrics.get('confusion_matrix') is not None:
                    plot_confusion_matrix(
                        metrics['confusion_matrix'], ['0', '1', '2', '3', '4'],
                        os.path.join(self.save_dir, f'cm_epoch{epoch+1}.png'))
                self._generate_gradcam(test_loader, epoch)

            self.scheduler.step()

            # 日志
            src_log = " | ".join(
                f"{sk.upper()}(n={metrics[sk]['n']}) Acc={metrics[sk]['accuracy']:.4f} "
                f"QWK={metrics[sk]['qwk']:.4f} AUROC={metrics[sk]['auroc']:.4f} AUPR={metrics[sk]['aupr']:.4f}"
                for sk in self.source_keys
            )
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f} | "
                  f"ALL Acc={metrics['accuracy']:.4f} QWK={metrics['qwk']:.4f} "
                  f"AUROC={metrics['auroc']:.4f} AUPR={metrics['aupr']:.4f} | {src_log} | "
                  f"Best={self.best_qwk:.4f}@ep{self.best_epoch+1} PC={self.patience_counter}/{self.patience}")

            with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f, indent=2)

            if self.stage == 2 and self.patience_counter >= self.patience:
                print(f"\n[EarlyStopping] QWK 连续 {self.patience} epoch 未提升, 在 epoch {epoch+1} 提前停止. "
                      f"Best QWK={self.best_qwk:.4f} @ epoch {self.best_epoch+1}")
                break

        print("\n训练完成, 生成最终报告...")
        if not self.no_vis:
            plot_training_curves(self.history, os.path.join(self.save_dir, 'final_curves.png'))
        final_metrics = self.validate(test_loader)
        print(f"\n最终: ALL Acc={final_metrics['accuracy']:.4f} QWK={final_metrics['qwk']:.4f} "
              f"AUROC={final_metrics['auroc']:.4f} AUPR={final_metrics['aupr']:.4f}")
        for sk in self.source_keys:
            sm = final_metrics[sk]
            print(f"最终 {sk.upper()}: Acc={sm['accuracy']:.4f} QWK={sm['qwk']:.4f} "
                  f"AUROC={sm['auroc']:.4f} AUPR={sm['aupr']:.4f} (n={sm['n']})")
        if not self.no_vis:
            if final_metrics.get('confusion_matrix') is not None:
                plot_confusion_matrix(
                    final_metrics['confusion_matrix'], ['0', '1', '2', '3', '4'],
                    os.path.join(self.save_dir, 'final_cm.png'))
            self._generate_gradcam(test_loader, epochs)
        summary = {
            'best_epoch': self.best_epoch + 1,
            'best_qwk_all': self.best_qwk,
            'source_keys': self.source_keys,
            'final_all': {k: v for k, v in final_metrics['all'].items() if k != 'confusion_matrix'},
            'final_thresholds': final_metrics['thresholds'],
        }
        for sk in self.source_keys:
            summary[f'final_{sk}'] = {k: v for k, v in final_metrics[sk].items() if k != 'confusion_matrix'}
        with open(os.path.join(self.save_dir, 'final_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        compute_model_efficiency(self.model, device=self.device)


# ------------------------------------------------------------
# 多源数据加载: ConcatDataset + 区间 test_sources
# ------------------------------------------------------------
def _build_transforms(img_size, strong_aug):
    if strong_aug:
        train_t = transforms.Compose([
            BenGrahamTransform(), CLAHETransform(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_t = transforms.Compose([
            BenGrahamTransform(), CLAHETransform(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        train_t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return train_t, test_t


def _split_random(root, train_t, test_t, test_size=0.3, seed=42):
    """无官方划分: 同一 root 按 7:3 stratified 划分。返回 (train_subset, test_subset, train_targets, test_classes)。"""
    raw = torchvision.datasets.ImageFolder(root=root)
    indices = np.arange(len(raw))
    targets = np.array(raw.targets)
    tr_idx, te_idx = train_test_split(indices, test_size=test_size,
                                      stratify=targets, random_state=seed)
    train_ds = torchvision.datasets.ImageFolder(root=root, transform=train_t)
    test_ds = torchvision.datasets.ImageFolder(root=root, transform=test_t)
    train_subset = Subset(train_ds, tr_idx.tolist())
    test_subset = Subset(test_ds, te_idx.tolist())
    train_tgt = np.array([train_ds.targets[i] for i in tr_idx])
    return train_subset, test_subset, train_tgt, raw.classes


def _split_official(train_root, test_root, train_t, test_t):
    """官方划分: train_root/test_root 两个目录。"""
    train_ds = torchvision.datasets.ImageFolder(root=train_root, transform=train_t)
    test_ds = torchvision.datasets.ImageFolder(root=test_root, transform=test_t)
    train_tgt = np.array(train_ds.targets)
    # 类目录顺序要和其他源一致 (依赖 0_No_DR/1_Mild/2_Moderate/3_Severe/4_Proliferate_DR 命名)
    return train_ds, test_ds, train_tgt, train_ds.classes


def get_loaders_multi(sources: Dict[str, Dict], batch_size=16, img_size=224, strong_aug=False,
                      num_workers=4):
    """
    sources: dict, e.g.
      {
        'aptos': {'type': 'random', 'root': '/.../colored_images'},
        'm2':    {'type': 'random', 'root': '/.../colored_images_messidor'},
        'idrid': {'type': 'official', 'train_root': '.../iDRiD/organized/train',
                  'test_root': '.../iDRiD/organized/test'},
        'ddr':   {'type': 'official', 'train_root': '...', 'test_root': '...'},
      }
    返回 (train_loader, test_loader, test_sources, source_keys, num_classes, classes)
    """
    train_t, test_t = _build_transforms(img_size, strong_aug)

    source_keys = list(sources.keys())
    train_subsets, test_subsets = [], []
    train_targets_list, sources_list = [], []
    classes_ref = None

    for sk in source_keys:
        cfg = sources[sk]
        if cfg['type'] == 'random':
            tr, te, tgt, classes = _split_random(cfg['root'], train_t, test_t)
        elif cfg['type'] == 'official':
            tr, te, tgt, classes = _split_official(cfg['train_root'], cfg['test_root'], train_t, test_t)
        else:
            raise ValueError(f"Unknown type for source {sk}: {cfg['type']}")

        if classes_ref is None:
            classes_ref = classes
        elif list(classes) != list(classes_ref):
            raise RuntimeError(f"Classes mismatch: {sk} -> {classes} vs ref {classes_ref}")

        train_subsets.append(tr)
        test_subsets.append(te)
        train_targets_list.append(tgt)
        sources_list.append(np.full(len(te), sk))
        print(f"  [{sk}] train={len(tr)}, test={len(te)}")

    train_set = ConcatDataset(train_subsets)
    test_set = ConcatDataset(test_subsets)
    all_train_targets = np.concatenate(train_targets_list)
    test_sources = np.concatenate(sources_list)

    # WeightedRandomSampler 用全量 train targets 算权重
    weights = compute_class_weight('balanced',
                                   classes=np.unique(all_train_targets),
                                   y=all_train_targets)
    sample_weights = weights[all_train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    num_classes = len(classes_ref)
    print(f"\nTotal train={len(train_set)}, test={len(test_set)}, classes={classes_ref}")
    print(f"Class weights (on train): {weights}")

    return train_loader, test_loader, test_sources, source_keys, num_classes, classes_ref


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def _parse_sources(args) -> Dict[str, Dict]:
    sources = {}
    if args.aptos_root:
        sources['aptos'] = {'type': 'random', 'root': args.aptos_root}
    if args.m2_root:
        sources['m2'] = {'type': 'random', 'root': args.m2_root}
    if args.idrid_train and args.idrid_test:
        sources['idrid'] = {'type': 'official',
                            'train_root': args.idrid_train,
                            'test_root': args.idrid_test}
    if args.ddr_train and args.ddr_test:
        sources['ddr'] = {'type': 'official',
                          'train_root': args.ddr_train,
                          'test_root': args.ddr_test}
    if not sources:
        raise ValueError("至少要指定一个源 (--aptos_root / --m2_root / --idrid_* / --ddr_*)")
    return sources


def main():
    parser = argparse.ArgumentParser()
    # 四源: 随机划分型
    parser.add_argument('--aptos_root', type=str, default=None)
    parser.add_argument('--m2_root', type=str, default=None)
    # 四源: 官方划分型
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
    parser.add_argument('--patience', type=int, default=999,
                        help='默认 999 (等价不触发早停)')
    parser.add_argument('--ema_decay', type=float, default=0.999)
    parser.add_argument('--disable_tta', action='store_true')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--retfound_weights', type=str, default=DEFAULT_RETFOUND_WEIGHTS)
    parser.add_argument('--strong_aug', action='store_true')
    parser.add_argument('--no_routing', action='store_true')
    parser.add_argument('--no_hierarchy', action='store_true')
    parser.add_argument('--no_consensus', action='store_true')
    parser.add_argument('--full_ckpt', action='store_true',
                        help='默认只存 EMA 权重 (省硬盘); 加此 flag 同时存 model_state_dict + history')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"HOMAR-V3 (multi-source) on {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    sources = _parse_sources(args)
    print(f"Sources: {list(sources.keys())}")

    train_loader, test_loader, test_sources, source_keys, num_classes, classes = \
        get_loaders_multi(sources, batch_size=args.batch_size, img_size=args.img_size,
                          strong_aug=args.strong_aug, num_workers=args.num_workers)
    print(f"num_classes={num_classes}, classes={classes}")

    model = HOMAR(
        dim=args.dim, num_classes=num_classes, pretrained=True,
        use_routing=not args.no_routing,
        use_hierarchy=not args.no_hierarchy,
        use_consensus=not args.no_consensus,
        retfound_weights=args.retfound_weights,
    )

    print(f"\n消融配置: Routing={'ON' if not args.no_routing else 'OFF'}, "
          f"Hierarchy={'ON' if not args.no_hierarchy else 'OFF'}, "
          f"Consensus={'ON' if not args.no_consensus else 'OFF'}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total/1e6:.2f}M")

    trainer = HOMARTrainerV3(
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
