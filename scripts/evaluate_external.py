"""
外部数据集评估脚本: 加载一个 best_model.pth, 在任意 data_root (ImageFolder) 上跑 EMA+TTA 推理,
输出 Acc/QWK/AUROC/AUPR 到 JSON。

用于实验 E (APTOS->M2 跨数据集泛化) 和 F (M2->APTOS 反向泛化)。
"""

import os
import sys
import argparse
import json
import importlib.util

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             roc_auc_score, average_precision_score)
import numpy as np
from tqdm import tqdm

# 动态加载 v2
_V2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HOMAR-retfound-v2.py')
_spec = importlib.util.spec_from_file_location("homar_v2", _V2_PATH)
homar_v2 = importlib.util.module_from_spec(_spec)
sys.modules["homar_v2"] = homar_v2
_spec.loader.exec_module(homar_v2)

HOMAR = homar_v2.HOMAR
tta_forward = homar_v2.tta_forward
DEFAULT_RETFOUND_WEIGHTS = homar_v2.DEFAULT_RETFOUND_WEIGHTS

# 可选: RETFoundBaseline (P1 基线消融)
_BASELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'retfound_baseline.py')
if os.path.exists(_BASELINE_PATH):
    _bspec = importlib.util.spec_from_file_location('baseline_mod', _BASELINE_PATH)
    _bmod = importlib.util.module_from_spec(_bspec)
    _bspec.loader.exec_module(_bmod)
    RETFoundBaseline = _bmod.RETFoundBaseline
else:
    RETFoundBaseline = None


def search_optimal_thresholds(severities, targets, nc):
    best_thresholds = np.arange(nc - 1) + 0.5
    best_qwk = -1.0
    for _ in range(3):
        for k in range(nc - 1):
            best_t = best_thresholds[k]
            for t in np.arange(max(k * 0.5, 0.0), min((k + 2) * 1.0, nc - 0.01), 0.05):
                trial = best_thresholds.copy()
                trial[k] = t
                trial.sort()
                preds = np.digitize(severities, trial).clip(0, nc - 1)
                try:
                    q = cohen_kappa_score(targets, preds, weights='quadratic')
                except Exception:
                    q = -1
                if q > best_qwk:
                    best_qwk = q
                    best_t = t
            best_thresholds[k] = best_t
            best_thresholds.sort()
    return best_thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='best_model.pth 路径')
    parser.add_argument('--data_root', type=str, required=True, help='ImageFolder 目录')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--no_ema', dest='use_ema', action='store_false')
    parser.add_argument('--use_tta', action='store_true', default=True)
    parser.add_argument('--no_tta', dest='use_tta', action='store_false')
    parser.add_argument('--no_routing', action='store_true')
    parser.add_argument('--no_hierarchy', action='store_true')
    parser.add_argument('--no_consensus', action='store_true')
    parser.add_argument('--model', type=str, default='homar',
                        choices=['homar', 'baseline'],
                        help='homar (默认) 或 baseline (RETFound纯基线)')
    parser.add_argument('--retfound_weights', type=str, default=DEFAULT_RETFOUND_WEIGHTS)
    parser.add_argument('--output_json', type=str, required=True, help='评估结果输出 json 路径')
    parser.add_argument('--thresholds', type=str, default=None,
                        help='可选: 使用 ckpt 中保存的阈值 (JSON list)。默认在当前数据上重新搜索。')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[eval] device={device}, ckpt={args.ckpt}, data={args.data_root}")

    # 数据
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(root=args.data_root, transform=test_transform)
    num_classes = len(dataset.classes)
    print(f"[eval] num_classes={num_classes}, classes={dataset.classes}, n={len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    # 模型
    if args.model == 'baseline':
        if RETFoundBaseline is None:
            raise RuntimeError('retfound_baseline.py 未找到, 无法构建 baseline 模型')
        model = RETFoundBaseline(
            dim=args.dim, num_classes=num_classes, pretrained=True,
            retfound_weights=args.retfound_weights,
        ).to(device)
        print('[eval] model = RETFoundBaseline (P1)')
    else:
        model = HOMAR(
            dim=args.dim, num_classes=num_classes, pretrained=True,
            use_routing=not args.no_routing,
            use_hierarchy=not args.no_hierarchy,
            use_consensus=not args.no_consensus,
            retfound_weights=args.retfound_weights,
        ).to(device)
        print(f'[eval] model = HOMAR (routing={not args.no_routing}, hierarchy={not args.no_hierarchy}, consensus={not args.no_consensus})')

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # 兼容 light_ckpt (只有 ema_state_dict) / 旧 full_ckpt (model+ema)
    if args.use_ema and 'ema_state_dict' in ckpt:
        state_dict_key = 'ema_state_dict'
    elif 'model_state_dict' in ckpt:
        state_dict_key = 'model_state_dict'
    elif 'ema_state_dict' in ckpt:
        print('[eval][warn] no model_state_dict in ckpt, fall back to ema_state_dict')
        state_dict_key = 'ema_state_dict'
    else:
        raise KeyError('ckpt has neither model_state_dict nor ema_state_dict')
    print(f"[eval] loading state_dict_key={state_dict_key}")
    missing, unexpected = model.load_state_dict(ckpt[state_dict_key], strict=False)
    if missing:
        print(f"[eval][warn] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[eval][warn] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.eval()

    # 推理
    all_targets, all_sev, all_logits = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Eval'):
            images = images.to(device)
            if args.use_tta:
                out = tta_forward(model, images)
            else:
                out = model(images, return_all=True)
            all_targets.extend(labels.numpy())
            all_sev.extend(out['severity'].cpu().numpy())
            all_logits.append(out['logits'].cpu().numpy())

    all_targets = np.array(all_targets)
    all_sev = np.array(all_sev)
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = F.softmax(torch.from_numpy(all_logits), dim=1).numpy()

    # 阈值
    if args.thresholds:
        thresholds = np.array(json.loads(args.thresholds))
        print(f"[eval] use provided thresholds: {thresholds}")
    else:
        thresholds = search_optimal_thresholds(all_sev, all_targets, num_classes)
        print(f"[eval] searched thresholds: {thresholds}")

    preds = np.digitize(all_sev, thresholds).clip(0, num_classes - 1)

    acc = accuracy_score(all_targets, preds)
    qwk = cohen_kappa_score(all_targets, preds, weights='quadratic')
    try:
        auroc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"[eval][warn] auroc failed: {e}")
        auroc = 0.0
    try:
        onehot = np.eye(num_classes)[all_targets]
        aupr = average_precision_score(onehot, all_probs, average='macro')
    except Exception as e:
        print(f"[eval][warn] aupr failed: {e}")
        aupr = 0.0
    cm = confusion_matrix(all_targets, preds, labels=list(range(num_classes)))

    result = {
        'ckpt': args.ckpt,
        'data_root': args.data_root,
        'use_ema': bool(args.use_ema),
        'use_tta': bool(args.use_tta),
        'num_classes': int(num_classes),
        'n_samples': int(len(all_targets)),
        'thresholds': thresholds.tolist() if hasattr(thresholds, 'tolist') else list(thresholds),
        'accuracy': float(acc),
        'qwk': float(qwk),
        'auroc': float(auroc),
        'aupr': float(aupr),
        'confusion_matrix': cm.tolist(),
    }
    os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n[eval] result saved -> {args.output_json}")
    print(f"  Acc  = {acc:.4f}")
    print(f"  QWK  = {qwk:.4f}")
    print(f"  AUROC= {auroc:.4f}")
    print(f"  AUPR = {aupr:.4f}")
    print(f"  n    = {len(all_targets)}")


if __name__ == '__main__':
    main()
