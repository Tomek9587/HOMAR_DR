"""
简洁效率测量脚本：Params / FLOPs / FPS
输出到 stdout，同时写 JSON 和 Markdown README
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn

# 动态加载模块 (文件名带 '-' 不能直接 import)
import importlib.util

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# 加载 v2 (HOMAR 和 compute_model_efficiency)
v2 = _load_module(os.path.join(SRC_DIR, 'HOMAR-retfound-v2.py'), 'homar_v2')
HOMAR = v2.HOMAR
compute_model_efficiency = v2.compute_model_efficiency

# 加载 baseline
bl = _load_module(os.path.join(SRC_DIR, 'retfound_baseline.py'), 'retfound_baseline')
RETFoundBaseline = bl.RETFoundBaseline

# 加载 dinov2_dr (RETFoundDR)
dr = _load_module(os.path.join(SRC_DIR, 'dinov2_dr.py'), 'dinov2_dr')
RETFoundDR = dr.RETFoundDR

from thop import profile

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SIZE = (1, 3, 224, 224)
WARMUP = 20
REPEATS = 100


def measure(model, name, device=DEVICE):
    model = model.to(device).eval()
    dummy = torch.randn(*INPUT_SIZE).to(device)

    # 1. Params
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # 2. FLOPs (thop)
    try:
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        flops_g = flops / 1e9
    except Exception as e:
        print(f"[{name}] FLOPs measurement failed: {e}")
        flops_g = None

    # 3. FPS
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(dummy)
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(REPEATS):
            _ = model(dummy)
    torch.cuda.synchronize()
    avg_time = (time.time() - start) / REPEATS
    fps = 1.0 / avg_time

    return {
        'name': name,
        'params_M': round(total_params, 2),
        'trainable_M': round(trainable_params, 2),
        'flops_G': round(flops_g, 2) if flops_g is not None else None,
        'inference_ms': round(avg_time * 1000, 2),
        'fps': round(fps, 1),
    }


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input size: {INPUT_SIZE}")
    print(f"Warmup: {WARMUP}, Repeats: {REPEATS}")
    print("-" * 60)

    results = []

    # 1. RETFound ViT-L/16 (frozen backbone + linear head) — 对应 dinov2_dr.py
    # 注意：RETFoundDR 默认会尝试加载权重，但我们传 weights_path=None 且不预训练
    print("\n[1/3] Measuring RETFoundDR (ViT-L/16 + linear head)...")
    m1 = RETFoundDR(num_classes=5, drop_path=0.1, weights_path=None)
    r1 = measure(m1, "RETFound ViT-L/16 (frozen + linear head)")
    results.append(r1)
    print(f"  Params={r1['params_M']}M, FLOPs={r1['flops_G']}G, FPS={r1['fps']}")
    del m1
    torch.cuda.empty_cache()

    # 2. P1 baseline (RETFound + CORAL head)
    print("\n[2/3] Measuring RETFoundBaseline (P1: RETFound + CORAL head)...")
    m2 = RETFoundBaseline(dim=384, num_classes=5, pretrained=False, retfound_weights=None)
    r2 = measure(m2, "P1 baseline (RETFound + CORAL head)")
    results.append(r2)
    print(f"  Params={r2['params_M']}M, FLOPs={r2['flops_G']}G, FPS={r2['fps']}")
    del m2
    torch.cuda.empty_cache()

    # 3. HOMAR full
    print("\n[3/3] Measuring HOMAR (full model)...")
    m3 = HOMAR(dim=384, num_classes=5, pretrained=False, retfound_weights=None)
    r3 = measure(m3, "HOMAR (ViT-L + Mamba head, full)")
    results.append(r3)
    print(f"  Params={r3['params_M']}M, FLOPs={r3['flops_G']}G, FPS={r3['fps']}")
    del m3
    torch.cuda.empty_cache()

    # 输出汇总
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        flops_str = f"{r['flops_G']:.2f}" if r['flops_G'] is not None else "N/A"
        print(f"{r['name']:<45} | Params={r['params_M']:>7}M | FLOPs={flops_str:>7}G | FPS={r['fps']:>6}")
    print("=" * 60)

    # 保存 JSON
    out_json = os.path.join(SRC_DIR, '..', 'efficiency_results.json')
    with open(out_json, 'w') as f:
        json.dump({
            'device': str(DEVICE),
            'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu',
            'input_size': INPUT_SIZE,
            'warmup': WARMUP,
            'repeats': REPEATS,
            'models': results,
        }, f, indent=2)
    print(f"\nSaved JSON -> {out_json}")

    # 保存 README.md
    out_md = os.path.join(SRC_DIR, '..', 'EFFICIENCY.md')
    md_lines = [
        "# Efficiency Comparison",
        "",
        f"**Device:** {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu'})",
        f"**Input:** {INPUT_SIZE}",
        f"**Measurement:** {WARMUP} warmup + {REPEATS} repeats, batch_size=1",
        "",
        "| Model | Params (M) | FLOPs (G) | FPS |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        flops_str = f"{r['flops_G']:.2f}" if r['flops_G'] is not None else "N/A"
        md_lines.append(f"| {r['name']} | {r['params_M']:.2f} | {flops_str} | {r['fps']:.1f} |")
    md_lines.append("")
    md_lines.append("*FLOPs measured with `thop` at 224×224. FPS measured with `torch.no_grad()` on the actual GPU.*")

    with open(out_md, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved README -> {out_md}")
