"""
HOMAR-V2: Hierarchical Ordinal Mamba with Anatomical Routing (优化版)
在 V1 baseline 上新增：
  A1. ModelEMA (指数移动平均，抗过拟合平台期)
  A2. TTA (Test-Time Augmentation: horizontal flip 平均)
  A3. Early Stopping + 缩短默认 epoch 到 40
  A4. Stage 2 head_lr 1e-4 → 5e-5 (减少后期震荡)
预期 QWK 从 0.907 → 0.92~0.925
"""

import os
import argparse
import json
import copy
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import math
import time
import timm

from PIL import Image

# 默认 RETFound MAE Nature CFP 权重路径
DEFAULT_RETFOUND_WEIGHTS = '/root/autodl-tmp/longfei/image_classification/models/RETFound_mae_natureCFP/RETFound_mae_natureCFP.pth'

# ------------------------------------------------------------
# APTOS 标准预处理 Transform
# ------------------------------------------------------------

class BenGrahamTransform:
    """APTOS 标准预处理：Ben Graham's method + Circle Crop"""
    def __call__(self, img_pil):
        img = np.array(img_pil)
        # 1. Ben Graham 高斯模糊增强（降低强度，避免过度锐化丢失细节）
        img = cv2.addWeighted(img, 3, cv2.GaussianBlur(img, (0,0), 10), -2, 128)
        # 2. 圆形掩码（去除黑色背景）
        h, w = img.shape[:2]
        center = (w//2, h//2)
        radius = min(center[0], center[1], w-center[0], h-center[1])
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = dist_from_center <= radius
        img[~mask] = 0
        return Image.fromarray(img)


class CLAHETransform:
    """对 RGB 三通道分别做 CLAHE 增强（保留颜色信息）
    使用 LAB 色彩空间，只对 L（亮度）通道做 CLAHE，A/B 通道保留颜色
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img_pil):
        img = np.array(img_pil)
        if len(img.shape) == 2:
            # 灰度图直接处理
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img = clahe.apply(img)
            img = np.stack([img]*3, axis=-1)
        else:
            # RGB 图：转 LAB 色彩空间，只对 L 通道做 CLAHE，保留颜色
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img)


# ------------------------------------------------------------
# 0. 通用工具：Grad-CAM / 效率分析 / 训练曲线
# ------------------------------------------------------------

class GradCAM:
    """Grad-CAM可视化"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output[0], torch.Tensor):
                self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor.requires_grad_(True)
        
        output = self.model(input_tensor)
        if isinstance(output, dict):
            output = output.get('logits', list(output.values())[0])
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        for i in range(output.size(0)):
            one_hot[i, target_class[i]] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return torch.zeros(input_tensor.size(0), 14, 14, device=input_tensor.device)
        
        if self.activations.dim() == 4:  # (B, C, H, W)
            weights = self.gradients.mean(dim=[2, 3], keepdim=True)
            cam = (weights * self.activations).sum(dim=1)
        else:  # (B, L, C)
            weights = self.gradients.mean(dim=1, keepdim=True)
            cam = (weights * self.activations).sum(dim=-1)
            L = cam.size(1)
            h = int(math.sqrt(L))
            # ViT 输出含 CLS token (L=197)，需要去掉
            if h * h != L and (L - 1) > 0:
                h_nocls = int(math.sqrt(L - 1))
                if h_nocls * h_nocls == L - 1:
                    cam = cam[:, 1:]
                    h = h_nocls
            cam = cam.reshape(-1, h, h)
        
        cam = F.relu(cam)
        cam_min = cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam

    @staticmethod
    def visualize_cam(images, cams, labels, preds, save_path, num_samples=4):
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        for i in range(min(num_samples, images.size(0))):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'GT: {labels[i]}', fontsize=12)
            axes[i, 0].axis('off')
            
            cam = cams[i].cpu().numpy()
            cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0).float()
            cam_resized = F.interpolate(cam_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            cam_resized = cam_resized.squeeze().numpy()
            cam_colored = plt.cm.jet(cam_resized)[:, :, :3]
            
            axes[i, 1].imshow(cam_colored)
            axes[i, 1].set_title('Grad-CAM', fontsize=12)
            axes[i, 1].axis('off')
            
            overlay = 0.5 * img + 0.5 * cam_colored
            axes[i, 2].imshow(np.clip(overlay, 0, 1))
            axes[i, 2].set_title(f'Overlay (Pred: {preds[i]})', fontsize=12)
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def compute_model_efficiency(model, input_size=(1, 3, 224, 224), device='cpu'):
    model = model.to(device).eval()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    try:
        from thop import profile
        dummy = torch.randn(*input_size).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        flops_str = f"{flops / 1e9:.2f} GFLOPs"
    except ImportError:
        flops_str = "N/A (install thop)"
    
    dummy = torch.randn(*input_size).to(device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
    
    if isinstance(device, torch.device) and device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy)
    if isinstance(device, torch.device) and device.type == 'cuda':
        torch.cuda.synchronize()
    avg_time = (time.time() - start) / 20
    
    result = {
        'total_params_M': f"{total_params / 1e6:.2f}M",
        'trainable_params_M': f"{trainable_params / 1e6:.2f}M",
        'flops': flops_str,
        'inference_ms': f"{avg_time * 1000:.1f}ms",
        'fps': f"{1.0 / avg_time:.1f}"
    }
    print("\n" + "="*50 + "\n模型效率分析")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("="*50)
    return result


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss'); axes[0, 0].set_title('Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['val_acc'], 'g-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy'); axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    
    if 'val_qwk' in history:
        axes[1, 0].plot(epochs, history['val_qwk'], 'm-', label='Val QWK', linewidth=2)
        axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('QWK'); axes[1, 0].set_title('QWK')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    
    if 'val_auroc' in history and 'val_aupr' in history:
        axes[1, 1].plot(epochs, history['val_auroc'], 'r-', label='AUROC', linewidth=2)
        axes[1, 1].plot(epochs, history['val_aupr'], 'orange', linewidth=2, linestyle='--', label='AUPR')
        axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Score'); axes[1, 1].set_title('AUROC & AUPR')
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------
# 1. Mamba胶囊专家（每个解剖区域一个）
# ------------------------------------------------------------

class MambaCapsuleExpert(nn.Module):
    """
    真正的 Mamba-Capsule 专家：
    1. 先用 Mamba SSM 处理完整区域 token 序列（有意义的序列长度）
    2. 再用 routing-by-agreement 从 SSM 输出中提取胶囊表示
    """
    def __init__(self, dim, num_caps, d_state=16, cap_dim=None, routing_iters=3):
        super().__init__()
        self.num_caps = num_caps
        self.cap_dim = cap_dim or dim
        self.routing_iters = routing_iters
        
        # === Mamba SSM 部分（处理完整序列）===
        self.norm_in = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, groups=dim, padding=3)
        self.log_A = nn.Parameter(torch.log(torch.arange(1, d_state+1).float().repeat(dim, 1)))
        self.B_proj = nn.Linear(dim, d_state, bias=False)
        self.C_proj = nn.Linear(dim, d_state, bias=False)
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
        
        # === Capsule Routing 部分 ===
        # 将 token 投影到 vote 空间: (B, L, C) -> (B, L, num_caps, cap_dim)
        self.vote_proj = nn.Linear(dim, num_caps * self.cap_dim)
        
        # 存在性预测
        self.existence_head = nn.Sequential(
            nn.Linear(self.cap_dim, self.cap_dim // 2),
            nn.ReLU(),
            nn.Linear(self.cap_dim // 2, 1),
            nn.Sigmoid()
        )
        self.norm_out = nn.LayerNorm(self.cap_dim)
    
    def squash(self, s):
        """Capsule squash非线性: 将向量压缩到[0,1)范围但保持方向"""
        sq_norm = (s ** 2).sum(dim=-1, keepdim=True)
        scale = sq_norm / (1 + sq_norm) / (torch.sqrt(sq_norm) + 1e-8)
        return scale * s
    
    def forward(self, region_tokens):
        """
        region_tokens: (B, L, C)
        返回: {'capsules': (B, num_caps, cap_dim), 'existence': (B, num_caps), 'routing_weights': (B, L, num_caps)}
        """
        B, L, C = region_tokens.shape
        
        # Step 1: Mamba SSM 处理完整序列
        mamba_out = self._mamba_forward(region_tokens)  # (B, L, C)
        
        # Step 2: Routing-by-Agreement 提取胶囊
        # 计算 votes: 每个 token 对每个 capsule 的投票
        votes = self.vote_proj(mamba_out).reshape(B, L, self.num_caps, self.cap_dim)  # (B, L, num_caps, cap_dim)
        
        # 迭代路由
        logits = torch.zeros(B, L, self.num_caps, device=region_tokens.device)
        
        for routing_iter in range(self.routing_iters):
            coupling = F.softmax(logits, dim=2)  # (B, L, num_caps) - 每个token分配给各capsule的权重
            # 加权求和得到 capsule 输入
            capsule_input = (coupling.unsqueeze(-1) * votes).sum(dim=1)  # (B, num_caps, cap_dim)
            # squash 非线性
            capsule_output = self.squash(capsule_input)  # (B, num_caps, cap_dim)
            
            if routing_iter < self.routing_iters - 1:
                # 计算 agreement（votes 和 capsule_output 的点积）
                agreement = (votes * capsule_output.unsqueeze(1)).sum(dim=-1)  # (B, L, num_caps)
                logits = logits + agreement
        
        capsules = self.norm_out(capsule_output)  # (B, num_caps, cap_dim)
        existence = self.existence_head(capsules).squeeze(-1)  # (B, num_caps)
        
        # 最终的 coupling coefficients 作为路由权重（可视化用）
        final_routing = F.softmax(logits, dim=2)  # (B, L, num_caps)
        
        return {
            'capsules': capsules,
            'existence': existence,
            'routing_weights': final_routing
        }
    
    def _mamba_forward(self, x):
        """Mamba SSM 前向（处理完整序列）"""
        B, L, C = x.shape
        residual = x
        x = self.norm_in(x)
        
        x_proj = self.in_proj(x)
        x_inner, gate = x_proj.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        dt = F.softplus(self.dt_proj(x_conv))
        B_ssm = self.B_proj(x_conv)
        C_ssm = self.C_proj(x_conv)
        A = -torch.exp(self.log_A).float()
        
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)
        
        # Sequential scan（完整序列，L通常是50-100范围，可接受）
        h = torch.zeros(B, C, A.size(1), device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x_conv[:, i].unsqueeze(-1)
            y = torch.einsum('bcd,bd->bc', h, C_ssm[:, i])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        
        y = y * F.silu(gate)
        y = self.out_proj(y)
        return y + residual


# ------------------------------------------------------------
# 2. 解剖学路由器（动态门控）
# ------------------------------------------------------------

class AnatomicalRouter(nn.Module):
    """
    根据图像内容动态路由到3个区域专家
    输出软权重（可学习），而非硬选择
    """
    def __init__(self, dim: int, num_regions: int = 3, temperature: float = 1.0):
        super().__init__()
        self.num_regions = num_regions
        self.temperature = temperature
        
        # 全局特征提取用于路由决策
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.router_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, num_regions)
        )
        
        # 可学习的先验偏置（反映先验知识：黄斑更重要）
        self.prior_bias = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))  # macula, optic, vessel
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, L, C)
        返回: routing_weights (B, 3)，软权重，和为1
        """
        # 全局特征
        global_feat = tokens.mean(dim=1)  # (B, C)
        
        # 路由logits
        logits = self.router_mlp(global_feat)  # (B, 3)
        
        # 加先验偏置，softmax
        weights = F.softmax((logits + self.prior_bias) / self.temperature, dim=1)
        
        return weights


# ------------------------------------------------------------
# 3. 层次化Mamba（3层）
# ------------------------------------------------------------

class MambaBlock(nn.Module):
    """独立的Mamba处理块，用于层次化处理"""
    def __init__(self, dim, d_state=16, kernel_size=4, dilation=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim,
                                padding=effective_kernel - 1, dilation=dilation)
        self.log_A = nn.Parameter(torch.log(torch.arange(1, d_state+1).float().repeat(dim, 1)))
        self.B_proj = nn.Linear(dim, d_state, bias=False)
        self.C_proj = nn.Linear(dim, d_state, bias=False)
        self.dt_proj = nn.Linear(dim, dim, bias=True)
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, L, C = x.shape
        residual = x
        x = self.norm(x)
        x_proj = self.in_proj(x)
        x_inner, gate = x_proj.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        dt = F.softplus(self.dt_proj(x_conv))
        B_ssm = self.B_proj(x_conv)
        C_ssm = self.C_proj(x_conv)
        A = -torch.exp(self.log_A).float()
        
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)
        
        h = torch.zeros(B, C, A.size(1), device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x_conv[:, i].unsqueeze(-1)
            y = torch.einsum('bcd,bd->bc', h, C_ssm[:, i])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        
        y = y * F.silu(gate)
        return self.out_proj(y) + residual


class HierarchicalMamba(nn.Module):
    """3层独立Mamba块"""
    def __init__(self, dim, num_levels=3):
        super().__init__()
        self.levels = nn.ModuleList([
            MambaBlock(dim, kernel_size=4, dilation=1),   # Level 1: Local
            MambaBlock(dim, kernel_size=4, dilation=3),   # Level 2: Dilated
            MambaBlock(dim, kernel_size=4, dilation=1),   # Level 3: Global
        ])
        self.transitions = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
            for _ in range(num_levels - 1)
        ])
        self.existence_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
            for _ in range(num_levels)
        ])
    
    def forward(self, fused_features):
        """fused_features: (B, L, C)"""
        level_outputs = []
        current = fused_features
        
        for i, level in enumerate(self.levels):
            out = level(current)  # (B, L, C)
            existence = self.existence_heads[i](out).mean(dim=1)  # (B, 1)
            level_outputs.append({
                'features': out,  # (B, L, C) — 不是capsules了
                'existence': existence
            })
            if i < len(self.transitions):
                current = self.transitions[i](out)
            else:
                current = out
        
        return level_outputs


# ------------------------------------------------------------
# 4. 跨层一致性（Consensus）
# ------------------------------------------------------------

class CrossLevelConsensus(nn.Module):
    """跨层注意力（替代纯加权平均）"""
    def __init__(self, dim, num_levels=3):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_levels)])
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, level_outputs):
        level_feats = []
        for i, out in enumerate(level_outputs):
            # out['features']: (B, L, C) -> mean pool -> (B, C)
            feat = out['features'].mean(dim=1)
            feat = self.projections[i](feat)
            level_feats.append(feat)
        
        # Stack: (B, 3, C)
        stacked = torch.stack(level_feats, dim=1)
        normed = self.norm(stacked)
        
        # Self-attention across levels
        attn_out, _ = self.cross_attn(normed, normed, normed)  # (B, 3, C)
        
        # 加权聚合为 (B, C)
        consensus = self.out_proj(attn_out.mean(dim=1))
        return consensus


# ------------------------------------------------------------
# 5. Wasserstein序数回归
# ------------------------------------------------------------

class WassersteinOrdinalHead(nn.Module):
    def __init__(self, dim: int, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # 5个等级原型
        self.prototypes = nn.Parameter(torch.randn(num_classes, 128))
        
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        x: (B, C) - Consensus特征
        """
        feat = self.feature_proj(x)  # (B, 128)
        
        # 相似度
        logits = torch.matmul(feat, self.prototypes.t())  # (B, 5)
        prob_dist = F.softmax(logits, dim=1)
        
        # Severity score
        severity = torch.sum(prob_dist * torch.arange(self.num_classes, device=x.device), dim=1)
        
        if targets is not None:
            # 损失计算
            ce = F.cross_entropy(logits, targets)
            
            # Wasserstein距离
            pred_cdf = torch.cumsum(prob_dist, dim=1)
            target_cdf = torch.cumsum(F.one_hot(targets, self.num_classes).float(), dim=1)
            w_dist = torch.sum(torch.abs(pred_cdf - target_cdf), dim=1).mean()
            
            # 序数惩罚
            target_f = targets.float()
            ord_pen = torch.abs(severity - target_f)
            ord_pen = torch.where(ord_pen > 1, ord_pen * 2.0, ord_pen).mean()
            
            loss = ce + 1.0 * w_dist + 0.1 * ord_pen
            return loss, severity, prob_dist, logits
        
        return severity, prob_dist, logits


# ------------------------------------------------------------
# 5b. CORAL序数回归（替代 WassersteinOrdinalHead）
# ------------------------------------------------------------

class CORALOrdinalHead(nn.Module):
    """
    CORAL: Consistent Rank Logits for Ordinal Regression
    参考: Cao et al., "Rank-consistent Ordinal Regression for Neural Networks"
    """
    def __init__(self, dim, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes - 1)  # K-1 个二分类任务
        )
    
    def forward(self, x, targets=None):
        logits = self.fc(x)  # (B, K-1)
        
        # sigmoid 得到 P(y > k)
        prob_levels = torch.sigmoid(logits)  # (B, K-1)
        # 移除 torch.sort：保留梯度流，单调性由 CORAL loss 自然约束
        
        # 转换为类别概率
        probs = torch.zeros(x.size(0), self.num_classes, device=x.device)
        probs[:, 0] = 1 - prob_levels[:, 0]
        for i in range(1, self.num_classes - 1):
            probs[:, i] = prob_levels[:, i-1] - prob_levels[:, i]
        probs[:, -1] = prob_levels[:, -1]
        
        # 数值安全：sigmoid 输出不保证单调递减，差值可能为负
        probs = probs.clamp(min=1e-7, max=1.0)
        probs = probs / probs.sum(dim=1, keepdim=True)  # 重归一化确保概率和为1
        
        # Severity score（加权求和）
        severity = torch.sum(probs * torch.arange(self.num_classes, device=x.device, dtype=x.dtype), dim=1)
        
        if targets is not None:
            # CORAL loss: K-1 个二元交叉熵
            target_binary = torch.zeros(x.size(0), self.num_classes - 1, device=x.device)
            for i in range(self.num_classes - 1):
                target_binary[:, i] = (targets > i).float()
            loss = F.binary_cross_entropy_with_logits(logits, target_binary, reduction='mean')
            return loss, severity, probs, logits
        
        return severity, probs, logits


# ------------------------------------------------------------
# 5c. Soft QWK Loss
# ------------------------------------------------------------

class SoftQWKLoss(nn.Module):
    """QWK 的软近似，直接可微优化"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        weights = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                weights[i, j] = (i - j) ** 2 / (num_classes - 1) ** 2
        self.register_buffer('weights', weights)
    
    def forward(self, logits_or_probs, targets, is_probs=False):
        if is_probs:
            probs = logits_or_probs
        else:
            probs = F.softmax(logits_or_probs, dim=1)
        target_onehot = F.one_hot(targets, self.num_classes).float()
        
        O = torch.matmul(target_onehot.t(), probs) / targets.size(0)
        row_sum = target_onehot.sum(dim=0) / targets.size(0)
        col_sum = probs.sum(dim=0) / targets.size(0)
        E = torch.outer(row_sum, col_sum)
        
        num = (self.weights * O).sum()
        den = (self.weights * E).sum() + 1e-8
        
        qwk = 1.0 - num / den
        return 1.0 - qwk  # 最小化 1 - QWK


# ------------------------------------------------------------
# 6. HOMAR主网络
# ------------------------------------------------------------

class RETFoundBackbone(nn.Module):
    """
    加载 RETFound MAE 预训练的 ViT-Large/16，输出 patch token 序列
    输入: (B, 3, 224, 224)
    输出: (B, 196, 1024) — 已去掉 CLS token
    """
    def __init__(self, weights_path=None, drop_path=0.1):
        super().__init__()
        self.vit = timm.create_model(
            'vit_large_patch16_224',
            pretrained=False,
            num_classes=0,
            drop_path_rate=drop_path,
            global_pool='',   # 保留 token 序列，不做 pooling
        )
        self.embed_dim = self.vit.embed_dim  # 1024

        if weights_path and os.path.exists(weights_path):
            print(f"[RETFoundBackbone] Loading weights: {weights_path}")
            raw = torch.load(weights_path, map_location='cpu', weights_only=False)

            if isinstance(raw, dict):
                if 'model' in raw:
                    state = raw['model']
                elif 'teacher' in raw:
                    state = raw['teacher']
                    state = {k.replace('backbone.', '', 1) if k.startswith('backbone.') else k: v
                             for k, v in state.items()}
                elif 'state_dict' in raw:
                    state = raw['state_dict']
                else:
                    state = raw
            else:
                state = raw

            # 过滤 MAE decoder / mask_token / head
            state = {k: v for k, v in state.items()
                    if not k.startswith('decoder')
                    and k != 'mask_token'
                    and not k.startswith('head')}
            # 注意: timm ViT 当 global_pool='' 时，最后的 LayerNorm 叫 self.norm，
            # 与 MAE 权重中的 'norm' key 一致，不需重命名。

            msg = self.vit.load_state_dict(state, strict=False)
            loaded = len(state) - len(msg.unexpected_keys)
            print(f"[RETFoundBackbone] loaded {loaded}/{len(state)} keys, "
                  f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
            if msg.missing_keys:
                print(f"  missing (first 5): {msg.missing_keys[:5]}")
            if msg.unexpected_keys:
                print(f"  unexpected (first 5): {msg.unexpected_keys[:5]}")
        else:
            print(f"[RETFoundBackbone] WARNING: no weights loaded, using random init. path={weights_path}")

    def forward(self, x):
        # timm ViT forward_features -> (B, 197, 1024)，包含 CLS
        feats = self.vit.forward_features(x)
        # 去掉 CLS token，只保留 patch tokens (B, 196, 1024)
        return feats[:, 1:, :]


class HOMAR(nn.Module):
    """
    Hierarchical Ordinal Mamba with Anatomical Routing
    整合：解剖学路由 + 胶囊专家 + 层次化Mamba + Consensus + 序数回归
    支持消融实验配置
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        dim: int = 384,
        num_classes: int = 5,
        pretrained: bool = True,
        use_routing: bool = True,
        use_hierarchy: bool = True,
        use_consensus: bool = True,
        num_experts: int = 3,
        retfound_weights: Optional[str] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.dim = dim
        self.use_routing = use_routing
        self.use_hierarchy = use_hierarchy
        self.use_consensus = use_consensus
        
        # 1. Backbone (RETFound MAE ViT-Large/16)
        self.backbone = RETFoundBackbone(
            weights_path=retfound_weights if pretrained else None
        )
        # ViT 输出 (B, 196, 1024) — 直接用 Linear 投影到 dim
        self.proj = nn.Sequential(
            nn.LayerNorm(self.backbone.embed_dim),
            nn.Linear(self.backbone.embed_dim, dim),
        )
        
        # 2. 解剖学分区和路由（可消融）
        # 基础掩码 (14x14 grid)
        self.register_buffer('macula_mask', self._create_mask(img_size, patch_size, 'center'))
        self.register_buffer('optic_mask', self._create_mask(img_size, patch_size, 'optic'))
        self.register_buffer('vessel_mask', self._create_mask(img_size, patch_size, 'arc'))
        
        # 3个区域专家（Mamba-Capsule）
        self.experts = nn.ModuleDict({
            'macula': MambaCapsuleExpert(dim, num_caps=8),
            'optic': MambaCapsuleExpert(dim, num_caps=4),
            'vessel': MambaCapsuleExpert(dim, num_caps=4)
        })
        
        # 路由器（可消融）
        if self.use_routing:
            self.router = AnatomicalRouter(dim, num_regions=3)
        
        # 3. 层次化Mamba（可消融）
        if self.use_hierarchy:
            self.hierarchy = HierarchicalMamba(dim)
        
        # 4. Consensus（可消融）
        if self.use_consensus:
            self.consensus = CrossLevelConsensus(dim)
        
        # 5. 序数回归（CORAL 替代 Wasserstein）
        self.ordinal_head = CORALOrdinalHead(dim, num_classes)
        
        # 6. 辅助分类头
        self.classifier = nn.Linear(dim, num_classes)
        
    def _create_mask(self, img_size, patch_size, mask_type):
        """创建解剖学掩码 (14x14)"""
        h = w = img_size // patch_size
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        
        if mask_type == 'center':
            # 黄斑中心圆
            cy, cx = h // 2, w // 2
            mask = ((y - cy)**2 + (x - cx)**2 <= (h * 0.15)**2).float()
        elif mask_type == 'optic':
            # 视盘（右上）
            cy, cx = int(h * 0.5), int(w * 0.75)
            mask = (((y - cy) / (h*0.1))**2 + ((x - cx) / (w*0.08))**2 <= 1).float()
        else:  # arc
            # 血管弓（左右两侧）
            mask = (((x < w * 0.3) | (x > w * 0.7)) & (torch.abs(y - h/2) < h*0.3)).float()
        
        return mask.flatten()  # (196,)
    
    def forward(self, img: torch.Tensor, return_all: bool = False):
        B = img.size(0)
        
        # 1. 特征提取（RETFound ViT-L/16）
        patch_tokens = self.backbone(img)          # (B, 196, 1024)
        tokens = self.proj(patch_tokens)           # (B, 196, dim)
        
        # 2. 解剖学分区 + 路由
        if self.use_routing:
            routing_weights = self.router(tokens)  # (B, 3)
        else:
            routing_weights = torch.ones(B, 3, device=img.device) / 3.0
        
        # 应用固定解剖学掩码到各区域
        region_tokens = {
            'macula': tokens * self.macula_mask.unsqueeze(0).unsqueeze(-1),
            'optic': tokens * self.optic_mask.unsqueeze(0).unsqueeze(-1),
            'vessel': tokens * self.vessel_mask.unsqueeze(0).unsqueeze(-1)
        }
        
        # 3. 专家处理（并行）
        expert_outputs = {}
        for name, expert in self.experts.items():
            expert_outputs[name] = expert(region_tokens[name])
        
        # 4. 路由加权聚合专家输出（软MoE）
        # 将所有专家的胶囊拼接并投影为统一序列
        all_capsules = []
        for i, (name, out) in enumerate(expert_outputs.items()):
            w = routing_weights[:, i:i+1, None]  # (B, 1, 1)
            all_capsules.append(out['capsules'] * w)  # (B, num_caps, C)
        
        fused_capsules = torch.cat(all_capsules, dim=1)  # (B, total_caps, C)
        
        # 5. 层次化Mamba（可消融）
        if self.use_hierarchy:
            level_outputs = self.hierarchy(fused_capsules)
        else:
            # 消融：直接将融合的胶囊当作单层输出
            level_outputs = [{
                'features': fused_capsules,
                'existence': torch.ones(B, 1, device=img.device)
            }]
        
        # 6. Consensus聚合（可消融）
        if self.use_consensus and len(level_outputs) > 1:
            consensus_feat = self.consensus(level_outputs)
        else:
            # 消融：简单平均池化
            all_feats = [out['features'].mean(dim=1) for out in level_outputs]
            consensus_feat = torch.stack(all_feats, dim=0).mean(dim=0)
        
        # 7. 预测
        severity, prob_dist, ord_logits = self.ordinal_head(consensus_feat, None)
        
        cls_logits = self.classifier(consensus_feat)
        
        if return_all:
            return {
                'logits': cls_logits,
                'severity': severity,
                'prob_dist': prob_dist,
                'ord_logits': ord_logits,
                'routing_weights': routing_weights,
                'expert_outputs': expert_outputs,
                'level_outputs': level_outputs,
                'consensus_feat': consensus_feat
            }
        
        return cls_logits
    
    def visualize(self, x, save_path='homar_vis.png'):
        """可视化路由决策和专家激活"""
        self.eval()
        with torch.no_grad():
            out = self(x[:4], return_all=True)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        for i in range(4):
            # 原图
            img = x[i].cpu().permute(1, 2, 0).numpy()
            img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            img = np.clip(img, 0, 1)
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f'Input')
            axes[i, 0].axis('off')
            
            # 路由权重（饼图/柱状图）
            weights = out['routing_weights'][i].cpu().numpy()
            axes[i, 1].bar(['Macula', 'Optic', 'Vessel'], weights)
            axes[i, 1].set_title(f'Routing\nM:{weights[0]:.2f} O:{weights[1]:.2f} V:{weights[2]:.2f}')
            axes[i, 1].set_ylim(0, 1)
            
            # 专家存在性（胶囊激活）
            mac_exist = out['expert_outputs']['macula']['existence'][i].mean().item()
            opt_exist = out['expert_outputs']['optic']['existence'][i].mean().item()
            ves_exist = out['expert_outputs']['vessel']['existence'][i].mean().item()
            axes[i, 2].bar(['Mac', 'Opt', 'Ves'], [mac_exist, opt_exist, ves_exist])
            axes[i, 2].set_title('Expert Activation')
            axes[i, 2].set_ylim(0, 1)
            
            # 最终预测
            sev = out['severity'][i].item()
            axes[i, 3].text(0.5, 0.5, f'Severity:\n{sev:.2f}', 
                           ha='center', va='center', fontsize=20)
            axes[i, 3].set_xlim(0, 1)
            axes[i, 3].set_ylim(0, 1)
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


# ------------------------------------------------------------
# 7. 训练器
# ------------------------------------------------------------

# ====== A1. ModelEMA: 指数移动平均 ======
class ModelEMA:
    """对模型权重做 EMA，只维护一份影子模型。
    更新规则: v_ema = decay * v_ema + (1 - decay) * v_current
    验证时用 self.ema 替换原模型权重。
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # deepcopy 并保持 eval 模式，不累积梯度
        self.ema = copy.deepcopy(self._get_core(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _get_core(model):
        return model.module if hasattr(model, 'module') else model

    @torch.no_grad()
    def update(self, model):
        msd = self._get_core(model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()


# ====== A2. TTA: 水平翻转平均 ======
@torch.no_grad()
def tta_forward(model, images):
    """对输入做水平翻转 + 原图两次前向，平均 severity 和 logits。
    返回与 model.forward(images, return_all=True) 结构相容的 dict
    (仅包含后续用到的 key: severity, logits, prob_dist, ord_logits)。
    """
    out1 = model(images, return_all=True)
    out2 = model(torch.flip(images, dims=[3]), return_all=True)
    merged = {
        'severity':   (out1['severity']   + out2['severity'])   / 2,
        'logits':     (out1['logits']     + out2['logits'])     / 2,
        'prob_dist':  (out1['prob_dist']  + out2['prob_dist'])  / 2,
        'ord_logits': (out1['ord_logits'] + out2['ord_logits']) / 2,
    }
    return merged


class HOMARTrainer:
    def __init__(self, model, device, save_dir='./homar_results',
                 freeze_epochs=10, total_epochs=100,
                 ema_decay=0.999, patience=10, use_tta=True):
        """
        两阶段训练（方案 B）+ EMA + TTA + Early Stopping (V2)：
          Stage 1 (epoch 0 ~ freeze_epochs-1): backbone 冻结，只训 head，head_lr=5e-4
          Stage 2 (epoch freeze_epochs ~ total_epochs-1): 解冻 backbone，backbone_lr=1e-5, head_lr=5e-5
          EMA decay=0.999，验证/保存时使用 EMA 权重
          TTA: 水平翻转平均
          Early Stopping: val QWK 连续 patience 个 epoch 不提升则停止
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.freeze_epochs = freeze_epochs
        self.total_epochs = total_epochs
        self.patience = patience
        self.use_tta = use_tta
        os.makedirs(save_dir, exist_ok=True)

        self.use_amp = torch.cuda.is_bf16_supported()
        self.scaler = GradScaler(enabled=self.use_amp)

        # ====== Stage 1: 冻结 backbone，只训 head ======
        for p in self.model.backbone.parameters():
            p.requires_grad_(False)

        self.stage = 1
        self._build_stage1_optimizer()

        # ====== A1. EMA ======
        self.ema = ModelEMA(self.model, decay=ema_decay)

        # Soft QWK Loss
        self.qwk_loss_fn = SoftQWKLoss(num_classes=model.ordinal_head.num_classes).to(device)

        self.best_qwk = 0.0
        self.best_epoch = -1
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_acc': [], 'val_qwk': [], 'val_auroc': [], 'val_aupr': []}

        print(f"[Trainer-V2] Stage1={freeze_epochs} ep (linear probing), "
              f"Stage2={max(total_epochs - freeze_epochs, 0)} ep (fine-tune), "
              f"EMA={ema_decay}, TTA={'ON' if use_tta else 'OFF'}, ES patience={patience}")

    # ---------- 分阶段优化器 ----------
    def _build_stage1_optimizer(self):
        """Stage 1: 仅训 head, head_lr=5e-4, 2 ep warmup + cosine 到 freeze_epochs"""
        model = self.model
        head_groups = [
            {'params': model.proj.parameters(),     'lr': 5e-4},
            {'params': model.experts.parameters(),  'lr': 5e-4},
        ]
        if hasattr(model, 'router') and model.use_routing:
            head_groups.append({'params': model.router.parameters(), 'lr': 5e-4})
        if hasattr(model, 'hierarchy') and model.use_hierarchy:
            head_groups.append({'params': model.hierarchy.parameters(), 'lr': 5e-4})
        if hasattr(model, 'consensus') and model.use_consensus:
            head_groups.append({'params': model.consensus.parameters(), 'lr': 5e-4})
        head_groups.extend([
            {'params': model.ordinal_head.parameters(), 'lr': 5e-4},
            {'params': model.classifier.parameters(),   'lr': 5e-4},
        ])
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
        """Stage 2: 解冻 backbone, backbone_lr=1e-5, head_lr=5e-5, 2 ep warmup + cosine 到 total_epochs (V2 降低后期震荡)"""
        model = self.model
        for p in model.backbone.parameters():
            p.requires_grad_(True)

        param_groups = [
            {'params': model.backbone.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
            {'params': model.proj.parameters(),     'lr': 5e-5},
            {'params': model.experts.parameters(),  'lr': 5e-5},
        ]
        if hasattr(model, 'router') and model.use_routing:
            param_groups.append({'params': model.router.parameters(), 'lr': 5e-5})
        if hasattr(model, 'hierarchy') and model.use_hierarchy:
            param_groups.append({'params': model.hierarchy.parameters(), 'lr': 5e-5})
        if hasattr(model, 'consensus') and model.use_consensus:
            param_groups.append({'params': model.consensus.parameters(), 'lr': 5e-5})
        param_groups.extend([
            {'params': model.ordinal_head.parameters(), 'lr': 5e-5},
            {'params': model.classifier.parameters(),   'lr': 5e-5},
        ])
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

    def unfreeze_backbone(self):
        self._build_stage2_optimizer()
        self.stage = 2
        print(f"[Trainer] >>> Stage 2 开始：backbone 解冻，backbone_lr=1e-5, head_lr=5e-5")
        
    def train_epoch(self, train_loader):
        self.model.train()
        # Stage 1: backbone 保持 eval 模式（关闭 dropout），避免扰乱预训练特征
        if self.stage == 1:
            self.model.backbone.eval()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f'Training(S{self.stage})'):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            dtype = torch.bfloat16 if self.use_amp else torch.float32
            with autocast(enabled=self.use_amp, dtype=dtype):
                # Stage 1 时 backbone 不需要梯度，用 no_grad 包住节省显存
                if self.stage == 1:
                    with torch.no_grad():
                        patch_tokens = self.model.backbone(images)
                    out = self.model(images, return_all=True, patch_tokens=patch_tokens) \
                          if self._model_supports_cached_tokens() else self.model(images, return_all=True)
                else:
                    out = self.model(images, return_all=True)

                # CORAL 序数损失
                nc = self.model.ordinal_head.num_classes
                target_binary = torch.zeros(images.size(0), nc - 1, device=images.device)
                for i in range(nc - 1):
                    target_binary[:, i] = (labels > i).float()
                coral_loss = F.binary_cross_entropy_with_logits(
                    out['ord_logits'], target_binary, reduction='mean'
                )

                # Soft QWK Loss
                qwk_loss = self.qwk_loss_fn(out['prob_dist'], labels, is_probs=True)

                # 辅助 CE + label smoothing
                ce = F.cross_entropy(out['logits'], labels, label_smoothing=0.1)

                total_batch = coral_loss + 1.0 * qwk_loss + 0.3 * ce

            self.scaler.scale(total_batch).backward()
            self.scaler.unscale_(self.optimizer)
            # 更严的梯度裁剪（抗崩盘）
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 0.5
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # A1. EMA 更新：每个 step 更新一次影子模型
            self.ema.update(self.model)

            total_loss += total_batch.item()

        return total_loss / len(train_loader)

    def _model_supports_cached_tokens(self):
        # 兼容性开关：若 HOMAR.forward 未接收 patch_tokens 关键字，则按原路径前向
        import inspect
        sig = inspect.signature(self.model.forward)
        return 'patch_tokens' in sig.parameters
    
    @torch.no_grad()
    def validate(self, test_loader, use_ema=True, use_tta=None):
        """
        V2 验证：
          - use_ema=True 时用 EMA 权重
          - use_tta=True 时水平翻转平均
        """
        if use_tta is None:
            use_tta = self.use_tta

        # 选择用于验证的模型（EMA 或原模型）
        if use_ema:
            # 临时将 EMA 权重 load 到原模型做验证，验证后恢复
            backup = copy.deepcopy(self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict(), strict=False)

        self.model.eval()
        all_targets, all_sev, all_logits = [], [], []
        nc = self.model.ordinal_head.num_classes

        for images, labels in tqdm(test_loader, desc='Validating' + (' [EMA+TTA]' if use_ema and use_tta else ' [EMA]' if use_ema else '')):
            images = images.to(self.device)
            if use_tta:
                out = tta_forward(self.model, images)
            else:
                out = self.model(images, return_all=True)
            severity = out['severity']
            all_targets.extend(labels.numpy())
            all_sev.extend(severity.cpu().numpy())
            all_logits.append(out['logits'].cpu().numpy())

        # 恢复原模型权重
        if use_ema:
            self.model.load_state_dict(backup)

        all_targets = np.array(all_targets)
        all_sev = np.array(all_sev)
        all_logits = np.concatenate(all_logits, axis=0)
        all_probs = F.softmax(torch.from_numpy(all_logits), dim=1).numpy()

        # --- 最优阈值搜索 ---
        best_thresholds = self._search_optimal_thresholds(all_sev, all_targets, nc)
        all_preds = np.digitize(all_sev, best_thresholds).clip(0, nc - 1)

        acc = accuracy_score(all_targets, all_preds)
        qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')

        # round 版本作为参考
        round_preds = np.round(all_sev).astype(int).clip(0, nc - 1)
        round_qwk = cohen_kappa_score(all_targets, round_preds, weights='quadratic')

        # AUROC / AUPR
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            auroc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            auroc = 0.0
        try:
            from sklearn.metrics import average_precision_score
            target_onehot = np.eye(nc)[all_targets]
            aupr = average_precision_score(target_onehot, all_probs, average='macro')
        except ValueError:
            aupr = 0.0

        return {
            'accuracy': acc,
            'qwk': qwk,
            'round_qwk': round_qwk,
            'auroc': auroc,
            'aupr': aupr,
            'thresholds': best_thresholds.tolist(),
            'confusion_matrix': confusion_matrix(all_targets, all_preds)
        }
    
    def _search_optimal_thresholds(self, severities, targets, nc):
        """网格搜索 K-1 个最优阈值，使 QWK 最大化"""
        # 初始阈值：0.5, 1.5, 2.5, 3.5
        best_thresholds = np.arange(nc - 1) + 0.5
        best_qwk = -1.0
        
        # 逐阈值贪心优化（迭代3轮收敛）
        for _ in range(3):
            for k in range(nc - 1):
                best_t = best_thresholds[k]
                for t in np.arange(max(k * 0.5, 0.0), min((k + 2) * 1.0, nc - 0.01), 0.05):
                    trial = best_thresholds.copy()
                    trial[k] = t
                    trial.sort()  # 保持单调性
                    preds = np.digitize(severities, trial).clip(0, nc - 1)
                    q = cohen_kappa_score(targets, preds, weights='quadratic')
                    if q > best_qwk:
                        best_qwk = q
                        best_t = t
                best_thresholds[k] = best_t
                best_thresholds.sort()
        
        return best_thresholds
    
    def train(self, train_loader, test_loader, epochs=100):
        for epoch in range(epochs):
            # 阶段切换：到达 freeze_epochs 时解冻 backbone
            if self.stage == 1 and epoch == self.freeze_epochs:
                self.unfreeze_backbone()
            train_loss = self.train_epoch(train_loader)
            metrics = self.validate(test_loader)  # V2: 默认 EMA + TTA
            
            self.history['train_loss'].append(train_loss)
            self.history['val_acc'].append(metrics['accuracy'])
            self.history['val_qwk'].append(metrics['qwk'])
            self.history['val_auroc'].append(metrics['auroc'])
            self.history['val_aupr'].append(metrics['aupr'])

            improved = metrics['qwk'] > self.best_qwk
            if improved:
                self.best_qwk = metrics['qwk']
                self.best_epoch = epoch
                self.patience_counter = 0
                # V2: 同时保存原模型 + EMA 权重
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'ema_state_dict': self.ema.state_dict(),
                    'metrics': metrics,
                    'history': self.history
                }, os.path.join(self.save_dir, 'best_model.pth'))
                
                if epoch > 0:
                    self.model.visualize(next(iter(test_loader))[0].to(self.device),
                                       os.path.join(self.save_dir, f'vis_epoch{epoch}.png'))
            else:
                self.patience_counter += 1
            
            # 每10轮生成可视化和报告
            if (epoch + 1) % 10 == 0:
                plot_training_curves(self.history, os.path.join(self.save_dir, f'curves_epoch{epoch+1}.png'))
                if metrics.get('confusion_matrix') is not None:
                    plot_confusion_matrix(
                        metrics['confusion_matrix'], ['0','1','2','3','4'],
                        os.path.join(self.save_dir, f'cm_epoch{epoch+1}.png')
                    )
                self._generate_gradcam(test_loader, epoch)
            
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={metrics['accuracy']:.4f}, "
                  f"AUROC={metrics['auroc']:.4f}, AUPR={metrics['aupr']:.4f}, "
                  f"QWK={metrics['qwk']:.4f} (round={metrics.get('round_qwk', 0):.4f}), "
                  f"Best={self.best_qwk:.4f}@ep{self.best_epoch+1}, PatienceCnt={self.patience_counter}/{self.patience}")
            
            with open(os.path.join(self.save_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f, indent=2)

            # A3. Early Stopping
            if self.stage == 2 and self.patience_counter >= self.patience:
                print(f"\n[EarlyStopping] QWK 连续 {self.patience} epoch 未提升，在 epoch {epoch+1} 提前停止。"
                      f" Best QWK = {self.best_qwk:.4f} @ epoch {self.best_epoch+1}")
                break
        
        # 最终报告
        print("\n训练完成，生成最终报告...")
        plot_training_curves(self.history, os.path.join(self.save_dir, 'final_curves.png'))
        final_metrics = self.validate(test_loader)
        print(f"\n最终结果: AUROC={final_metrics['auroc']:.4f}, AUPR={final_metrics['aupr']:.4f}, "
              f"Acc={final_metrics['accuracy']:.4f}, QWK={final_metrics['qwk']:.4f}")
        if final_metrics.get('confusion_matrix') is not None:
            plot_confusion_matrix(
                final_metrics['confusion_matrix'], ['0','1','2','3','4'],
                os.path.join(self.save_dir, 'final_cm.png')
            )
        self._generate_gradcam(test_loader, epochs)
        compute_model_efficiency(self.model, device=self.device)
    
    def _generate_gradcam(self, test_loader, epoch):
        try:
            images, labels = next(iter(test_loader))
            images = images[:4].to(self.device)
            labels_np = labels[:4].numpy()
            
            # RETFound ViT-L/16: 使用最后一个 Transformer block 作为目标层
            target_layer = self.model.backbone.vit.blocks[-1]
            cam_gen = GradCAM(self.model, target_layer)
            
            with torch.enable_grad():
                cams = cam_gen.generate(images.clone().requires_grad_(True))
            
            with torch.no_grad():
                preds = self.model(images).argmax(dim=1).cpu().numpy()
            
            GradCAM.visualize_cam(
                images, cams, labels_np, preds,
                os.path.join(self.save_dir, f'gradcam_epoch{epoch}.png')
            )
        except Exception as e:
            print(f"Grad-CAM生成失败: {e}")


# ------------------------------------------------------------
# 8. 数据加载
# ------------------------------------------------------------

def get_loaders(data_root, batch_size=32, img_size=224, train_dir=None, test_dir=None,
                strong_aug=False):
    """
    strong_aug=False (默认): RETFound 官方协议弱增强，Resize + HFlip + Normalize
    strong_aug=True: 启用 BenGraham + CLAHE 等强增强（可能与 MAE 预训练分布不匹配）
    """
    if strong_aug:
        train_transform = transforms.Compose([
            BenGrahamTransform(),
            CLAHETransform(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            BenGrahamTransform(),
            CLAHETransform(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # RETFound 官方协议的弱增强
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if train_dir and test_dir:
        # 官方划分模式 (IDRiD 等)
        train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
        train_set = train_dataset
        test_set = test_dataset
        targets = np.array(train_dataset.targets)
        full_dataset = train_dataset  # 用于打印类别信息
    else:
        # 随机划分模式 (APTOS 等)
        full_dataset = torchvision.datasets.ImageFolder(root=data_root)
        total = len(full_dataset)
        all_targets = np.array(full_dataset.targets)
        all_indices = np.arange(total)

        # 分层划分（保证各类别在 train/test 中分布一致）
        train_indices, test_indices = train_test_split(
            all_indices, test_size=0.3, stratify=all_targets, random_state=42
        )
        train_indices = train_indices.tolist()
        test_indices = test_indices.tolist()

        # 创建两个独立的 dataset，分别应用不同的 transform（避免数据泄露）
        train_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=test_transform)

        train_set = torch.utils.data.Subset(train_dataset, train_indices)
        test_set = torch.utils.data.Subset(test_dataset, test_indices)
        targets = np.array([train_dataset.targets[i] for i in train_indices])

    # 类别权重（基于训练集）
    weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
    sample_weights = weights[targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Loaded {len(train_set)} train, {len(test_set)} test samples")
    print(f"Classes: {full_dataset.classes}")
    print(f"Class weights: {weights}")

    return train_loader, test_loader


# ------------------------------------------------------------
# 9. 主函数
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=None, help='单目录数据集路径 (随机划分模式)')
    parser.add_argument('--train_dir', type=str, default=None, help='训练集目录 (官方划分模式)')
    parser.add_argument('--test_dir', type=str, default=None, help='测试集目录 (官方划分模式)')
    parser.add_argument('--batch_size', type=int, default=16, help='ViT-L 显存大，建议 16–24')
    parser.add_argument('--dim', type=int, default=384)
    parser.add_argument('--epochs', type=int, default=40,
                       help='V2 默认 40（配合 early stopping，原 V1 默认 100）')
    parser.add_argument('--freeze_epochs', type=int, default=5,
                       help='Stage 1 linear probing 的 epoch 数（backbone 冻结），V2 默认 5')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience（val QWK 连续不提升次数）')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                       help='EMA decay 系数，0.999 最常用')
    parser.add_argument('--disable_tta', action='store_true',
                       help='禁用 TTA（默认开启水平翻转 TTA）')
    parser.add_argument('--save_dir', type=str, default='/root/autodl-tmp/longfei/image_classification/runs/homar_retfound_v2_aptos')
    parser.add_argument('--retfound_weights', type=str, default=DEFAULT_RETFOUND_WEIGHTS,
                       help='RETFound MAE 预训练权重 .pth 路径')
    parser.add_argument('--strong_aug', action='store_true',
                       help='启用 BenGraham/CLAHE 强增强（默认关闭，与 RETFound 预训练分布对齐）')
    # 消融实验参数
    parser.add_argument('--no_routing', action='store_true', help='消融: 禁用解剖学路由，用均等权重')
    parser.add_argument('--no_hierarchy', action='store_true', help='消融: 禁用层次化Mamba')
    parser.add_argument('--no_consensus', action='store_true', help='消融: 禁用跨层一致性，用简单平均')
    parser.add_argument('--efficiency_only', action='store_true', help='只计算模型效率')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"HOMAR Training on {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 先检测类别数
    if args.train_dir:
        temp_dataset = torchvision.datasets.ImageFolder(root=args.train_dir)
    else:
        assert args.data_root, '必须指定 --data_root 或 --train_dir/--test_dir'
        temp_dataset = torchvision.datasets.ImageFolder(root=args.data_root)
    num_classes = len(temp_dataset.classes)
    print(f"Detected num_classes: {num_classes}, Classes: {temp_dataset.classes}")

    model = HOMAR(
        dim=args.dim, num_classes=num_classes, pretrained=True,
        use_routing=not args.no_routing,
        use_hierarchy=not args.no_hierarchy,
        use_consensus=not args.no_consensus,
        retfound_weights=args.retfound_weights,
    )

    # 打印消融配置
    print(f"\n消融配置: Routing={'ON' if not args.no_routing else 'OFF'}, "
          f"Hierarchy={'ON' if not args.no_hierarchy else 'OFF'}, "
          f"Consensus={'ON' if not args.no_consensus else 'OFF'}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total/1e6:.2f}M")

    if args.efficiency_only:
        compute_model_efficiency(model, device=device)
        return

    train_loader, test_loader = get_loaders(
        args.data_root, batch_size=args.batch_size,
        train_dir=args.train_dir, test_dir=args.test_dir,
        strong_aug=args.strong_aug,
    )

    trainer = HOMARTrainer(
        model, device,
        save_dir=args.save_dir,
        freeze_epochs=args.freeze_epochs,
        total_epochs=args.epochs,
        ema_decay=args.ema_decay,
        patience=args.patience,
        use_tta=not args.disable_tta,
    )
    trainer.train(train_loader, test_loader, epochs=args.epochs)

if __name__ == "__main__":
    main()