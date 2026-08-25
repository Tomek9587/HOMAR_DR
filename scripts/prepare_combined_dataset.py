"""
合并 APTOS + Messidor-2 为统一的 ImageFolder 结构（使用 symlink 节省空间）

输出: /root/autodl-tmp/longfei/colored_images_combined/
    0_No_DR/
    1_Mild/
    2_Moderate/
    3_Severe/
    4_Proliferate_DR/
"""
import os
import pandas as pd
from pathlib import Path

APTOS_ROOT = Path('/root/autodl-tmp/longfei/colored_images')
MESSIDOR_IMG = Path('/root/autodl-tmp/longfei/messidor/IMG2')
MESSIDOR_CSV = Path('/root/autodl-tmp/longfei/messidor/messidor_data.csv')
OUT_ROOT = Path('/root/autodl-tmp/longfei/colored_images_combined')

CLASS_NAMES = {
    0: '0_No_DR',
    1: '1_Mild',
    2: '2_Moderate', 
    3: '3_Severe',
    4: '4_Proliferate_DR',
}


def symlink_force(src: Path, dst: Path):
    """强制创建 symlink（若已存在则覆盖）"""
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def main():
    # 1. 创建输出目录
    for cls_name in CLASS_NAMES.values():
        (OUT_ROOT / cls_name).mkdir(parents=True, exist_ok=True)
    print(f'[1/4] 创建输出目录: {OUT_ROOT}')

    # 2. APTOS: 对每个类别下的文件创建 symlink
    print('[2/4] 链接 APTOS 图像...')
    aptos_count = {cls: 0 for cls in CLASS_NAMES.values()}
    for cls_dir in APTOS_ROOT.iterdir():
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        for f in cls_dir.iterdir():
            if not f.is_file():
                continue
            dst = OUT_ROOT / cls_name / f'aptos_{f.name}'
            symlink_force(f.resolve(), dst)
            aptos_count[cls_name] += 1
    for k, v in aptos_count.items():
        print(f'    {k}: {v}')

    # 3. Messidor-2: 按 CSV 标签链接
    print('[3/4] 链接 Messidor-2 图像...')
    df = pd.read_csv(MESSIDOR_CSV)
    # 过滤 gradable=1 且 dr_grade 非 NaN
    df = df[(df['adjudicated_gradable'] == 1) & df['adjudicated_dr_grade'].notna()]
    print(f'    可用样本: {len(df)}')

    # 建立 ZIP 文件名 (大小写无关) → 实际文件路径 的映射
    messidor_files = {f.name.lower(): f for f in MESSIDOR_IMG.iterdir() if f.is_file()}
    print(f'    Messidor 实际图像文件数: {len(messidor_files)}')

    messidor_count = {cls: 0 for cls in CLASS_NAMES.values()}
    missing = 0
    for _, row in df.iterrows():
        image_id = row['image_id']
        grade = int(row['adjudicated_dr_grade'])
        cls_name = CLASS_NAMES[grade]

        src = messidor_files.get(image_id.lower())
        if src is None:
            missing += 1
            continue
        dst = OUT_ROOT / cls_name / f'm2_{src.name}'
        symlink_force(src.resolve(), dst)
        messidor_count[cls_name] += 1

    for k, v in messidor_count.items():
        print(f'    {k}: {v}')
    if missing > 0:
        print(f'    [WARN] 缺失 {missing} 张图像（CSV 有但文件找不到）')

    # 4. 验证
    print('[4/4] 最终汇总:')
    total = 0
    for cls_name in CLASS_NAMES.values():
        n = len(list((OUT_ROOT / cls_name).iterdir()))
        print(f'    {cls_name}: {n}  (APTOS {aptos_count[cls_name]} + M2 {messidor_count[cls_name]})')
        total += n
    print(f'    合计: {total}')
    expected = 3662 + 1744
    if total == expected:
        print(f'[OK] 数量匹配预期 {expected}')
    else:
        print(f'[WARN] 预期 {expected}, 实际 {total}')


if __name__ == '__main__':
    main()
