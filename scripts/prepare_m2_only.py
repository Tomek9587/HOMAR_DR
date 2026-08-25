"""
为实验 B (Messidor-2 独立训练独立测试) 准备独立 ImageFolder 目录。
从 messidor/messidor_data.csv 读取 adjudicated_gradable==1 且 dr_grade 非 NaN 的样本,
在 messidor/IMG2/ 下找对应图像, 以 symlink 形式组织到 colored_images_messidor/{class}/m2_<origname>。
类目录命名与 APTOS 保持一致: 0_No_DR, 1_Mild, 2_Moderate, 3_Severe, 4_Proliferate_DR。
"""

import os
import csv

ROOT = '/root/autodl-tmp/longfei'
CSV_PATH = os.path.join(ROOT, 'messidor', 'messidor_data.csv')
IMG_DIR = os.path.join(ROOT, 'messidor', 'IMG2')
OUT_DIR = os.path.join(ROOT, 'colored_images_messidor')

CLASS_NAMES = {0: '0_No_DR', 1: '1_Mild', 2: '2_Moderate', 3: '3_Severe', 4: '4_Proliferate_DR'}


def main():
    # 准备输出目录
    for name in CLASS_NAMES.values():
        os.makedirs(os.path.join(OUT_DIR, name), exist_ok=True)

    # 建立 IMG_DIR 下文件名的 case-insensitive 索引
    real_files = os.listdir(IMG_DIR)
    lower_to_real = {n.lower(): n for n in real_files}

    total, kept, missing, ungradable, nan_cnt = 0, 0, 0, 0, 0
    class_count = {k: 0 for k in CLASS_NAMES}

    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            name = row['image_id'].strip()
            grade_str = row['adjudicated_dr_grade'].strip()
            gradable_str = row['adjudicated_gradable'].strip()

            if gradable_str != '1':
                ungradable += 1
                continue
            if grade_str == '' or grade_str.lower() == 'nan':
                nan_cnt += 1
                continue
            grade = int(float(grade_str))
            if grade not in CLASS_NAMES:
                continue

            real = lower_to_real.get(name.lower())
            if real is None:
                missing += 1
                continue

            src = os.path.join(IMG_DIR, real)
            dst = os.path.join(OUT_DIR, CLASS_NAMES[grade], f'm2_{real}')
            if not os.path.exists(dst):
                os.symlink(src, dst)
            kept += 1
            class_count[grade] += 1

    print(f"CSV 行数: {total}")
    print(f"保留: {kept}, 缺图: {missing}, 不可评估: {ungradable}, NaN 分级: {nan_cnt}")
    for k, n in CLASS_NAMES.items():
        print(f"  {n}: {class_count[k]}")
    print(f"输出目录: {OUT_DIR}")


if __name__ == '__main__':
    main()
