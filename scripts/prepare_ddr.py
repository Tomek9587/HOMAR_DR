"""
DDR 数据集 ImageFolder 化 (用 symlink, 不占硬盘)

输入:
  /root/autodl-tmp/longfei/DDR/extracted/DDR-dataset/DR_grading/
    train/*.jpg, train.txt  (6835 行, 6 类: 0-5, class 5=ungradable)
    valid/*.jpg, valid.txt  (2733 行)
    test/*.jpg,  test.txt   (4105 行)

输出:
  /root/autodl-tmp/longfei/colored_images_ddr/
    train/{0_No_DR,1_Mild,2_Moderate,3_Severe,4_Proliferate_DR}/
    test/ 同上
  过滤掉 class 5 (ungradable)
  文件名不加前缀 (训练脚本靠区间推断 source)

本脚本仅创建 symlink, 可反复执行。
"""

import os

ROOT = '/root/autodl-tmp/longfei/DDR/extracted/DDR-dataset/DR_grading'
OUT = '/root/autodl-tmp/longfei/colored_images_ddr'

CLASS_NAMES = {
    0: '0_No_DR', 1: '1_Mild', 2: '2_Moderate',
    3: '3_Severe', 4: '4_Proliferate_DR',
}


def build_split(split_name, txt_name, out_split):
    src_img_dir = os.path.join(ROOT, split_name)
    txt_path = os.path.join(ROOT, txt_name)
    for cname in CLASS_NAMES.values():
        os.makedirs(os.path.join(out_split, cname), exist_ok=True)

    total, kept, skip5, missing = 0, 0, 0, 0
    cnt = {k: 0 for k in CLASS_NAMES}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            parts = line.split()
            if len(parts) < 2:
                continue
            name, grade_s = parts[0], parts[1]
            try:
                grade = int(grade_s)
            except ValueError:
                continue
            if grade == 5:
                skip5 += 1
                continue
            if grade not in CLASS_NAMES:
                continue
            src = os.path.join(src_img_dir, name)
            if not os.path.exists(src):
                missing += 1
                continue
            dst = os.path.join(out_split, CLASS_NAMES[grade], name)
            if not os.path.exists(dst):
                os.symlink(src, dst)
            kept += 1
            cnt[grade] += 1

    print(f"[{split_name}] total={total}, kept={kept}, skip_class5={skip5}, missing={missing}")
    for k, n in CLASS_NAMES.items():
        print(f"  {n}: {cnt[k]}")


def main():
    os.makedirs(OUT, exist_ok=True)
    build_split('train', 'train.txt', os.path.join(OUT, 'train'))
    build_split('test',  'test.txt',  os.path.join(OUT, 'test'))
    # valid 暂不使用 (项目评估不使用验证集)
    print(f"\n输出目录: {OUT}")


if __name__ == '__main__':
    main()
