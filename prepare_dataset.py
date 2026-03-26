# prepare_dataset.py
import os
import cv2
from preprocess import apply_cv2_enhancement

SOURCE_DIR = './lemon_dataset'
TARGET_DIR = './lemon_dataset_enhanced'
TARGET_SIZE = (128, 128)

def generate_enhanced_dataset():
    """生成增强数据集"""
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    for category in os.listdir(SOURCE_DIR):
        source_cat_dir = os.path.join(SOURCE_DIR, category)
        target_cat_dir = os.path.join(TARGET_DIR, category)

        if not os.path.isdir(source_cat_dir):
            continue
        if not os.path.exists(target_cat_dir):
            os.makedirs(target_cat_dir)

        print(f"正在处理类别: {category}...")
        for img_name in os.listdir(source_cat_dir):
            img_path = os.path.join(source_cat_dir, img_name)
            save_path = os.path.join(target_cat_dir, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # 缩放并应用 preprocess.py 中的增强算法
            img_resized = cv2.resize(img, TARGET_SIZE)
            img_enhanced = apply_cv2_enhancement(img_resized)

            # 将增强后的图片保存至新文件夹
            cv2.imwrite(save_path, img_enhanced)

    print(f"\n增强数据集已生成并保存至 {TARGET_DIR}")

if __name__ == '__main__':
    generate_enhanced_dataset()