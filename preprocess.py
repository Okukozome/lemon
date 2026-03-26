# preprocess.py
import cv2
import numpy as np


def apply_cv2_enhancement(img_bgr):
    """OpenCV 增强预处理"""
    # 转换到 LAB 空间并应用 CLAHE
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # 高斯模糊去噪
    img_blurred = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
    return img_blurred


def preprocess_for_inference(image_path, target_size=(128, 128)):
    """供 GUI 调用，输出未归一化的 RGB 数组 (0-255)"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 缩放
    img_resized = cv2.resize(img, target_size)
    # 调用 OpenCV 增强算法
    img_enhanced = apply_cv2_enhancement(img_resized)
    # 转换为模型期望的 RGB 格式
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)

    return img_rgb.astype(np.float32)