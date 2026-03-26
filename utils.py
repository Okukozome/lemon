import base64
import threading
import pyttsx3
import cv2
import os
import numpy as np
import time

def image_to_base64(image_path):
    """将图片转换为 Base64 编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

def speak_text(text):
    """异步语音播报"""
    def run_tts():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"语音播报失败: {e}")

    threading.Thread(target=run_tts, daemon=True).start()


def check_camera():
    """检测摄像头是否可用的算法"""
    if os.name == 'nt':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    if not cap or not cap.isOpened():
        return None
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return None
    std_dev = np.std(frame)
    if std_dev < 5.0:
        cap.release()
        return None
    diff_x = np.abs(np.diff(frame.astype(np.int16), axis=1))
    mean_diff = np.mean(diff_x)
    if mean_diff > 50:
        cap.release()
        return None

    return cap