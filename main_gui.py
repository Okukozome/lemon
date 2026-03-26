import sys
import os
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from preprocess import preprocess_for_inference

# 常量配置
MODEL_PATH = './models/lemon_cnn_model.h5'
CLASS_NAMES = ['bad_quality', 'empty_background', 'good_quality']
CLASS_TRANSLATION = {
    'bad_quality': '劣质柠檬',
    'empty_background': '空背景',
    'good_quality': '优质柠檬'
}


class LemonQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("柠檬品质分类系统")
        self.root.geometry("800x550")

        # 加载模型
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("错误", f"无法加载模型，请先运行 train.py 训练模型。\n{str(e)}")
            self.root.destroy()
            return

        self.current_image_path = None
        self.setup_ui()

    def setup_ui(self):
        """配置界面布局：左侧大图显示，右侧控制与结果"""
        # 左侧图像显示区域
        self.left_frame = tk.Frame(self.root, width=500, height=500, bg='white', relief=tk.SUNKEN, bd=2)
        self.left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)
        self.left_frame.pack_propagate(False)  # 防止子组件改变 frame 大小

        self.image_label = tk.Label(self.left_frame, text="暂无图片", bg='white', font=('Arial', 14))
        self.image_label.pack(expand=True)

        # 右侧控制与结果区域
        self.right_frame = tk.Frame(self.root, width=250)
        self.right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.Y)

        # 标题栏
        title_lbl = tk.Label(self.right_frame, text="控制面板", font=('Arial', 18, 'bold'))
        title_lbl.pack(pady=20)

        # 按钮
        btn_load = tk.Button(self.right_frame, text="加载图片", font=('Arial', 12), width=15, command=self.load_image)
        btn_load.pack(pady=10)

        btn_predict = tk.Button(self.right_frame, text="检测品质", font=('Arial', 12), width=15, bg='#4CAF50',
                                fg='white', command=self.predict_quality)
        btn_predict.pack(pady=10)

        btn_exit = tk.Button(self.right_frame, text="退出系统", font=('Arial', 12), width=15, command=self.root.quit)
        btn_exit.pack(pady=50)

        # 结果显示框
        res_title = tk.Label(self.right_frame, text="检测结果:", font=('Arial', 14))
        res_title.pack(anchor='w')

        self.result_label = tk.Label(self.right_frame, text="等待检测...", font=('Arial', 16, 'bold'), fg='blue',
                                     wraplength=200)
        self.result_label.pack(pady=10)

    def load_image(self):
        """打开文件对话框，选择并显示图片"""
        file_path = filedialog.askopenfilename(
            title="选择要检测的图片",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.current_image_path = file_path
            # 使用 PIL 显示图片
            img = Image.open(file_path)
            img.thumbnail((450, 450))  # 缩放以适应显示框
            self.tk_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_image, text="")
            self.result_label.config(text="等待检测...", fg='blue')

    def predict_quality(self):
        """执行 OpenCV 预处理和 CNN 推理"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先加载一张图片！")
            return

        try:
            # 调用自定义的 OpenCV 预处理逻辑
            processed_img = preprocess_for_inference(self.current_image_path, target_size=(128, 128))

            # 增加 Batch 维度 (1, 128, 128, 3)
            input_tensor = np.expand_dims(processed_img, axis=0)

            # 模型预测
            predictions = self.model.predict(input_tensor, verbose=0)
            predicted_class_index = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # 解析结果并显示
            class_name_eng = CLASS_NAMES[predicted_class_index]
            class_name_chs = CLASS_TRANSLATION[class_name_eng]

            # 根据结果改变文字颜色
            color = 'green' if class_name_eng == 'good_quality' else (
                'red' if class_name_eng == 'bad_quality' else 'gray')

            result_text = f"{class_name_chs}\n置信度: {confidence:.2%}"
            self.result_label.config(text=result_text, fg=color)

        except Exception as e:
            messagebox.showerror("预测失败", f"处理图像时发生错误:\n{str(e)}")


if __name__ == '__main__':
    root = tk.Tk()
    app = LemonQualityApp(root)
    root.mainloop()