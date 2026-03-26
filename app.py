import os
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageTk

from config import BG_IMAGE_PATH, MODEL_PATH
from pages import DetectionPage, HelpPage

class LemonQualityApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("双擎柠檬品质分类检测系统 V2.0")
        self.geometry("900x650")
        self.resizable(False, False) # 锁定窗口大小

        self.detect_logs = []

        # 加载背景图
        self.bg_image = None
        if os.path.exists(BG_IMAGE_PATH):
            img = Image.open(BG_IMAGE_PATH).resize((900, 650))
            self.bg_image = ImageTk.PhotoImage(img)

        # 加载 CNN 模型
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("模型错误", f"无法加载CNN模型，请先训练。\n{str(e)}")
            self.model = None

        os.makedirs('./test_images', exist_ok=True)

        # 主容器
        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # 初始化所有页面
        self.frames = {}
        for F in (DetectionPage, HelpPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("DetectionPage")

    def show_frame(self, page_name):
        """切换页面"""
        frame = self.frames[page_name]
        frame.tkraise()

    def export_logs(self):
        """导出日志到 Excel"""
        if not self.detect_logs:
            messagebox.showinfo("提示", "当前没有检测记录可导出！")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")], title="保存检测日志"
        )
        if file_path:
            try:
                df = pd.DataFrame(self.detect_logs)
                df.to_excel(file_path, index=False)
                messagebox.showinfo("成功", f"日志已成功导出至:\n{file_path}")
            except Exception as e:
                messagebox.showerror("导出失败", str(e))