import os
import time
import threading
import requests
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

from config import *
from utils import image_to_base64, speak_text, check_camera
from preprocess import preprocess_for_inference


class BasePage(tk.Frame):
    """底层画布类，承载全屏背景图"""

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # 使用 Canvas 替代 Frame 作为底板，实现透明/透视效果
        self.canvas = tk.Canvas(self, width=900, height=650, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        if self.controller.bg_image:
            self.canvas.create_image(0, 0, image=self.controller.bg_image, anchor="nw")


class HelpPage(BasePage):
    """帮助与系统说明页"""

    def __init__(self, parent, controller):
        super().__init__(parent, controller)

        # 悬浮说明卡片
        content_frame = tk.Frame(self.canvas, bg='white', bd=0, highlightbackground="#e0e0e0", highlightthickness=1)
        self.canvas.create_window(450, 325, window=content_frame, anchor=tk.CENTER, width=700, height=500)

        title = tk.Label(content_frame, text="系统操作说明与帮助", font=('Microsoft YaHei', 20, 'bold'), bg='white')
        title.pack(pady=20)

        btn_back = tk.Button(content_frame, text="返回检测系统", font=FONT_BTN, bg='#2196F3', fg='white',
                             width=15, cursor="hand2", relief=tk.FLAT,
                             command=lambda: controller.show_frame("DetectionPage"))
        btn_back.pack(side=tk.BOTTOM, pady=20)

        help_text = """
1. 图片导入：点击“本地图片”选择电脑中的柠檬图片进行检测。
2. 拍照识别：点击“摄像头拍照”唤起本地摄像头进行检测。
3. AI 深度评价：系统调用大模型结合画面细节生成趣味评价。
4. 语音播报：检测完成后，自动语音播报评价内容。
5. 导出日志：可将检测记录导出为 Excel 文件。

--------------------------------------------------
技术架构：TensorFlow + OpenCV + GLM-4.5V + Tkinter
                    """
        text_widget = tk.Text(content_frame, font=FONT_MAIN, bg='#f8f8f8', fg='#333333',
                              wrap=tk.WORD, bd=0, padx=20, pady=10)
        text_widget.insert(tk.END, help_text.strip())
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(padx=30, pady=5, fill=tk.BOTH, expand=True)


class DetectionPage(BasePage):
    """核心检测主页面"""

    def __init__(self, parent, controller):
        super().__init__(parent, controller)

        # 顶部导航栏
        nav_frame = tk.Frame(self.canvas, bg='#2c3e50', height=55)
        self.canvas.create_window(0, 0, window=nav_frame, anchor="nw", width=900, height=55)
        nav_frame.pack_propagate(False)

        tk.Label(nav_frame, text="双擎柠檬品质检测系统", font=FONT_TITLE, fg='white', bg='#2c3e50').pack(side=tk.LEFT,
                                                                                                         padx=20)

        tk.Button(nav_frame, text="帮助说明", font=FONT_MAIN, bg='#ecf0f1', fg='#2c3e50', relief=tk.FLAT,
                  command=lambda: controller.show_frame("HelpPage")).pack(side=tk.RIGHT, padx=10, pady=10)
        tk.Button(nav_frame, text="导出日志", font=FONT_MAIN, bg='#e67e22', fg='white', relief=tk.FLAT,
                  command=controller.export_logs).pack(side=tk.RIGHT, padx=10, pady=10)

        # 左侧：图像显示卡片
        self.left_card = tk.Frame(self.canvas, bg='#ffffff', bd=0, highlightbackground="#cccccc", highlightthickness=1)
        self.canvas.create_window(40, 80, window=self.left_card, anchor="nw", width=460, height=520)
        self.left_card.pack_propagate(False)

        self.image_label = tk.Label(self.left_card, text="请导入图片或使用摄像头", bg='#f9f9f9', font=FONT_MAIN,
                                    fg='#7f8c8d')
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # 右侧：控制与结果卡片
        self.right_card = tk.Frame(self.canvas, bg='#ffffff', bd=0, highlightbackground="#cccccc", highlightthickness=1)
        self.canvas.create_window(530, 80, window=self.right_card, anchor="nw", width=330, height=520)
        self.right_card.pack_propagate(False)

        # 操作按钮区
        btn_frame = tk.Frame(self.right_card, bg='white')
        btn_frame.pack(pady=20, fill=tk.X, padx=15)

        tk.Button(btn_frame, text="本地图片", font=FONT_BTN, width=12, relief=tk.GROOVE,
                  command=self.load_image).pack(side=tk.LEFT, padx=5)

        self.cam_btn = tk.Button(btn_frame, text="打开摄像头", font=FONT_BTN, width=12, relief=tk.GROOVE,
                                 command=self.toggle_camera)
        self.cam_btn.pack(side=tk.RIGHT, padx=5)

        self.default_btn_bg = self.cam_btn.cget('bg')
        self.default_btn_fg = self.cam_btn.cget('fg')

        tk.Button(self.right_card, text="开始双擎检测", font=('Microsoft YaHei', 13, 'bold'),
                  bg='#27ae60', fg='white', relief=tk.FLAT, command=self.run_pipeline).pack(pady=10, fill=tk.X, padx=20)

        # 结果展示区
        res_frame = tk.LabelFrame(self.right_card, text=" 检测结果 ", font=FONT_SUBTITLE, bg='white', fg='#34495e',
                                  padx=15, pady=10)
        res_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        self.cnn_res_label = tk.Label(res_frame, text="CNN 结果: 待检测", font=FONT_SUBTITLE, bg='white', fg='#95a5a6')
        self.cnn_res_label.pack(anchor='w', pady=5)

        tk.Label(res_frame, text="AI 评价:", font=FONT_MAIN, bg='white', fg='#34495e').pack(anchor='w',
                                                                                            pady=(15, 0))

        self.ai_text = tk.Text(res_frame, font=FONT_MAIN, wrap=tk.WORD, bg='#f4f6f7', bd=0, padx=10, pady=10)
        self.ai_text.pack(fill=tk.BOTH, expand=True, pady=10)

        self.current_image_path = None

        # 追踪摄像头设备和弹窗的状态
        self.cap = None
        self.cam_win = None

    def toggle_camera(self):
        """状态机逻辑：根据当前窗口状态，决定是打开摄像头还是拍照"""
        if self.cam_win is None or not self.cam_win.winfo_exists():
            self.open_camera_window()
        else:
            self.take_photo()

    def open_camera_window(self):
        """唤起摄像头窗口，并改变主界面按钮状态"""
        self.cap = check_camera()
        if self.cap is None:
            messagebox.showerror("设备错误", "未检测到可用的摄像头，或画面获取失败！请检查硬件连接。")
            return

        # 成功打开后，将主界面按钮变为红色“拍照”
        self.cam_btn.config(text="拍照", bg='#e74c3c', fg='white')

        self.cam_win = tk.Toplevel(self)
        self.cam_win.title("摄像头画面")
        self.cam_win.geometry("640x480")
        self.cam_win.resizable(False, False)

        # 绑定窗口关闭事件：如果用户直接点右上角“X”关掉弹窗，需要恢复按钮状态
        self.cam_win.protocol("WM_DELETE_WINDOW", self.close_camera_window)

        video_label = tk.Label(self.cam_win)
        video_label.pack(expand=True, fill=tk.BOTH)

        def update_frame():
            # 确保窗口和摄像头仍在工作状态
            if self.cam_win and self.cam_win.winfo_exists() and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)
                self.cam_win.after(30, update_frame)

        update_frame()

    def take_photo(self):
        """执行抓拍，并自动关闭窗口恢复状态"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(TEMP_CAPTURE_PATH, frame)
                self.set_image(TEMP_CAPTURE_PATH)

        # 拍完照自动关闭窗口并恢复原样
        self.close_camera_window()

    def close_camera_window(self):
        """资源释放与状态还原"""
        if self.cap:
            self.cap.release()
            self.cap = None

        if self.cam_win and self.cam_win.winfo_exists():
            self.cam_win.destroy()
            self.cam_win = None

        # 恢复主界面按钮的默认文字和颜色
        self.cam_btn.config(text="打开摄像头", bg=self.default_btn_bg, fg=self.default_btn_fg)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="选择检测图片", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.set_image(file_path)

    def set_image(self, path):
        self.current_image_path = path
        img = Image.open(path)
        img.thumbnail((440, 480))
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image, text="", bg='white')

        self.cnn_res_label.config(text="CNN 结果: 待检测", fg='#95a5a6')
        self.ai_text.delete(1.0, tk.END)

    def run_pipeline(self):
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先加载一张图片或拍照！")
            return
        if self.controller.model is None:
            messagebox.showerror("错误", "CNN模型未加载！")
            return

        try:
            processed_img = preprocess_for_inference(self.current_image_path, target_size=(128, 128))
            input_tensor = np.expand_dims(processed_img, axis=0)
            predictions = self.controller.model.predict(input_tensor, verbose=0)

            idx = np.argmax(predictions[0])
            conf = np.max(predictions[0])
            cls_eng = CLASS_NAMES[idx]
            cls_chs = CLASS_TRANSLATION[cls_eng]

            color = '#27ae60' if cls_eng == 'good_quality' else ('#e74c3c' if cls_eng == 'bad_quality' else '#7f8c8d')
            self.cnn_res_label.config(text=f"{cls_chs} (置信度: {conf:.2%})", fg=color)

            self.ai_text.delete(1.0, tk.END)
            self.ai_text.insert(tk.END, "正在呼叫 GLM-4.5V 分析中，请稍候...")
            self.call_llm_async(cls_chs, float(conf))

        except Exception as e:
            messagebox.showerror("CNN检测失败", str(e))

    def call_llm_async(self, cnn_result, confidence):
        def run():
            if not API_KEY:
                self.after(0, self.update_ai_result, "未配置 API_KEY，跳过 AI 分析。", cnn_result, confidence)
                return

            try:
                base64_img = image_to_base64(self.current_image_path)
                prompt = (f"观察图中的物体，本地CNN柠檬分类模型检测结果为：【{cnn_result}】，置信度：【{confidence:.2%}】。"
                          f"请结合图片画面，用不超过50个字的简短文字，给出一句专业评价或处理建议。"
                          f"注意，不要重复检测结果本身，回复使用纯文本，不要包含任何markdown标记。"
                          f"边缘情况说明：由于CNN模型是三分类模型，如果识别错误，也请如实指出。")

                payload = {
                    "model": "GLM-4.5V",
                    "messages": [
                        {"role": "user", "content": [{"type": "text", "text": prompt},
                                                     {"type": "image_url", "image_url": {"url": base64_img}}]}
                    ]
                }
                headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

                response = requests.post(API_URL, json=payload, headers=headers, timeout=15)
                if response.status_code == 200:
                    ai_reply = response.json().get('choices', [{}])[0].get('message', {}).get('content', 'AI未返回内容')
                else:
                    ai_reply = f"API请求失败: {response.status_code}"
            except Exception as e:
                ai_reply = f"大模型调用异常: {str(e)}"

            self.after(0, self.update_ai_result, ai_reply, cnn_result, confidence)

        threading.Thread(target=run, daemon=True).start()

    def update_ai_result(self, text, cnn_result, confidence):
        self.ai_text.delete(1.0, tk.END)
        self.ai_text.insert(tk.END, text)

        self.controller.detect_logs.append({
            "检测时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            "图片来源": os.path.basename(self.current_image_path),
            "分类结果": cnn_result,
            "置信度": f"{confidence:.2%}",
            "AI评价": text
        })

        if "未配置" not in text and "异常" not in text and "失败" not in text:
            speak_text(text)