import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
API_URL = os.getenv("API_URL", "https://api.edgefn.net/v1/chat/completions")
API_KEY = os.getenv("API_KEY", "")

# 常量配置
MODEL_PATH = './models/lemon_cnn_model.h5'
BG_IMAGE_PATH = './background.png'
TEMP_CAPTURE_PATH = './test_images/temp_capture.jpg'

CLASS_NAMES = ['bad_quality', 'empty_background', 'good_quality']
CLASS_TRANSLATION = {
    'bad_quality': '劣质柠檬',
    'empty_background': '图中无柠檬',
    'good_quality': '优质柠檬'
}

# 全局字体配置
FONT_TITLE = ('Microsoft YaHei', 16, 'bold')
FONT_SUBTITLE = ('Microsoft YaHei', 14, 'bold')
FONT_MAIN = ('Microsoft YaHei', 11)
FONT_BTN = ('Microsoft YaHei', 12)