# config.py

"""
本檔案為專案的設定檔。
集中管理所有硬式編碼的常數、路徑與 API 金鑰，方便維護與調整。
"""

import os
from dotenv import load_dotenv

# --- 安全性設定 ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DASH_USER = os.getenv("DASH_USER", "admin")
DASH_PASS = os.getenv("DASH_PASS", "admin123")


# --- LLM 模型設定 ---
LLM_MODEL_EMOTION = "gpt-4o-mini"
LLM_MODEL_SUMMARY = "gpt-4o-mini"


# --- 電腦視覺演算法設定 ---
YOLO_MODEL_PATH = "yolov8n.pt"
FOODISH_CLASSES = {
    "cake", "pizza", "hot dog", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "bowl", "cup", "wine glass", "bottle", "spoon", "fork",
    "knife", "plate"
}

NOD_BUFFER_LEN = 36
NOD_AMP_THRESH = 0.03
NOD_COOLDOWN_SECONDS = 1.0

# --- 流程與效能控制設定 ---
EMOTE_INTERVAL_SECONDS = 1.5

# --- 新增：攝影機與緩衝區設定 ---
# 降低解析度可以大幅提升畫面流暢度
CAMERA_RESOLUTION_WIDTH = 640
CAMERA_RESOLUTION_HEIGHT = 360
# 佇列的緩衝大小，設為 1 或 2 可以確保最低的延遲
CAMERA_BUFFER_SIZE = 2