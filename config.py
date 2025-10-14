# customer_analysis_mvp/config.py

"""
本檔案為專案的設定檔。
集中管理所有硬式編碼的常數、路徑與 API 金鑰，方便維護與調整。
"""

import os
from dotenv import load_dotenv

# --- 安全性設定 ---
# 建議您建立一個名為 .env 的檔案在專案根目錄
# 內容為： OPENAI_API_KEY="sk-..."
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- LLM 模型設定 ---
# 將模型名稱集中管理，未來若要升級或更換模型 (例如 gpt-4o)，只需修改此處。
LLM_MODEL_EMOTION = "gpt-4o-mini"
LLM_MODEL_SUMMARY = "gpt-4o-mini"


# --- 電腦視覺演算法設定 ---

# YOLO 模型相關
YOLO_MODEL_PATH = "yolov8n.pt"
FOODISH_CLASSES = {
    "cake", "pizza", "hot dog", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "bowl", "cup", "wine glass", "bottle", "spoon", "fork",
    "knife", "plate"
}

# 點頭偵測 (NodDetector) 相關參數
NOD_BUFFER_LEN = 48           # 點頭偵測的歷史軌跡緩衝區長度
NOD_AMP_THRESH = 0.03         # 點頭動作的振幅閾值
NOD_COOLDOWN_SECONDS = 1.0    # 偵測到一次點頭後的冷卻時間

# --- 流程控制設定 ---

# 表情分析的節流間隔（秒），避免 API 請求過於頻繁
# 在多執行緒環境下，稍微拉長此間隔有助於穩定性和成本控制
EMOTE_INTERVAL_SECONDS = 1.5

