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
# 這樣可以避免將您的 API 金鑰直接寫在程式碼中。
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --- 模型與演算法設定 ---

# YOLO 模型路徑與相關類別
YOLO_MODEL_PATH = "yolov8n.pt"  # 如果您有自訂模型，可在此修改路徑
FOODISH_CLASSES = {
    "cake", "pizza", "hot dog", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "bowl", "cup", "wine glass", "bottle", "spoon", "fork",
    "knife", "plate"
}

# 飲品相關關鍵字 (用於 LLM 分類的佐證)
BEVERAGE_KEYWORDS = ["咖啡", "拿鐵", "美式", "卡布奇諾", "奶茶", "茶", "飲", "可樂", "果汁"]

# 點頭偵測演算法參數
NOD_BUFFER_LEN = 48           # 點頭偵測的歷史軌跡緩衝區長度
NOD_AMP_THRESH = 0.03         # 點頭動作的振幅閾值
NOD_COOLDOWN_SECONDS = 1.0    # 點頭偵測的冷卻時間

# --- UI 與流程控制設定 ---

# 表情分析的節流間隔（秒），避免 API 請求過於頻繁
EMOTE_INTERVAL_SECONDS = 1.0
