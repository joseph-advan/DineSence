# core/types.py
"""
本檔案定義了整個專案中可以共用的數據結構。
使用 dataclasses 可以讓結構更清晰，並提供自動的初始化等方法，
這有助於減少因為打錯字典鍵 (key) 而造成的錯誤。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np

@dataclass
class FrameResult:
    """
    用於存放單一畫格分析結果的數據結構。
    這個物件會在背景執行緒和主執行緒之間傳遞，作為標準的溝通格式。
    """
    # 經過處理後要顯示在 UI 上的畫面
    processed_frame: np.ndarray
    
    # --- 分析出的「事件」(Event) ---
    # 這些是布林值或字串，用於觸發主執行緒中的計數器更新
    nod_detected: bool = False
    emotion_detected: str = ""
    plate_leftover_detected: str = ""
    
    # --- 顯示在畫面上的「資訊」(Info) ---
    # 這些是純粹用來顯示在畫面上的文字，不一定會觸發狀態更新
    display_info: Dict = field(default_factory=dict)

