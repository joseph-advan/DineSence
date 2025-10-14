# customer_analysis_mvp/utils/state_manager.py

"""
本模組專門用於管理 Streamlit 的 session_state。
"""

import streamlit as st
from collections import Counter, deque

def initialize_state():
    """
    初始化應用程式所有分頁都會用到的 session_state。
    此函式應在主程式 app.py 的最開始被呼叫一次。
    """
    states = {
        "nod_count": 0,
        "emotion_counter": Counter(),
        "leftover_counter": Counter(),
        "food_hist": deque(maxlen=5), # 用於即時模式下的食物辨識多幀投票
        "food_last_ts": 0.0,
    }

    for key, default_value in states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
