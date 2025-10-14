# customer_analysis_mvp/app.py

"""
顧客分析 MVP+ 應用程式主進入點。
"""

import streamlit as st

# 導入設定檔、服務模組、UI 模組與工具
import config
from services import llm_handler
from ui import live_view, video_view
from utils import state_manager

# --- 1. 頁面設定 (只在主程式執行一次) ---
st.set_page_config(
    page_title="顧客分析 MVP+",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 初始化 (只執行一次) ---

# 初始化 OpenAI client
# @st.cache_resource 可以快取這個物件，避免每次頁面刷新都重新建立
@st.cache_resource
def get_client():
    return llm_handler.get_openai_client(config.OPENAI_API_KEY)

client = get_client()

# 初始化 session state
state_manager.initialize_state()

# --- 3. 側邊欄 UI (全域共用) ---
with st.sidebar:
    st.header("⚙️ LLM 偏好設定")
    st.caption("調整此處設定會影響最終 AI 生成的摘要報告風格。")
    
    store_type = st.selectbox("您的店型", ["一般餐廳", "早午餐", "咖啡店", "美式速食", "火鍋店"], index=0)
    tone = st.selectbox("摘要語氣", ["專業精準", "親切口語", "營運顧問風"], index=0)
    tips_style = st.selectbox("建議風格", ["可執行優先級", "行銷洞察", "營運流程優化"], index=0)
    
    st.divider()

    st.header("📝 您的菜單")
    menu_text = st.text_area(
        "每行一項（AI 會優先從此清單辨識餐點）", 
        "草莓蛋糕\n抹茶蛋糕\n美式咖啡\n拿鐵咖啡\n總匯三明治",
        height=120
    )
    menu_items = [x.strip() for x in menu_text.splitlines() if x.strip()]
    st.caption("若不輸入，AI 會直接顯示 YOLO 偵測到的通用物件名稱。")

# 將 LLM 偏好打包成一個字典，方便傳遞給 UI 模組
llm_preferences = {
    "store_type": store_type,
    "tone": tone,
    "tips_style": tips_style
}

# --- 4. 主頁面 UI ---
st.title("📊 顧客分析 MVP+")

if not client:
    st.error("⚠️ 偵測不到 OpenAI API 金鑰！", icon="🚨")
    st.markdown(
        "請在專案根目錄下建立一個 `.env` 檔案，並在其中加入 `OPENAI_API_KEY='sk-...'`。\n\n"
        "設定完成後，請重新啟動 Streamlit。"
    )
else:
    tab_live, tab_video = st.tabs(["🟢 即時鏡頭分析", "🎞️ 影片離線分析"])

    with tab_live:
        # 將控制權交給 live_view 模組
        live_view.display(client, menu_items, llm_preferences)

    with tab_video:
        # 將控制權交給 video_view 模組
        video_view.display(client, menu_items, llm_preferences)
