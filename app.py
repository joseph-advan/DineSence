# app.py

import streamlit as st

# 導入設定檔、服務模組、UI 模組與工具
import config
from services import llm_handler, vision_analysis as va
# --- MODIFIED: 導入所有 UI 模組，包括新的 login 和 dashboard ---
from ui import live_view, video_view, dashboard_view, login_view
from utils import state_manager

# --- 1. 頁面設定 (只應被呼叫一次) ---
st.set_page_config(
    page_title="DineSence 顧客分析平台",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 初始化與快取資源 ---
@st.cache_resource
def load_models():
    """集中載入所有昂貴的模型物件"""
    openai_client = llm_handler.get_openai_client(config.OPENAI_API_KEY)
    pose_detector = va.get_pose_detector()
    face_detector = va.get_face_detector()
    # 將 yolo 也加入快取，即使它是在 vision_analysis 中初始化
    # 這裡只是確保它被觸發載入
    va.detect_food_regions_yolo # 觸發 YOLO 模型載入
    return openai_client, pose_detector, face_detector

# 初始化 session state，確保 'auth' key 存在
state_manager.initialize_state()
client, pose_detector, face_detector = load_models()

# --- 3. 登入閘門 (Login Gate) ---
# 這是應用程式的核心流程控制
# 如果 st.session_state.auth 是 False，就只顯示登入頁面並停止執行
if not st.session_state.auth:
    login_view.display()
    st.stop()

# =====================================================================
# 只有在登入成功後 (st.session_state.auth is True)，才會執行以下的程式碼
# =====================================================================

# --- 4. 側邊欄 UI (登入後顯示) ---
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

# 將設定打包，方便傳遞給各個 view
llm_preferences = {
    "store_type": store_type,
    "tone": tone,
    "tips_style": tips_style
}
model_pack = {
    "client": client,
    "pose_detector": pose_detector,
    "face_detector": face_detector
}

# --- 5. 主頁面 UI (登入後顯示) ---
st.title("🍽️ DineSence 顧客分析平台")

if not client:
    st.error("⚠️ 偵測不到 OpenAI API 金鑰！", icon="🚨")
    st.markdown("""
        請確認您的專案根目錄下有名為 `.env` 的檔案，且內容包含：
        ```
        OPENAI_API_KEY="sk-..."
        ```
        修改後請重新整理頁面。
    """)
else:
    # --- MODIFIED: 新增第三個分頁 "本月數據儀表板" ---
    tab_live, tab_video, tab_dashboard = st.tabs([
        "🟢 即時鏡頭分析", 
        "🎞️ 影片離線分析",
        "📈 本月數據儀表板"
    ])

    with tab_live:
        live_view.display(model_pack, menu_items, llm_preferences)

    with tab_video:
        video_view.display(client, menu_items, llm_preferences)
        
    # --- MODIFIED: 在新的分頁中呼叫 dashboard_view 的 display 函式 ---
    with tab_dashboard:
        dashboard_view.display()


