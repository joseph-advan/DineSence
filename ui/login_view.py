# ui/login_view.py

import streamlit as st
import config  # 導入我們的設定檔來獲取帳號密碼

def display():
    """
    顯示登入頁面並處理驗證邏輯。
    如果登入成功，會將 st.session_state['auth'] 設為 True 並重新整理頁面。
    """
    st.markdown("<h1 style='text-align: center;'>DineSence 顧客分析平台</h1>", unsafe_allow_html=True)
    st.markdown("---")

    lcol, ccol, rcol = st.columns([1, 1.5, 1])

    with ccol:
        with st.container(border=True):
            st.markdown("<h3 style='margin-bottom:0'>🔐 請登入以繼續</h3>", unsafe_allow_html=True)
            st.caption("請輸入您的使用者名稱與密碼。")

            with st.form("login_form", clear_on_submit=False):
                username = st.text_input(
                    "使用者名稱",
                    placeholder="預設: admin",
                    key="login_username"
                )
                password = st.text_input(
                    "密碼",
                    type="password",
                    placeholder="預設: admin123",
                    key="login_password"
                )
                
                submitted = st.form_submit_button("登入", use_container_width=True, type="primary")

                if submitted:
                    # 從 config 模組讀取正確的帳號密碼
                    correct_username = config.DASH_USER
                    correct_password = config.DASH_PASS

                    if username == correct_username and password == correct_password:
                        st.session_state['auth'] = True
                        st.success("登入成功，正在載入主畫面...")
                        st.rerun()  # 觸發重新整理，app.py 會偵測到 auth 狀態改變
                    else:
                        st.error("帳號或密碼錯誤，請重新輸入。")