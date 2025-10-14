# ui/live_view.py
"""
簡化後的即時分析頁面。
職責：建立UI元件、管理 LiveAnalyzer 引擎的生命週期、顯示分析結果。
"""
import streamlit as st
import cv2
import time
import asyncio

# 導入我們的核心分析引擎和標準數據結構
from core.live_analyzer import LiveAnalyzer
from core.types import FrameResult
# 導入 LLM 服務用於摘要功能
from services import llm_handler as llm

def display(model_pack: dict, menu_items: list, llm_preferences: dict):
    """
    主執行緒函式。
    負責繪製所有 UI 元件、管理背景執行緒的生命週期，
    以及安全地更新 st.session_state 和畫面。
    """
    lcol, rcol = st.columns([2, 1])
    
    with rcol:
        st.subheader("分析項目")
        opt_nod = st.checkbox("點頭偵測", value=True)
        opt_emote = st.checkbox("表情分類", value=True)
        # 【修改點 1】加入 UI 選項
        opt_plate = st.checkbox("餐盤殘留分析", value=True)
        
        # 將 UI 選項打包成字典，方便傳遞給引擎
        analysis_options = {
            "opt_nod": opt_nod, 
            "opt_emote": opt_emote,
            "opt_plate": opt_plate # 將選項加入字典
        }
        
        st.divider()
        st.subheader("控制")
        run_live = st.toggle("開啟鏡頭", value=False, key="live_toggle")
        fps_display = st.slider("UI 更新 FPS 上限", 5, 30, 15)
        
        st.divider()
        st.subheader("📈 即時統計")
        stat_nod = st.empty()
        stat_emotion = st.empty()
        # 【修改點 2】加入統計顯示區
        stat_leftover = st.empty()
        
        st.divider()
        if st.button("產生摘要（LLM）", use_container_width=True, disabled=not model_pack.get("client")):
            stats = {
                "nod": st.session_state.nod_count,
                "emotion": dict(st.session_state.emotion_counter),
                "leftover": dict(st.session_state.leftover_counter), # 將餐盤統計也加入摘要
            }
            with st.spinner("LLM 摘要生成中..."):
                summary = asyncio.run(llm.summarize_session(
                    stats, **llm_preferences, client=model_pack["client"]
                ))
            st.success("今日摘要")
            st.write(summary)
        
        st.caption("＊分析任務會在背景執行以保持畫面流暢。")

    with lcol:
        st.subheader("📹 即時監視畫面")
        frame_slot = st.empty()

    # --- 引擎生命週期管理 ---
    if run_live and st.session_state.analyzer is None:
        st.session_state.analyzer = LiveAnalyzer(model_pack, menu_items, analysis_options)
        st.session_state.analyzer.start()
        st.toast("分析引擎已啟動！", icon="🚀")

    if not run_live and st.session_state.analyzer is not None:
        st.session_state.analyzer.stop()
        st.session_state.analyzer = None
        st.toast("分析引擎已停止。")

    # --- 主顯示迴圈 ---
    if run_live and st.session_state.analyzer:
        while True:
            result: FrameResult = st.session_state.analyzer.get_latest_result()
            
            if result:
                # --- 由主執行緒安全地更新 session_state ---
                if result.nod_detected:
                    st.session_state.nod_count += 1
                if result.emotion_detected:
                    st.session_state.emotion_counter[result.emotion_detected] += 1
                
                # 【修改點 3】加入餐盤計數器更新邏輯
                if result.plate_leftover_detected:
                    st.session_state.leftover_counter[result.plate_leftover_detected] += 1
                
                # --- 在畫面上繪製從引擎傳來的資訊 ---
                frame = result.processed_frame
                display_info = result.display_info
                
                cv2.putText(frame, f"[Nod] {st.session_state.nod_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2)
                if display_info.get("emotion"):
                     cv2.putText(frame, f"[Emotion] {display_info['emotion']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,0,200), 2)

                # 【修改點 4】加入餐盤資訊的繪圖邏輯
                if display_info.get("plate_label"):
                    cv2.putText(frame, f"[Plate] {display_info['plate_label']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                if display_info.get("plate_circle"):
                    x, y, r = display_info["plate_circle"]
                    cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # --- 持續更新統計數據的 UI 顯示 ---
            stat_nod.metric("點頭次數", st.session_state.nod_count)
            stat_emotion.write(f"表情分布: `{dict(st.session_state.emotion_counter)}`")
            # 【修改點 5】更新餐盤統計的顯示
            stat_leftover.write(f"餐盤統計: `{dict(st.session_state.leftover_counter)}`")
            
            time.sleep(1.0 / fps_display)
    else:
        frame_slot.info("請點擊「開啟鏡頭」以開始即時分析。")

