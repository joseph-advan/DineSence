# ui/live_view.py
"""
【版本 3.0 - 含 Session 歷史紀錄版】
此版本在高效能的基礎上，新增了完整的 Session 紀錄功能。
- 自動偵測鏡頭開關，以定義 Session 的生命週期。
- Session 結束時，自動儲存該次的完整分析數據。
- 在 UI 上呈現所有歷史紀錄，並 Highlight 關鍵資訊。
"""
import streamlit as st
import cv2
import time
import asyncio
from collections import Counter
from datetime import datetime

# 導入我們的核心分析引擎和數據結構
from core.live_analyzer import LiveAnalyzer
from core.types import AnalysisResult
# 導入 LLM 服務
from services import llm_handler as llm

def display(model_pack: dict, menu_items: list, llm_preferences: dict):
    """
    主執行緒函式 (最終版)。
    負責繪製 UI、管理 LiveAnalyzer 生命週期，並處理 Session 歷史紀錄的邏輯與呈現。
    """
    lcol, rcol = st.columns([2, 1])
    
    with rcol:
        st.subheader("分析項目")
        opt_nod = st.checkbox("點頭偵測", value=True)
        opt_emote = st.checkbox("表情分類", value=True)
        opt_plate = st.checkbox("餐盤殘留分析", value=True)
        
        analysis_options = { "opt_nod": opt_nod, "opt_emote": opt_emote, "opt_plate": opt_plate }
        
        st.divider()
        st.subheader("控制")
        # 使用 key="live_toggle" 讓 Streamlit 追蹤此元件的狀態
        run_live = st.toggle("開啟鏡頭", value=False, key="live_toggle")
        fps_display = st.slider("UI 更新 FPS 上限", 5, 30, 20)
        
        st.divider()
        st.subheader("📈 即時統計 (本次)")
        stat_nod = st.empty()
        stat_emotion = st.empty()
        stat_leftover = st.empty()
        
        st.divider()
        
        # --- [MODIFIED] 摘要按鈕邏輯 ---
        if st.button("產生摘要（LLM）", use_container_width=True, disabled=not model_pack.get("client")):
            stats = {
                "nod": st.session_state.nod_count,
                "emotion": dict(st.session_state.emotion_counter),
                "leftover": dict(st.session_state.leftover_counter),
            }
            with st.spinner("LLM 摘要生成中..."):
                # 接收 (摘要文字, Token用量) 元組
                summary, usage = asyncio.run(llm.summarize_session(
                    stats, **llm_preferences, client=model_pack["client"]
                ))
                
                # 如果成功獲取 Token 用量，就進行累加
                if usage:
                    st.session_state.session_token_usage['prompt_tokens'] += usage.prompt_tokens
                    st.session_state.session_token_usage['completion_tokens'] += usage.completion_tokens
                    st.session_state.session_token_usage['total_tokens'] += usage.total_tokens
                
                # 將摘要結果暫存到 session_state，以便 Session 結束時儲存
                st.session_state.current_summary = summary

            st.success("今日摘要")
            # 顯示暫存的摘要
            st.write(st.session_state.get("current_summary", "尚未產生摘要。"))

    # --- [NEW] Session 開始與結束的核心邏輯 ---
    current_toggle_state = run_live
    last_toggle_state = st.session_state.live_toggle_last_state

    # 偵測到「開啟」事件
    if current_toggle_state and not last_toggle_state:
        st.toast("新的 Session 已開始！", icon="▶️")
        # 重設所有計數器
        st.session_state.nod_count = 0
        st.session_state.emotion_counter = Counter()
        st.session_state.leftover_counter = Counter()
        st.session_state.session_token_usage = Counter()
        st.session_state.current_summary = "" # 清空上次的摘要
        # 記錄開始時間
        st.session_state.session_start_time = datetime.now()

    # 偵測到「關閉」事件
    if not current_toggle_state and last_toggle_state:
        st.toast("Session 已結束並儲存紀錄。", icon="💾")
        end_time = datetime.now()
        start_time = st.session_state.session_start_time
        duration = (end_time - start_time).total_seconds() if start_time else 0
        
        # 打包所有數據
        session_data = {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "N/A",
            "duration_seconds": int(duration),
            "nod_count": st.session_state.nod_count,
            "emotion_counter": dict(st.session_state.emotion_counter),
            "leftover_counter": dict(st.session_state.leftover_counter),
            "token_usage": dict(st.session_state.session_token_usage),
            "summary": st.session_state.get("current_summary", "無摘要")
        }
        # 將打包好的數據存入歷史紀錄列表
        st.session_state.session_history.insert(0, session_data) # insert(0, ..) 讓最新的在最前面

    # 更新上一次的狀態，為下一次偵測做準備
    st.session_state.live_toggle_last_state = current_toggle_state

    # --- 引擎生命週期管理 (不變) ---
    if run_live and st.session_state.analyzer is None:
        st.session_state.analyzer = LiveAnalyzer(model_pack, menu_items, analysis_options)
        st.session_state.analyzer.start()
    if not run_live and st.session_state.analyzer is not None:
        st.session_state.analyzer.stop()
        st.session_state.analyzer = None

    # --- 主顯示迴圈 (不變) ---
    with lcol:
        st.subheader("📹 即時監視畫面")
        frame_slot = st.empty()

    latest_analysis_data = AnalysisResult()
    if run_live and st.session_state.analyzer:
        while True:
            frame = st.session_state.analyzer.get_latest_frame()
            if frame is None:
                time.sleep(1.0 / fps_display)
                continue

            analysis_result = st.session_state.analyzer.get_latest_analysis_result()
            if analysis_result:
                latest_analysis_data = analysis_result
                if analysis_result.nod_event: st.session_state.nod_count += 1
                if analysis_result.emotion_event: st.session_state.emotion_counter[analysis_result.emotion_event] += 1
                if analysis_result.plate_event: st.session_state.leftover_counter[analysis_result.plate_event] += 1
            
            display_info = latest_analysis_data.display_info
            cv2.putText(frame, f"[Nod] {st.session_state.nod_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2)
            emotion_to_show = latest_analysis_data.emotion_event or "N/A"
            cv2.putText(frame, f"[Emotion] {emotion_to_show}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,0,200), 2)
            if display_info.get("plate_label"): cv2.putText(frame, f"[Plate] {display_info['plate_label']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            if display_info.get("plate_circle"):
                x, y, r = display_info["plate_circle"]
                cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
            
            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            stat_nod.metric("點頭次數", st.session_state.nod_count)
            stat_emotion.write(f"表情分布: `{dict(st.session_state.emotion_counter)}`")
            stat_leftover.write(f"餐盤統計: `{dict(st.session_state.leftover_counter)}`")
            
            time.sleep(1.0 / fps_display)
    else:
        frame_slot.info("請點擊「開啟鏡頭」以開始即時分析。")

    # --- [NEW] 歷史紀錄 UI 呈現 ---
    with rcol:
        st.divider()
        st.subheader("📜 本次運行歷史紀錄")
        if not st.session_state.session_history:
            st.caption("目前尚無歷史紀錄。關閉鏡頭後將會儲存一筆紀錄。")

        # 遍歷所有歷史紀錄 (最新的在最上面)
        for i, session_data in enumerate(st.session_state.session_history):
            # 預設只展開最新的一筆
            expander = st.expander(
                f"**Session @ {session_data['start_time']}** (持續 {session_data['duration_seconds']} 秒)",
                expanded=(i == 0)
            )
            with expander:
                # 找出出現最多次的情緒
                emotions = session_data['emotion_counter']
                if emotions:
                    max_emotion = max(emotions, key=emotions.get)
                    st.write(f"**主要情緒: {max_emotion}**")
                    
                    # 顯示所有情緒並 Highlight 最高者
                    for emotion, count in emotions.items():
                        if emotion == max_emotion:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;**- {emotion}: {count} 次 (最高)**")
                        else:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;- {emotion}: {count} 次")
                else:
                    st.write("**主要情緒: N/A**")

                st.metric("點頭總次數", session_data['nod_count'])
                
                st.write("**餐盤統計:**")
                if session_data['leftover_counter']:
                    for status, count in session_data['leftover_counter'].items():
                        st.markdown(f"&nbsp;&nbsp;&nbsp;- {status}: {count} 次")
                else:
                    st.caption("無餐盤紀錄")

                st.write("**Token 用量:**")
                tokens = session_data['token_usage']
                st.markdown(f"&nbsp;&nbsp;&nbsp;- 總計: `{tokens.get('total_tokens', 0)}`")
                st.markdown(f"&nbsp;&nbsp;&nbsp;- 輸入: `{tokens.get('prompt_tokens', 0)}`")
                st.markdown(f"&nbsp;&nbsp;&nbsp;- 輸出: `{tokens.get('completion_tokens', 0)}`")
                
                st.write("**AI 摘要快照:**")
                st.info(session_data['summary'] or "本次 Session 未產生摘要。")
