# customer_analysis_mvp/ui/video_view.py

import streamlit as st
import os
import time
import cv2
from collections import Counter

# 導入我們的服務模組
from services import vision_analysis as va
from services import llm_handler as llm

def display(client, menu_items, llm_preferences):
    """
    顯示「影片離線分析」分頁的所有 UI 元素與分析邏輯。
    
    Args:
        client (OpenAI): 初始化後的 OpenAI 客戶端物件。
        menu_items (list): 從側邊欄獲取的菜單項目列表。
        llm_preferences (dict): 包含 store_type, tone, tips_style 的字典。
    """
    st.subheader("🎞️ 上傳影片進行離線分析與摘要")
    up = st.file_uploader("支援 .mp4 / .avi 格式", type=["mp4", "avi"])

    col1, col2, col3 = st.columns(3)
    with col1:
        sample_sec = st.number_input("抽樣間隔（秒）", min_value=1, max_value=30, value=5, step=1, help="每隔幾秒分析一張畫面")
    with col2:
        do_plate_v = st.checkbox("分析餐盤殘留", value=True)
    with col3:
        do_emote_v = st.checkbox("分析表情", value=True)

    do_food_v = st.checkbox("分析食物/飲品（YOLO→菜單分類）", value=True)

    if up is not None:
        # 為了處理上傳的檔案，先將其寫入暫存檔
        tmp_path = os.path.join(".", f"tmp_{up.name}")
        with open(tmp_path, "wb") as f:
            f.write(up.getbuffer())

        st.video(tmp_path)
        
        if st.button("🚀 開始分析影片", type="primary", use_container_width=True, disabled=not client):
            progress_bar = st.progress(0, text="準備開始分析...")
            
            # 初始化偵測器與計數器
            pose_detector = va.get_pose_detector()
            face_detector = va.get_face_detector()
            nod_detector = va.NodDetector()
            
            leftover_counter = Counter()
            emotion_counter = Counter()
            food_counter = Counter()
            nod_total = 0
            timeline = []

            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, int(fps * sample_sec))

            try:
                for fr in range(0, total_frames, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                    ok, frame = cap.read()
                    if not ok:
                        break

                    progress_text = f"分析進度：{fr}/{total_frames} frames..."
                    progress_bar.progress(fr / total_frames, text=progress_text)

                    timestamp_s = int(fr / fps)
                    mm_ss = f"{timestamp_s//60:02d}:{timestamp_s%60:02d}"

                    # --- 執行各項分析 ---
                    plate_label, food_label, emo_label, nod_flag = "-", "-", "-", 0
                    
                    if do_plate_v:
                        label, _, _ = va.estimate_plate_leftover(frame)
                        if label in ["剩餘50%以上", "無剩餘"]:
                            leftover_counter[label] += 1
                            plate_label = label

                    if do_food_v:
                        regions = va.detect_food_regions_yolo(frame, conf=0.3, min_area_ratio=0.01)[:1]
                        if regions:
                            if menu_items and client:
                                x1, y1, x2, y2 = regions[0]["xyxy"]
                                crop = frame[y1:y2, x1:x2].copy()
                                res_food = llm.gpt_food_from_menu(crop, menu_items, client)
                                food_label = res_food["label"]
                            else:
                                food_label = regions[0]["label"]
                            food_counter[food_label] += 1

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = pose_detector.process(rgb)
                    if res.pose_landmarks:
                        lm = res.pose_landmarks.landmark
                        nose_y = lm[0].y
                        ref_y = (lm[11].y + lm[12].y) / 2 if len(lm) > 12 else 0.5
                        if nod_detector.update_and_check(nose_y, ref_y):
                            nod_total += 1
                            nod_flag = 1
                    
                    if do_emote_v:
                        face_crop = va.crop_face_with_mediapipe(frame, face_detector)
                        emo = llm.gpt_image_classify_3cls(face_crop, client)
                        if emo in ["喜歡", "中性", "討厭"]:
                            emotion_counter[emo] += 1
                            emo_label = emo
                    
                    timeline.append({
                        "t": mm_ss, "leftover": plate_label, "food": food_label, 
                        "nod": "✔" if nod_flag else " ", "emotion": emo_label
                    })

                progress_bar.progress(1.0, text="分析完成！正在產生摘要...")

                stats = {
                    "leftover": dict(leftover_counter), "food": dict(food_counter),
                    "nod": nod_total, "emotion": dict(emotion_counter),
                    "timeline": timeline
                }
                
                with st.expander("查看原始統計數據 (JSON)", expanded=False):
                    st.json(stats)
                
                summary = llm.summarize_session(stats, **llm_preferences, client=client)
                st.success("🎯 影片分析摘要")
                st.markdown(summary)

            finally:
                cap.release()
                os.remove(tmp_path) # 刪除暫存檔
                progress_bar.empty() # 清理進度條