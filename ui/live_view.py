# customer_analysis_mvp/ui/live_view.py

import streamlit as st
import cv2
import time
from collections import Counter
import numpy as np

# 導入我們拆分出去的服務模組，並使用縮寫以方便呼叫
from services import vision_analysis as va
from services import llm_handler as llm

# 導入設定檔中的常數
from config import EMOTE_INTERVAL_SECONDS, BEVERAGE_KEYWORDS

def display(client, menu_items, llm_preferences):
    """
    顯示「即時鏡頭」分頁的所有 UI 元素與主迴圈邏輯。
    
    Args:
        client (OpenAI): 初始化後的 OpenAI 客戶端物件。
        menu_items (list): 從側邊欄獲取的菜單項目列表。
        llm_preferences (dict): 包含 store_type, tone, tips_style 的字典。
    """
    lcol, rcol = st.columns([2, 1])

    with rcol:
        st.subheader("分析項目")
        opt_plate = st.checkbox("餐盤殘留（50% / 無）", value=True)
        opt_nod = st.checkbox("點頭偵測（好吃點頭）", value=True)
        opt_emote = st.checkbox("表情分類（喜歡/中性/討厭）", value=True)
        st.divider()
        st.subheader("食物/飲品偵測 → 再分類")
        use_food_detection = st.checkbox("啟用 YOLO 偵測並逐框分類", value=True, help="建議開啟以獲得更準確的食物辨識")
        max_food_boxes = st.slider("最多顯示食物框數", 1, 5, 3)

        st.divider()
        st.subheader("控制")
        run_live = st.toggle("開啟鏡頭", value=False)
        fps_display = st.slider("顯示 FPS 上限", 5, 30, 15)

        st.divider()
        st.subheader("📈 即時統計")
        stat_leftover = st.empty()
        stat_nod = st.empty()
        stat_emotion = st.empty()

        st.divider()
        if st.button("產生摘要（LLM）", use_container_width=True, disabled=not client):
            stats = {
                "leftover": dict(st.session_state.leftover_counter),
                "nod": st.session_state.nod_count,
                "emotion": dict(st.session_state.emotion_counter),
            }
            with st.spinner("LLM 生成摘要中..."):
                # 將 llm_preferences 字典解包傳入
                summary = llm.summarize_session(stats, **llm_preferences, client=client)
            st.success("今日摘要")
            st.write(summary)
        
        st.caption("＊表情分類每秒最多請求一次，以節省 Token。")

    with lcol:
        st.subheader("📹 即時監視畫面")
        frame_slot = st.empty()

        if run_live:
            # 從 vision_analysis 模組獲取初始化後的偵測器
            pose_detector = va.get_pose_detector()
            face_detector = va.get_face_detector()
            nod_detector = va.NodDetector()
            last_emote_ts = 0.0

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("無法開啟攝影機。請檢查權限或是否有其他程式正在使用。")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            while run_live: # 當 toggle 開啟時，迴圈會持續
                ok, frame = cap.read()
                if not ok:
                    st.warning("讀取鏡頭失敗。")
                    break

                # A) 餐盤殘留
                if opt_plate:
                    label, _, circle = va.estimate_plate_leftover(frame)
                    if circle:
                        x, y, r = circle
                        cv2.circle(frame, (x, y), r, (0, 255, 255), 2)
                    cv2.putText(frame, f"[Plate] {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
                    if label in ["剩餘50%以上", "無剩餘"]:
                        st.session_state.leftover_counter[label] += 1

                # B) 食物/飲品偵測
                if use_food_detection:
                    regions = va.detect_food_regions_yolo(frame, conf=0.3)[:max_food_boxes]
                    cup_flag = va.has_big_cup(frame)

                    for reg in regions:
                        x1, y1, x2, y2 = reg["xyxy"]
                        crop = frame[y1:y2, x1:x2].copy()
                        
                        label_food = reg["label"]
                        confv = reg["conf"]

                        if menu_items and client:
                            res_food = llm.gpt_food_from_menu(crop, menu_items, client)
                            label_food = res_food["label"]
                            confv = res_food["confidence"]
                            # 杯子佐證邏輯
                            if (not cup_flag) and any(k in label_food for k in BEVERAGE_KEYWORDS) and confv < 0.7:
                                confv *= 0.5
                            # 更新投票池
                            now_ts = time.time()
                            if (now_ts - st.session_state.food_last_ts) > 1.5:
                                st.session_state.food_hist.append({"label": label_food, "confidence": confv})
                                st.session_state.food_last_ts = now_ts
                        
                        color = (50, 180, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{label_food} ({confv:.2f})", (x1, max(y1-6, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # 多幀投票結果
                    if st.session_state.food_hist:
                        cnt = Counter(x["label"] for x in st.session_state.food_hist)
                        if cnt:
                            top_label, votes = cnt.most_common(1)[0]
                            avg_conf = np.mean([x["confidence"] for x in st.session_state.food_hist if x["label"] == top_label])
                            if votes >= 3 and avg_conf >= 0.45:
                                cv2.putText(frame, f"[Food] {top_label} ({avg_conf:.2f})", (20, 160),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,180,255), 2)

                # C) 點頭偵測
                if opt_nod:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = pose_detector.process(rgb)
                    if res.pose_landmarks:
                        lm = res.pose_landmarks.landmark
                        nose_y = lm[0].y
                        ref_y = (lm[11].y + lm[12].y) / 2 if len(lm) > 12 else 0.5
                        if nod_detector.update_and_check(nose_y, ref_y):
                            st.session_state.nod_count += 1
                    cv2.putText(frame, f"[Nod] count={st.session_state.nod_count}", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 2)

                # D) 表情分類 (節流)
                if opt_emote and client and (time.time() - last_emote_ts) > EMOTE_INTERVAL_SECONDS:
                    face_crop = va.crop_face_with_mediapipe(frame, face_detector)
                    cls = llm.gpt_image_classify_3cls(face_crop, client)
                    last_emote_ts = time.time()
                    if cls in ["喜歡", "中性", "討厭"]:
                        st.session_state.emotion_counter[cls] += 1
                    cv2.putText(frame, f"[Emotion] {cls}", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,0,200), 2)
                
                # 更新畫面與統計數據
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                stat_leftover.metric("餐盤殘留", f"剩: {st.session_state.leftover_counter['剩餘50%以上']} / 光: {st.session_state.leftover_counter['無剩餘']}")
                stat_nod.metric("點頭次數", st.session_state.nod_count)
                stat_emotion.write(f"表情分布: `{dict(st.session_state.emotion_counter)}`")
                
                time.sleep(1.0 / fps_display)

            # 釋放資源
            cap.release()
            cv2.destroyAllWindows()
            # 關閉 toggle 開關後，用一個提示訊息佔據畫面
            frame_slot.info("攝影機已關閉。")
        else:
            frame_slot.info("請點擊「開啟鏡頭」以開始即時分析。")