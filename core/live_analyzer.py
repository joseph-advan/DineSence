# core/live_analyzer.py

"""
【版本 2.0 - 高流暢度版】
即時分析引擎 (LiveAnalyzer) 的主類別。
此版本徹底分離了影像流與數據流，以實現流暢的 UI 畫面更新。
- 影像流: 透過 _frame_display_queue 直接將原始畫面高速傳送至 UI。
- 數據流: _analysis_worker 執行緒將耗時的分析結果放入 _analysis_result_queue。
UI 層可以獨立地、非同步地獲取這兩者。
"""

import cv2
import time
import asyncio
import threading
from queue import Queue, Empty, Full
from typing import Optional, Dict
import numpy as np

# 從外部服務導入功能
from services import vision_analysis as va
from services import llm_handler as llm
from config import EMOTE_INTERVAL_SECONDS, CAMERA_RESOLUTION_WIDTH, CAMERA_RESOLUTION_HEIGHT, CAMERA_BUFFER_SIZE

# --- 導入新的數據結構 ---
# 我們需要一個新的 dataclass 只用來存放分析結果
from .types import AnalysisResult

class LiveAnalyzer:
    def __init__(self, model_pack: dict, menu_items: list, analysis_options: dict):
        self.model_pack = model_pack
        self.menu_items = menu_items
        self.analysis_options = analysis_options
        
        # --- MODIFIED: 建立三條獨立的佇列 ---
        # 1. 高速公路: 原始畫面直送 UI，確保流暢度
        self._frame_display_queue = Queue(maxsize=CAMERA_BUFFER_SIZE)
        
        # 2. 待辦事項: 待分析的原始畫面
        self._frame_analysis_queue = Queue(maxsize=CAMERA_BUFFER_SIZE) 
        
        # 3. 慢車道: 已分析完成的數據結果
        self._analysis_result_queue = Queue(maxsize=CAMERA_BUFFER_SIZE)
        
        # 用於優雅地停止執行緒的事件信號
        self._stop_event = threading.Event()
        self._camera_thread = None
        self._worker_thread = None

    def _camera_loop(self):
        """
        [執行緒 1: 生產者]
        專門負責從攝影機抓取畫面，並立即將原始畫面放入兩個佇列：
        一個給 UI 即時顯示，一個給背景分析。
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("錯誤：無法開啟攝影機。")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_HEIGHT)

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            
            # --- MODIFIED: 將畫面同時放入兩個佇列 ---
            # 使用非阻塞方式，如果佇列滿了就丟棄舊的，確保即時性
            try:
                self._frame_display_queue.put_nowait(frame)
            except Full:
                pass # 顯示佇列滿了沒關係，代表 UI 暫時跟不上，下一幀就好

            try:
                self._frame_analysis_queue.put_nowait(frame)
            except Full:
                pass # 分析佇列滿了也沒關係，代表分析執行緒忙不過來

            time.sleep(0.02) # 控制擷取頻率約 30-50 FPS
        
        cap.release()

    def _analysis_worker(self):
        """
        [執行緒 2: 消費者]
        負責從 _frame_analysis_queue 取出畫面，執行所有耗時分析，
        然後只將「分析結果數據」放入 _analysis_result_queue。
        """
        client = self.model_pack["client"]
        pose_detector = self.model_pack["pose_detector"]
        face_detector = self.model_pack["face_detector"]
        nod_detector = va.NodDetector()
        last_emote_ts = 0.0

        while not self._stop_event.is_set():
            try:
                frame = self._frame_analysis_queue.get(timeout=1)
            except Empty:
                continue

            # --- 初始化本次分析的結果容器 ---
            result = AnalysisResult()

            # --- 1. 執行非同步的 LLM 任務 (表情分析) ---
            # 這個設計確保了 LLM 呼叫不會阻塞下面的 CV 任務
            async def run_emotion_task(frame_copy):
                nonlocal last_emote_ts
                if self.analysis_options.get("opt_emote") and client and (time.time() - last_emote_ts) > EMOTE_INTERVAL_SECONDS:
                    last_emote_ts = time.time()
                    face_crop = va.crop_face_with_mediapipe(frame_copy, face_detector)

                    # [MODIFIED] 接收 (emotion, usage) 元組
                    emotion, usage = await llm.gpt_image_classify_3cls(face_crop, client)

                    if emotion and emotion not in ["無臉", "API 錯誤", ""]:
                        result.emotion_event = emotion
                    if usage:
                        # [NEW] 將 usage 存入 AnalysisResult
                        result.token_usage_event = {
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens
                        }
                
            asyncio.run(run_emotion_task(frame.copy()))

            # --- 2. 執行同步的 CV 任務 ---
            if self.analysis_options.get("opt_nod"):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose_detector.process(rgb)
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    nose_y = lm[0].y
                    ref_y = (lm[11].y + lm[12].y) / 2 if len(lm) > 12 else 0.5
                    if nod_detector.update_and_check(nose_y, ref_y):
                        result.nod_event = True
            
            if self.analysis_options.get("opt_plate"):
                label, _, circle = va.estimate_plate_leftover(frame)
                if label in ["剩餘50%以上", "無剩餘"]:
                    result.plate_event = label
                result.display_info["plate_label"] = label
                if circle:
                    result.display_info["plate_circle"] = circle
            
            # --- 3. 將純數據的分析結果放入結果佇列 ---
            try:
                self._analysis_result_queue.put_nowait(result)
            except Full:
                pass # 如果 UI 來不及取，就丟掉舊的數據

    def start(self):
        """對外公開的方法：啟動整個分析引擎。"""
        if self._camera_thread and self._camera_thread.is_alive():
            return
        self._stop_event.clear()
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self._camera_thread.start()
        self._worker_thread.start()

    def stop(self):
        """對外公開的方法：停止整個分析引擎。"""
        self._stop_event.set()
        if self._camera_thread: self._camera_thread.join(timeout=2)
        if self._worker_thread: self._worker_thread.join(timeout=2)
        self._camera_thread = None
        self._worker_thread = None

    # --- MODIFIED: 提供兩個獨立的獲取方法 ---
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """(UI呼叫) 從高速公路獲取最新的原始畫面。"""
        try:
            return self._frame_display_queue.get_nowait()
        except Empty:
            return None

    def get_latest_analysis_result(self) -> Optional[AnalysisResult]:
        """(UI呼叫) 從慢車道獲取最新的分析數據。"""
        try:
            return self._analysis_result_queue.get_nowait()
        except Empty:
            return None