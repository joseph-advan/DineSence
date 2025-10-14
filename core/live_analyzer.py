# core/live_analyzer.py
"""
即時分析引擎 (LiveAnalyzer) 的主類別。
這個類別封裝了所有關於影像擷取、多執行緒、佇列通訊及背景分析的複雜邏輯。
UI 層只需要與這個類別的 start(), stop(), get_latest_result() 方法互動即可。
"""

import cv2
import time
import asyncio
import threading
from queue import Queue, Empty, Full
from typing import Optional

# 從同一個 core 目錄導入我們定義的數據結構
from .types import FrameResult
# 從外部服務導入功能
from services import vision_analysis as va
from services import llm_handler as llm
from config import EMOTE_INTERVAL_SECONDS

class LiveAnalyzer:
    def __init__(self, model_pack: dict, menu_items: list, analysis_options: dict):
        self.model_pack = model_pack
        self.menu_items = menu_items
        self.analysis_options = analysis_options
        
        # 建立兩個佇列用於執行緒間通訊
        # 佇列大小設為1，確保我們永遠只處理最新的畫面，避免延遲累積
        self._frame_queue = Queue(maxsize=1)  # 存放待分析的原始畫面
        self._result_queue = Queue(maxsize=1) # 存放已分析完成的結果
        
        # 用於優雅地停止執行緒的事件信號
        self._stop_event = threading.Event()
        self._camera_thread = None
        self._worker_thread = None

    def _camera_loop(self):
        """
        [執行緒 1: 生產者]
        這個函式在獨立的執行緒中運行，專門負責從攝影機抓取畫面，
        並將畫面放入 `_frame_queue`。
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("錯誤：無法開啟攝影機。")
            # 可以在此處將錯誤訊息放入結果佇列，讓 UI 知道
            return

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            
            # 使用非阻塞方式放入佇列，以維持攝影機畫面的即時性
            try:
                self._frame_queue.put_nowait(frame)
            except Full:
                # 如果佇列是滿的（代表分析執行緒來不及處理），
                # 就把舊的畫面丟掉，換成最新的畫面。
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    pass
                self._frame_queue.put_nowait(frame)

            time.sleep(0.03) # 控制擷取頻率約 30 FPS
        
        cap.release()

    def _analysis_worker(self):
        """
        [執行緒 2: 消費者]
        這個函式在另一個獨立的執行緒中運行，負責從 `_frame_queue` 取出畫面，
        執行所有耗時的 CV 和 LLM 分析，然後將結果放入 `_result_queue`。
        """
        client = self.model_pack["client"]
        pose_detector = self.model_pack["pose_detector"]
        face_detector = self.model_pack["face_detector"]
        nod_detector = va.NodDetector()
        last_emote_ts = 0.0

        while not self._stop_event.is_set():
            try:
                # 阻塞式等待，直到有新的畫面可以分析
                frame = self._frame_queue.get(timeout=1)
            except Empty:
                continue

            # --- 執行所有分析任務 ---
            nod_event = False
            plate_event = ""
            emotion_event = ""
            display_info = {}

            # 1. 執行非同步的 LLM 任務
            async def run_llm_tasks(frame_copy):
                nonlocal last_emote_ts
                tasks = []
                if self.analysis_options.get("opt_emote") and client and (time.time() - last_emote_ts) > EMOTE_INTERVAL_SECONDS:
                    face_crop = va.crop_face_with_mediapipe(frame_copy, face_detector)
                    tasks.append(llm.gpt_image_classify_3cls(face_crop, client))
                    last_emote_ts = time.time()
                else:
                    tasks.append(asyncio.sleep(0, result="")) # 空任務佔位
                
                return await asyncio.gather(*tasks)

            llm_results = asyncio.run(run_llm_tasks(frame.copy()))
            emote_cls = llm_results[0]
            if emote_cls and emote_cls not in ["無臉", "API 錯誤", ""]:
                emotion_event = emote_cls
            display_info["emotion"] = emote_cls

            # 2. 執行同步的 CV 任務
            if self.analysis_options.get("opt_nod"):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose_detector.process(rgb)
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    nose_y = lm[0].y
                    ref_y = (lm[11].y + lm[12].y) / 2 if len(lm) > 12 else 0.5
                    if nod_detector.update_and_check(nose_y, ref_y):
                        nod_event = True
            
            # 3. 將所有結果打包成標準格式
            result = FrameResult(
                processed_frame=frame,
                nod_detected=nod_event,
                emotion_detected=emotion_event,
                plate_leftover_detected=plate_event,
                display_info=display_info
            )
            
            # 4. 將最終結果放入結果佇列
            try:
                self._result_queue.put_nowait(result)
            except Full:
                try: 
                    self._result_queue.get_nowait()
                except Empty:
                    pass
                self._result_queue.put_nowait(result)

    def start(self):
        """對外公開的方法：啟動整個分析引擎。"""
        if self._camera_thread and self._camera_thread.is_alive():
            print("警告：引擎已經在運行中。")
            return
            
        self._stop_event.clear()
        
        # 啟動攝影機執行緒
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        
        # 啟動分析執行緒
        self._worker_thread = threading.Thread(target=self._analysis_worker, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """對外公開的方法：停止整個分析引擎。"""
        self._stop_event.set()
        if self._camera_thread and self._camera_thread.is_alive():
            self._camera_thread.join(timeout=2)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2)
        
        # 清理執行緒物件，確保下次可以重新啟動
        self._camera_thread = None
        self._worker_thread = None

    def get_latest_result(self) -> Optional[FrameResult]:
        """對外公開的方法：讓 UI 層從這裡獲取最新的分析結果。"""
        try:
            # 使用非阻塞方式獲取，避免 UI 卡住
            return self._result_queue.get_nowait()
        except Empty:
            return None

