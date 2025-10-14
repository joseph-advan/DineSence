# customer_analysis_mvp/services/vision_analysis.py

"""
本模組包含所有電腦視覺相關的函式與類別。
- 初始化與提供 YOLOv8、MediaPipe 等模型。
- 包含餐盤殘留、點頭、臉部裁切、食物偵測等核心 CV 演算法。
"""

import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp
from ultralytics import YOLO

# 從我們的設定檔導入常數
from config import YOLO_MODEL_PATH, FOODISH_CLASSES, NOD_BUFFER_LEN

# --- 模型初始化 ---
# 這些模型物件在模組首次被導入時僅會被載入一次，效率較高。

try:
    _yolo_food = YOLO(YOLO_MODEL_PATH)
    _yolo_ok = True
except Exception as e:
    print(f"無法載入 YOLO 模型: {e}")
    _yolo_food = None
    _yolo_ok = False

mp_pose_solution = mp.solutions.pose
mp_face_solution = mp.solutions.face_detection

def get_pose_detector():
    """返回一個新的 MediaPipe Pose 偵測器實例。"""
    return mp_pose_solution.Pose(
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def get_face_detector():
    """返回一個新的 MediaPipe Face Detection 偵測器實例。"""
    return mp_face_solution.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )

# --- (A) 餐盤殘留分析 ---
def estimate_plate_leftover(bgr_frame):
    """
    使用霍夫圓轉換與顏色閾值估計餐盤殘留量。
    Returns:
        tuple: (狀態標籤, 殘留比例, 圓心與半徑)
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
        param1=100, param2=30, minRadius=60, maxRadius=0
    )
    if circles is None:
        return "未偵測到餐盤", None, None

    circles = np.round(circles[0, :]).astype("int")
    x, y, r = max(circles, key=lambda c: c[2])
    h, w = bgr_frame.shape[:2]
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return "餐盤不完整", None, (x, y, r)

    roi = bgr_frame[y - r:y + r, x - r:x + r].copy()
    mask = np.zeros((2 * r, 2 * r), dtype=np.uint8)
    cv2.circle(mask, (r, r), r - 2, 255, -1)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    white_lo = (S < 50) & (V > 200)
    white_hi = (S < 35) & (V > 210)
    food_mask = (~white_lo) | (~white_hi)
    food_mask = food_mask & (mask > 0)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    food_mask = cv2.morphologyEx(food_mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, k, iterations=1)

    total = np.count_nonzero(mask)
    food_pixels = np.count_nonzero(food_mask)
    if total == 0:
        return "餐盤區域無效", None, (x, y, r)

    non_white_ratio = food_pixels / total
    label = "剩餘50%以上" if non_white_ratio >= 0.5 else "無剩餘"
    return label, float(non_white_ratio), (x, y, r)


# --- (B) 點頭偵測 ---
class NodDetector:
    """
    一個有狀態的類別，用於偵測連續畫面中的點頭動作。
    """
    def __init__(self, buf_len=NOD_BUFFER_LEN, amp_thresh=0.03, cooldown=1.0):
        self.y_hist = deque(maxlen=buf_len)
        self.last_nod_ts = 0.0
        self.amp_thresh = amp_thresh
        self.cooldown = cooldown
        self.count = 0

    def update_and_check(self, nose_y, ref_y):
        rel = nose_y - ref_y
        self.y_hist.append(rel)

        if len(self.y_hist) < self.y_hist.maxlen:
            return False

        arr = np.array(self.y_hist, dtype=np.float32)
        arr = cv2.GaussianBlur(arr.reshape(-1, 1), (5, 1), 0).flatten()

        amp = arr.max() - arr.min()
        sign_changes = np.sum(np.diff(np.sign(np.diff(arr))) != 0)

        now = time.time()
        if amp > self.amp_thresh and sign_changes >= 2 and (now - self.last_nod_ts) > self.cooldown:
            self.last_nod_ts = now
            self.count += 1
            return True
        return False


# --- (C) 臉部裁切 ---
def crop_face_with_mediapipe(bgr_frame, detector, min_conf=0.6):
    """使用 MediaPipe 偵測並裁切出臉部 ROI。"""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)
    if not res.detections:
        return None
    det = res.detections[0]
    if det.score[0] < min_conf:
        return None
    h, w = bgr_frame.shape[:2]
    bbox = det.location_data.relative_bounding_box
    x1 = max(0, int(bbox.xmin * w)); y1 = max(0, int(bbox.ymin * h))
    x2 = min(w, int((bbox.xmin + bbox.width) * w)); y2 = min(h, int((bbox.ymin + bbox.height) * h))
    if x2 <= x1 or y2 <= y1: return None
    return bgr_frame[y1:y2, x1:x2]


# --- (D) YOLO 食物/杯子偵測 ---
def detect_food_regions_yolo(bgr, conf=0.3, min_area_ratio=0.01):
    """使用 YOLOv8 偵測所有與食物相關的物件。"""
    if not _yolo_ok: return []
    
    out = []
    res = _yolo_food(bgr, conf=conf, iou=0.45, verbose=False)[0]
    h, w = bgr.shape[:2]

    for b in res.boxes:
        cls_id = int(b.cls.item())
        name = res.names.get(cls_id, "")
        if name not in FOODISH_CLASSES:
            continue
        
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        if area / (w * h) < min_area_ratio:
            continue
            
        out.append({
            "xyxy": (x1, y1, x2, y2),
            "label": name,
            "conf": float(b.conf.item())
        })
    # 按面積大小排序，讓最大的物件在最前面
    out.sort(key=lambda d: (d["xyxy"][2]-d["xyxy"][0])*(d["xyxy"][3]-d["xyxy"][1]), reverse=True)
    return out

def has_big_cup(bgr, min_area_ratio=0.04):
    """檢查畫面中是否有一個尺寸足夠大的杯子。"""
    if not _yolo_ok: return False
    
    res = _yolo_food(bgr, conf=0.3, iou=0.45, verbose=False)[0]
    h, w = bgr.shape[:2]
    
    for b in res.boxes:
        name = res.names.get(int(b.cls.item()), "")
        if name in ["cup", "wine glass", "bottle"]:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            if ((x2 - x1) * (y2 - y1)) / (w * h) >= min_area_ratio:
                return True
    return False