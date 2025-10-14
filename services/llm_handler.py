# customer_analysis_mvp/services/llm_handler.py

"""
本模組負責處理所有與大型語言模型 (LLM) 的互動。
- 初始化 OpenAI 客戶端。
- 封裝 VLM (視覺語言模型) 相關的 API 呼叫，如表情分類、食物分類。
- 封裝純文字的 API 呼叫，如最終的摘要生成。
"""

import base64
import io
import json
from openai import OpenAI
from PIL import Image
import cv2

def get_openai_client(api_key):
    """根據 API Key 初始化並返回 OpenAI 客戶端物件。"""
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        print(f"初始化 OpenAI Client 失敗: {e}")
        return None

def _image_to_base64(pil_image):
    """將 PIL.Image 物件轉換為 base64 字串。"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def gpt_image_classify_3cls(face_bgr, client: OpenAI, model="gpt-4o-mini"):
    """使用 GPT-4o-mini 進行三分類表情辨識。"""
    if face_bgr is None: return "無臉"
    if client is None: return "（未設定 API）"

    pil_img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    b64_img = _image_to_base64(pil_img)
    
    prompt = (
        "請根據臉部表情，在三類中擇一輸出（請只輸出一個詞）：\n"
        "『喜歡』（正向/微笑）、『中性』、或『討厭』（厭惡/皺眉）。\n"
        "只輸出：喜歡 / 中性 / 討厭。"
    )
    
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                ],
            }],
            temperature=0, max_tokens=10
        )
        text = resp.choices[0].message.content.strip()
        if "喜歡" in text: return "喜歡"
        if "討厭" in text: return "討厭"
        return "中性"
    except Exception as e:
        print(f"表情分類 API 錯誤: {e}")
        return "API 錯誤"


def gpt_food_from_menu(plate_bgr, menu_items, client: OpenAI, model="gpt-4o-mini"):
    """根據提供的菜單，使用 GPT-4o-mini 辨識畫面中的餐點。"""
    if plate_bgr is None: return {"label": "未知", "confidence": 0.0, "rationale": "無ROI"}
    if client is None: return {"label": "（未設定API）", "confidence": 0.0, "rationale": ""}

    pil_img = Image.fromarray(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB))
    b64_img = _image_to_base64(pil_img)

    options = ", ".join(menu_items)
    prompt = (
        "請只根據提供的菜單清單判斷畫面中的餐點屬於哪一項，並以 JSON 格式輸出：\n"
        '{ "label": "<從菜單中擇一>", "confidence": 0.0~1.0 的小數, "rationale": "一句簡短的判斷原因" }\n'
        f"菜單清單：[{options}]\n"
        "如果畫面中的餐點與任何清單項目都不太相符，請選擇最相近的一項，但給予較低的信心度（<=0.4）。\n"
        "請務必只輸出 JSON 物件，不要包含任何前後的文字或 markdown 標籤。"
    )

    try:
        resp = client.chat.completions.create(
            model=model, response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                ],
            }],
            temperature=0
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(text)
        return {
            "label": str(data.get("label", "未知")),
            "confidence": float(data.get("confidence", 0.0)),
            "rationale": str(data.get("rationale", ""))[:120]
        }
    except Exception as e:
        print(f"食物分類 API 錯誤: {e}")
        return {"label": "解析失敗", "confidence": 0.0, "rationale": str(e)[:120]}


def summarize_session(stats: dict, store_type: str, tone: str, tips_style: str,
                      client: OpenAI, model="gpt-4o-mini"):
    """根據統計數據，生成客製化的顧客體驗摘要報告。"""
    if client is None:
        return "（未設定 OPENAI_API_KEY，無法產生摘要）"

    system_prompt = (
        f"你是一位專業的{store_type}顧客體驗分析師。你的任務是將量化的行為數據，轉化為生動且富有洞察力的質化觀察報告。"
        f"請使用「{tone}」的語氣撰寫，並從「{tips_style}」的角度提出具體建議。"
    )
    user_prompt = (
        "這是一份顧客用餐期間的行為與情緒數據紀錄。請模擬你是一位在現場的觀察員，深入解讀這些數據背後的顧客體驗故事。\n\n"
        "數據 (JSON格式):\n"
        f"{json.dumps(stats, indent=2, ensure_ascii=False)}\n\n"
        "請嚴格遵循以下格式輸出報告，用詞需自然流暢，避免制式化：\n\n"
        "### 顧客動態觀察\n"
        "（請依據 timeline 或情緒變化，用 2-3 句話生動地描述顧客可能的用餐過程，例如：「顧客在餐點剛上桌時表情愉悅，並出現點頭動作，顯示對餐點的第一印象不錯。但用餐中期表情轉為中性，且最後餐盤有剩餘，可能表示份量或口味的後續體驗有落差。」）\n\n"
        "### 整體體驗評估\n"
        "（綜合所有數據，歸納顧客本次的整體滿意度趨勢，點出正面與負面的關鍵指標。）\n\n"
        "### actionable 營運建議\n"
        "（根據你的觀察，提出 2-3 點具體、可執行的改善建議。）\n\n"
        "**特別注意**：如果數據中出現『討厭』表情或『剩餘50%以上』的次數總和超過 5 次，請在報告最上方加上一個【⚠️ 警示：需優先關注】的標題，並在建議中強調可能的問題點。"
    )

    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7, max_tokens=800,
        )
        return r.choices[0].message.content
    except Exception as e:
        print(f"摘要生成 API 錯誤: {e}")
        return f"生成摘要時發生錯誤：{e}"