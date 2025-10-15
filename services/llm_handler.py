# services/llm_handler.py

import base64
import io
import json
import asyncio
import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from PIL import Image
import cv2

# --- 導入我們建立的 prompt 讀取工具 ---
from utils.prompt_loader import load_prompt_template

# 使用 AsyncClient 進行非同步請求
aclient = httpx.AsyncClient()

def get_openai_client(api_key):
    """根據 API Key 初始化並返回異步的 OpenAI 客戶端物件。"""
    if not api_key:
        return None
    try:
        return AsyncOpenAI(api_key=api_key)
    except Exception as e:
        print(f"初始化 OpenAI Client 失敗: {e}")
        return None

def _image_to_base64(pil_image):
    """將 PIL.Image 物件轉換為 base64 字串。"""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

async def gpt_image_classify_3cls(face_bgr, client: AsyncOpenAI, model="gpt-4o-mini"):
    """
    (非同步) 使用 GPT-4o-mini 進行三分類表情辨識。
    [MODIFIED] 現在返回 (情緒字串, Token用量物件) 的元組。
    """
    if face_bgr is None: return "無臉", None
    if client is None: return "（未設定 API）", None

    pil_img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    b64_img = _image_to_base64(pil_img)
    
    prompt = (
        "請根據臉部表情，在三類中擇一輸出（請只輸出一個詞）：\n"
        "『喜歡』（正向/微笑）、『中性』、或『討厭』（厭惡/皺眉）。\n"
        "只輸出：喜歡 / 中性 / 討厭。"
    )
    
    try:
        resp = await client.chat.completions.create(
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
        usage = resp.usage  # <-- [NEW] 獲取 Token 用量

        emotion = "中性"
        if "喜歡" in text: emotion = "喜歡"
        if "討厭" in text: emotion = "討厭"
        
        return emotion, usage  # <-- [MODIFIED] 返回元組

    except Exception as e:
        print(f"表情分類 API 錯誤: {e}")
        return "API 錯誤", None


async def gpt_food_from_menu(plate_bgr, menu_items, client: AsyncOpenAI, model="gpt-4o-mini"):
    """(非同步) 根據提供的菜單，使用 GPT-4o-mini 辨識畫面中的餐點。"""
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
        resp = await client.chat.completions.create(
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


# --- [MODIFIED] 修改此函式以返回 (摘要文字, Token用量) 的元組 ---
async def summarize_session(stats: dict, store_type: str, tone: str, tips_style: str,
                      client: AsyncOpenAI, model="gpt-4o-mini"):
    """
    (非同步) 根據統計數據，生成客製化的顧客體驗摘要報告。
    Prompt 模板從 /prompts 資料夾動態載入。
    
    Returns:
        tuple[str, openai.types.CompletionUsage | None]: (摘要文字, Token 用量物件)
    """
    if client is None:
        return "（未設定 OPENAI_API_KEY，無法產生摘要）", None

    try:
        system_template = load_prompt_template('summarize_session', 'system')
        user_template = load_prompt_template('summarize_session', 'user')
    except FileNotFoundError as e:
        error_msg = f"找不到 Prompt 模板檔案：{e}。請確認 'prompts/summarize_session/' 資料夾及內部的 .txt 檔案是否存在。"
        print(error_msg)
        return error_msg, None

    system_prompt = system_template.format(
        store_type=store_type,
        tone=tone,
        tips_style=tips_style
    )
    user_prompt = user_template.format(
        stats_json=json.dumps(stats, indent=2, ensure_ascii=False)
    )
    
    try:
        # 接收完整的 API 回應物件
        r: ChatCompletion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7, max_tokens=800,
        )
        # 從回應物件中分別取出「內容」和「用量」
        summary_text = r.choices[0].message.content
        usage_data = r.usage
        
        # 將兩者作為一個元組返回
        return summary_text, usage_data
        
    except Exception as e:
        error_msg = f"生成摘要時發生錯誤：{e}"
        print(f"摘要生成 API 錯誤: {e}")
        return error_msg, None