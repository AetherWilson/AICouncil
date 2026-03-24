from openai import OpenAI
import httpx
import os
import dotenv
from datetime import datetime
import opencc
import base64
import logging

# Load environment variables
dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Initialize OpenCC converter for Simplified to Traditional Chinese
try:
    converter = opencc.OpenCC('s2t.json')  # s2t: simplified to traditional
except Exception as exc:
    logger.warning("OpenCC converter unavailable: %s", exc)
    converter = None

redirect_url = os.getenv('gpt_redirect_url')
api_key = os.getenv('gpt_api_key')

if redirect_url:
    client = OpenAI(
        api_key=api_key,
        base_url=redirect_url,
        http_client=httpx.Client(
            base_url=redirect_url,
            follow_redirects=True,
        ),
    )
else:
    client = OpenAI(
        api_key=api_key
    )

def log_info(folder, file_name, file_content):
    """保存信息到指定文件夾和文件。
    param folder: 文件夾名稱
    param file_name: 文件名稱
    param file_content: 文件內容
    """
    
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Check if there are more than 10 .txt files and delete the oldest one
    txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    if len(txt_files) > 10:
        # Sort by creation time (oldest first)
        txt_files_with_time = [(f, os.path.getctime(os.path.join(folder, f))) for f in txt_files]
        txt_files_with_time.sort(key=lambda x: x[1])
        # Delete the oldest file
        oldest_file = os.path.join(folder, txt_files_with_time[0][0])
        os.remove(oldest_file)
    
    # Create full file path
    file_path = os.path.join(folder, file_name)
    
    # Write content to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(file_content)
    
    return file_path

def _build_user_content(user_prompt, image_urls=None, pdf_inputs=None):
    """Build multi-modal user content blocks for the OpenAI-compatible API."""
    content = [{"type": "text", "text": user_prompt}]

    if image_urls:
        for img_url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })

    if pdf_inputs:
        for pdf_info in pdf_inputs:
            file_path = pdf_info.get('filepath')
            if not file_path:
                continue
            try:
                with open(file_path, 'rb') as pdf_file:
                    encoded = base64.b64encode(pdf_file.read()).decode('utf-8')
                filename = pdf_info.get('filename') or os.path.basename(file_path)
                content.append({
                    "type": "file",
                    "file": {
                        "filename": filename,
                        "file_data": f"data:application/pdf;base64,{encoded}"
                    }
                })
            except FileNotFoundError:
                logger.warning("Skipping missing PDF file: %s", file_path)
                continue
            except OSError as exc:
                logger.warning("Skipping unreadable PDF file %s: %s", file_path, exc)
                continue

    return content


def _build_messages(system_prompt, user_prompt, chat_history=None, prefix=None, image_urls=None, pdf_inputs=None):
    """Build the message array used by both streaming and non-streaming calls."""
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    if chat_history:
        for message in chat_history:
            messages.append({"role": message["role"], "content": message["content"]})

    user_content = _build_user_content(user_prompt, image_urls=image_urls, pdf_inputs=pdf_inputs)
    if len(user_content) == 1 and not image_urls and not pdf_inputs:
        messages.append({"role": "user", "content": user_prompt})
    else:
        messages.append({"role": "user", "content": user_content})

    if prefix:
        messages.append({"role": "assistant", "content": prefix})

    return messages


def completion_response(model, system_prompt, user_prompt, chat_history = None, prefix = None, temperature=1.0, image_urls=None, pdf_inputs=None):
    """
    使用OpenAI API生成聊天回應。
    :param model: 使用的模型名稱。
    :param system_prompt: 系統提示語。
    :param user_prompt: 用戶輸入的提示語。
    :param chat_history: 可選的聊天歷史記錄，格式為字典。
    :param prefix: 可選的前綴，用於回應內容。
    :param temperature: 控制生成文本的隨機性，默認為1.0。
    :param image_urls: 可選的圖片URL列表或base64編碼的圖片列表。
    :return: 生成的回應內容。
    """
    messages = _build_messages(
        system_prompt,
        user_prompt,
        chat_history=chat_history,
        prefix=prefix,
        image_urls=image_urls,
        pdf_inputs=pdf_inputs
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
    except httpx.TimeoutException as exc:
        logger.exception("Model request timed out for %s", model)
        raise RuntimeError(f"Model request timed out for {model}") from exc
    except httpx.HTTPError as exc:
        logger.exception("Network error during model request for %s", model)
        raise RuntimeError(f"Network error during model request for {model}") from exc
    except Exception as exc:
        logger.exception("Unexpected model request failure for %s", model)
        raise RuntimeError(f"Model request failed for {model}") from exc

    # Get response content and add prefix if needed
    content = response.choices[0].message.content
    
    # Prepare log content with input and output clearly labeled
    log_content = f"""{'='*80}
INPUT
{'='*80}
System Prompt:
{system_prompt}

User Prompt:
{user_prompt}

Chat History:
{chat_history if chat_history else 'None'}

{'='*80}
OUTPUT
{'='*80}
{content}
"""
    
    log_info("gpt_responses", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model}.txt", log_content)
    return f"{prefix or ''}{content}"

def completion_response_stream(model, system_prompt, user_prompt, chat_history=None, prefix=None, temperature=1.0, image_urls=None, pdf_inputs=None):
    """
    使用OpenAI API以串流模式生成聊天回應，逐塊產出文字。
    :param model: 使用的模型名稱。
    :param system_prompt: 系統提示語。
    :param user_prompt: 用戶輸入的提示語。
    :param chat_history: 可選的聊天歷史記錄。
    :param prefix: 可選的前綴，用於回應內容。
    :param temperature: 控制生成文本的隨機性，默認為1.0。
    :param image_urls: 可選的圖片URL列表或base64編碼的圖片列表。
    :return: 逐塊產出文字的生成器。
    """
    messages = _build_messages(
        system_prompt,
        user_prompt,
        chat_history=chat_history,
        prefix=prefix,
        image_urls=image_urls,
        pdf_inputs=pdf_inputs
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True
        )
    except httpx.TimeoutException as exc:
        logger.exception("Stream start timed out for %s", model)
        raise RuntimeError(f"Model stream timed out for {model}") from exc
    except httpx.HTTPError as exc:
        logger.exception("Network error starting stream for %s", model)
        raise RuntimeError(f"Network error starting stream for {model}") from exc
    except Exception as exc:
        logger.exception("Unexpected stream start failure for %s", model)
        raise RuntimeError(f"Model stream failed for {model}") from exc

    full_content = prefix or ''
    full_thinking = ''
    in_think_tag = False  # state for parsing inline <think>...</think>
    completed = False

    try:
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # ── Source 1: DeepSeek-R1 / many proxy gateways ──────────────────────
            rc = getattr(delta, 'reasoning_content', None)
            if rc:
                full_thinking += rc
                yield ('thinking', rc)
                continue

            # ── Source 2: Some Anthropic-compatible proxies ───────────────────────
            th = getattr(delta, 'thinking', None)
            if th:
                full_thinking += th
                yield ('thinking', th)
                continue

            # ── Source 3: Inline <think>…</think> tags in content stream ─────────
            raw = delta.content if delta.content else ''
            while raw:
                if in_think_tag:
                    end = raw.find('</think>')
                    if end == -1:
                        full_thinking += raw
                        yield ('thinking', raw)
                        raw = ''
                    else:
                        thinking_chunk = raw[:end]
                        if thinking_chunk:
                            full_thinking += thinking_chunk
                            yield ('thinking', thinking_chunk)
                        raw = raw[end + 8:]   # skip </think>
                        in_think_tag = False
                else:
                    start = raw.find('<think>')
                    if start == -1:
                        full_content += raw
                        yield ('content', raw)
                        raw = ''
                    else:
                        before = raw[:start]
                        if before:
                            full_content += before
                            yield ('content', before)
                        raw = raw[start + 7:]  # skip <think>
                        in_think_tag = True

        completed = True

    finally:
        # Always close the HTTP connection — this is a no-op on normal completion
        # but immediately cancels the request when the generator is closed externally
        # (e.g. when the stop button is pressed and the caller breaks out of the loop)
        try:
            response.close()
        except (RuntimeError, OSError) as exc:
            logger.debug("Failed to close stream response cleanly: %s", exc)

    # Only log when the stream finished naturally (not when stopped early)
    if completed:
        log_content = f"""{'='*80}
INPUT
{'='*80}
System Prompt:
{system_prompt}

User Prompt:
{user_prompt}

Chat History:
{chat_history if chat_history else 'None'}

{'='*80}
THINKING
{'='*80}
{full_thinking if full_thinking else 'None'}

{'='*80}
OUTPUT
{'='*80}
{full_content}
"""
        log_info("gpt_responses", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model}.txt", log_content)


def convert_to_traditional_chinese(text):
    """
    將簡體中文轉換為繁體中文。
    :param text: 需要轉換的文本
    :return: 轉換後的文本
    """
    if converter is None:
        return text
    try:
        return converter.convert(text)
    except Exception as e:
        print(f"Error converting to Traditional Chinese: {e}")
        return text

if __name__ == "__main__":
  completion = completion_response(
      model="deepseek-v3.2-thinking", 
      system_prompt="你是一位專業的小説作家，請根據以下需求撰寫文章：",
      user_prompt="請撰寫一個二百字以内的賽博朋克世界觀。"
  )
  print(completion)