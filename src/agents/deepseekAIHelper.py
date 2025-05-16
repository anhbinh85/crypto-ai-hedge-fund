import os
import requests
import logging
import time

logger = logging.getLogger("deepseekAIHelper")
logger.setLevel(logging.INFO)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DEEPSEEK_API_KEY:
    logger.error("DEEPSEEK_API_KEY is missing from environment variables!")
else:
    logger.info("DEEPSEEK_API_KEY loaded.")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY is missing from environment variables!")
else:
    logger.info("GEMINI_API_KEY loaded.")

def call_deepseek(prompt, max_tokens=512, retries=2, timeout=60):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    logger.info(f"Calling DeepSeek API with prompt length {len(prompt)} and max_tokens={max_tokens}")
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=timeout)
            resp.raise_for_status()
            logger.info("DeepSeek API call successful.")
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout as e:
            logger.error(f"DeepSeek API call timed out (attempt {attempt+1}/{retries+1}): {e}")
            if attempt < retries:
                time.sleep(2)  # Wait before retrying
                continue
            else:
                raise
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            raise

def call_gemini(prompt, max_tokens=512):
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.7}
    }
    logger.info(f"Calling Gemini API with prompt length {len(prompt)} and max_tokens={max_tokens}")
    try:
        resp = requests.post(url, headers=headers, params=params, json=data, timeout=20)
        resp.raise_for_status()
        logger.info("Gemini API call successful.")
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise

def analyze_indicators_with_llm(indicator_summary, model_preference='deepseek'):
    prompt = (
        "You are a professional crypto trading analyst. "
        "Given the following technical indicator summary, explain the current market situation, "
        "highlight key signals, and conclude the likely trend (bullish, bearish, or neutral). "
        "Be concise, actionable, and use markdown and emojis for clarity. "
        "Here are the indicators:\n\n"
        f"{indicator_summary}\n\n"
        "Explain and conclude:"
    )
    logger.info(f"analyze_indicators_with_llm called with model_preference={model_preference}")
    try:
        if model_preference == 'deepseek':
            return call_deepseek(prompt)
        else:
            return call_gemini(prompt)
    except Exception as e:
        logger.error(f"Primary LLM ({model_preference}) failed: {e}")
        if model_preference == 'deepseek':
            # fallback to gemini
            try:
                return call_gemini(prompt)
            except Exception as e2:
                logger.error(f"Gemini fallback also failed: {e2}")
                return f"❌ Both DeepSeek and Gemini failed: {e} | {e2}"
        else:
            return f"❌ LLM call failed: {e}" 