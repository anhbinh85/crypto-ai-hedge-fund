import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print("DEEPSEEK_API_KEY:", repr(DEEPSEEK_API_KEY))
print("GEMINI_API_KEY:", repr(GEMINI_API_KEY))

prompt = "Say hello from DeepSeek/Gemini."

def test_deepseek():
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.7,
    }
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=20)
        resp.raise_for_status()
        print("DeepSeek response:", resp.json()["choices"][0]["message"]["content"])
    except Exception as e:
        print("DeepSeek API error:", e)

def test_gemini():
    url = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 32, "temperature": 0.7}
    }
    try:
        resp = requests.post(url, headers=headers, params=params, json=data, timeout=20)
        resp.raise_for_status()
        print("Gemini response:", resp.json()["candidates"][0]["content"]["parts"][0]["text"])
    except Exception as e:
        print("Gemini API error:", e)

print("\nTesting DeepSeek...")
test_deepseek()
print("\nTesting Gemini...")
test_gemini() 