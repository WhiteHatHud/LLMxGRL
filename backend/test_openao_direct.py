# test_openai_direct.py
import httpx
import json

# Replace with your new API key after revoking the exposed one
API_KEY = "insert key here"

url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Say hello"}],
    "temperature": 0.7,
    "max_tokens": 50
}

try:
    response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    response.raise_for_status()
    data = response.json()
    print(f"Success! Response: {data['choices'][0]['message']['content']}")
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, 'response'):
        print(f"Response body: {e.response.text}")