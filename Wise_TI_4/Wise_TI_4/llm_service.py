from dotenv import load_dotenv
load_dotenv()  # loads .env file BEFORE anything else reads env vars

import requests
import json
import os

# Ollama Cloud — uses your Pro subscription, runs on Ollama's GPUs
# Falls back to local Ollama automatically if no API key is set
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
USE_CLOUD = bool(OLLAMA_API_KEY)

# Using /api/chat endpoint — handles prompt formatting automatically for any model
OLLAMA_CLOUD_URL = "https://ollama.com/api/chat"
OLLAMA_LOCAL_URL = "http://localhost:11434/api/chat"

CLOUD_MODEL_NAME = "gemma3:12b"
LOCAL_MODEL_NAME = "qwen3:0.6b"


def call_llm(system_prompt: str, user_message: str, max_tokens: int = 1000) -> str:
    """
    THE ONLY FUNCTION THAT TALKS TO THE LLM.
    Every feature in the project calls this function.

    Uses Ollama Cloud (gpt-oss:20b) if OLLAMA_API_KEY is set in .env,
    otherwise falls back to local Ollama (qwen3:0.6b) automatically.

    Uses the /api/chat endpoint which handles prompt formatting
    automatically for any model (no need to know chat template).

    CONNECTED TO:
    - features/team_formation.py  (team rationale)
    - features/email_drafting.py  (all emails)
    - features/evaluation.py      (guides + anomaly explanation)
    - features/rag.py             (committee Q&A chatbot)
    """

    url = OLLAMA_CLOUD_URL if USE_CLOUD else OLLAMA_LOCAL_URL
    model = CLOUD_MODEL_NAME if USE_CLOUD else LOCAL_MODEL_NAME

    headers = {}
    if USE_CLOUD:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
        headers["Content-Type"] = "application/json"

    # Build messages array for /api/chat endpoint
    # This works for any model — no need to know chat template format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": None,           # Don't stop on any specific tokens
            "repeat_penalty": 1.1   # Prevent repetition
        }
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=120 if USE_CLOUD else 300   # cloud is fast, fail faster if something's wrong
        )

        if response.status_code != 200:
            error_text = response.text[:500]
            raise Exception(f"Ollama returned status {response.status_code}: {error_text}")

        result = response.json()

        # /api/chat returns content in "message.content" field
        # Different from /api/generate which uses "response" field
        if "message" in result and "content" in result["message"]:
            content = result["message"]["content"].strip()
            if not content:
                # Some models (like gpt-oss) put thinking in a separate field
                # If content is empty, the model might be using "thinking" mode
                # Try to get thinking content as fallback
                if "thinking" in result and result["thinking"]:
                    return result["thinking"].strip()
                raise Exception("LLM returned empty content. Response: " + json.dumps(result)[:300])
            return content
        elif "response" in result:
            # Fallback for /api/generate format
            return result["response"].strip()
        else:
            raise Exception("Unexpected response format: " + json.dumps(result)[:300])

    except requests.exceptions.ConnectionError:
        if USE_CLOUD:
            raise Exception(
                "Cannot connect to Ollama Cloud. Check your internet connection and API key."
            )
        raise Exception(
            "Cannot connect to Ollama. Make sure Ollama is running — "
            "open a terminal and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise Exception(
            "LLM took too long to respond. Try again in a moment."
        )
    except Exception as e:
        # Re-raise with more context
        raise Exception(f"LLM call failed: {str(e)}")
