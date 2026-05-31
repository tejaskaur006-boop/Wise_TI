import requests
import json
import os

# Ollama runs locally on your machine at this address
# No API key needed — completely free and offline
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:0.6b"

def call_llm(system_prompt: str, user_message: str, max_tokens: int = 1000) -> str:
    """
    THE ONLY FUNCTION THAT TALKS TO OLLAMA.
    Every feature in the project calls this function.
    
    Now uses Qwen3 8B running locally via Ollama instead of Groq.
    No internet needed, no API key needed, completely free.
    
    CONNECTED TO:
    - features/team_formation.py  (team rationale)
    - features/email_drafting.py  (all emails)
    - features/evaluation.py      (guides + anomaly explanation)
    - features/rag.py             (committee Q&A chatbot)
    """
    
    # Combine system prompt and user message
    # Ollama uses a single prompt field
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,          # get full response at once, not word by word
        "options": {
            "num_predict": max_tokens,   # max length of response
            "temperature": 0.7,          # 0 = consistent, 1 = creative
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300   # wait up to 2 minutes for response
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama returned status {response.status_code}: {response.text}")
        
        result = response.json()
        return result["response"].strip()
    
    except requests.exceptions.ConnectionError:
        raise Exception(
            "Cannot connect to Ollama. Make sure Ollama is running — "
            "open a terminal and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise Exception(
            "Ollama took too long to respond. "
            "This can happen if your RAM is full. Close other apps and try again."
        )