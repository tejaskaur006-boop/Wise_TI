import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# This creates one Groq client that the whole project uses
# It reads your API key from the .env file
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(system_prompt: str, user_message: str, max_tokens: int = 1000) -> str:
    """
    THE ONLY FUNCTION THAT TALKS TO GROQ.
    Every feature in the project calls this function.
    
    system_prompt = instructions telling the LLM what role to play
    user_message  = the actual request with all the data
    max_tokens    = how long the response can be (1000 is fine for most cases)
    
    Returns: plain string response from the LLM
    
    CONNECTED TO:
    - features/team_formation.py  (calls this for team rationale)
    - features/email_drafting.py  (calls this for all emails)
    - features/evaluation.py      (calls this for guides + anomaly explanation)
    - features/rag.py             (calls this for committee Q&A)
    """
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # llama3 8B model, free on Groq, good quality
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ]
    )
    
    return response.choices[0].message.content
