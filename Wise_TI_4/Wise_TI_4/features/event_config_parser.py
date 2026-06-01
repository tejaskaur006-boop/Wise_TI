import json
import re
from llm_service import call_llm

# ─────────────────────────────────────────────
# HELPER: Safe JSON Parser
# Ollama/Qwen3 sometimes adds <think> tags or
# extra explanation around JSON — this handles that
# ─────────────────────────────────────────────

def safe_parse_json(text: str) -> dict:
    """
    Extracts JSON from LLM response even if model adds extra text.
    Qwen3 adds <think>...</think> blocks — we strip those first.
    """
    # Remove <think>...</think> blocks (Qwen3 thinking mode)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON block inside the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    # If all fails return empty dict — caller handles this
    return {}


# ─────────────────────────────────────────────
# LLM CALL 1: Parse free text into structured JSON
# ─────────────────────────────────────────────

def parse_event_description(description: str) -> dict:
    """
    Takes free text description from committee and extracts
    structured event configuration as JSON.
    
    Example input:
    "We are running a 2-day hackathon. Teams of 3. Balanced skills.
     No two from same college. 4 judges. Criteria: innovation 40%,
     technical depth 40%, presentation 20%. Top 3 qualify.
     Flag scores differing by more than 20 points."
    
    Example output:
    {
      "event_name": "Hackathon",
      "team_size": 3,
      "team_rules": {"skill_balance": true, "no_same_institution": true},
      "stages": [...],
      "evaluation": {"judges_count": 4, "criteria": [...], "anomaly_threshold": 20},
      "progression": {"top_n": 3}
    }
    
    CALLED BY: main.py in POST /api/events/configure
    CALLS: call_llm() from llm_service.py
    """
    
    system = """You are an event configuration parser.
Extract structured configuration from natural language event descriptions.
Respond ONLY in valid JSON. No explanation, no markdown, no backticks, no <think> tags.
If information is missing or unclear, set the value to null.
Never guess — only extract what is explicitly stated."""

    user = f"""Parse this event description into structured configuration:

"{description}"

Return JSON with exactly this structure:
{{
  "event_name": "string or null",
  "duration_days": number or null,
  "team_size": number or null,
  "team_rules": {{
    "skill_balance": true/false or null,
    "no_same_institution": true/false or null,
    "experience_balance": true/false or null
  }},
  "stages": [
    {{
      "name": "string",
      "type": "work_period/evaluation/team_formation/communication/ceremony",
      "day": number or null,
      "duration_hours": number or null,
      "description": "string"
    }}
  ],
  "evaluation": {{
    "judges_count": number or null,
    "minutes_per_team": number or null,
    "aggregation": "average/weighted/sum or null",
    "anomaly_threshold": number or null,
    "criteria": [
      {{"name": "string", "weight": number}}
    ]
  }},
  "progression": {{
    "top_n": number or null,
    "method": "string or null"
  }},
  "communication_touchpoints": ["list of when emails should be sent"] or []
}}"""

    response = call_llm(system, user, max_tokens=1000)
    config = safe_parse_json(response)
    
    if not config:
        raise Exception(
            "Could not parse your description into a configuration. "
            "Please try describing your event with more specific details."
        )
    
    return config


# ─────────────────────────────────────────────
# LLM CALL 2: Find what information is missing
# ─────────────────────────────────────────────

def find_config_gaps(config: dict) -> list:
    """
    Checks the parsed config for missing critical information.
    Fixed version — handles Qwen3's tendency to repeat JSON.
    """
    import re

    # First do a simple Python check for truly critical fields
    # This avoids an LLM call for obvious cases
    critical_missing = []

    if not config.get("team_size"):
        critical_missing.append("What is the team size (how many people per team)?")

    if not config.get("stages"):
        critical_missing.append("What are the stages or phases of your event?")

    eval_section = config.get("evaluation", {})
    if eval_section.get("judges_count") and not eval_section.get("criteria"):
        critical_missing.append("What are the scoring criteria and their percentage weights?")

    if eval_section.get("judges_count", 0) > 1 and not eval_section.get("anomaly_threshold"):
        critical_missing.append("What score difference should trigger an anomaly flag?")

    # If we already found critical missing fields, return them
    # No need to call LLM
    if critical_missing:
        return critical_missing

    # Everything critical is present — ask LLM only for non-obvious gaps
    system = """You are an event configuration validator.
Look at this event config and decide if it has enough information to run.
YOU MUST respond with ONLY one of these two options:
Option A: The single word COMPLETE (if enough info exists)
Option B: A numbered list of questions (if critical info is missing)
DO NOT repeat the JSON. DO NOT add explanation. ONLY respond with COMPLETE or questions."""

    user = f"""Event config to validate:
team_size: {config.get('team_size')}
stages count: {len(config.get('stages', []))}
judges: {config.get('evaluation', {}).get('judges_count')}
criteria: {config.get('evaluation', {}).get('criteria')}
anomaly_threshold: {config.get('evaluation', {}).get('anomaly_threshold')}
top_n: {config.get('progression', {}).get('top_n')}

Is there enough information to run this event?
If yes: respond COMPLETE
If no: list only the missing critical fields as questions"""

    response = call_llm(system, user, max_tokens=200)

    # Remove <think> tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # If response contains JSON characters it repeated the config — treat as COMPLETE
    if '{' in response or '"event_name"' in response:
        return []

    if "COMPLETE" in response.upper():
        return []

    # Parse numbered questions
    lines = response.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        if line and "COMPLETE" not in line.upper() and '{' not in line:
            cleaned = re.sub(r'^[\d]+[.)]\s*', '', line).strip()
            if cleaned and len(cleaned) > 10:  # ignore very short lines
                questions.append(cleaned)

    return questions


# ─────────────────────────────────────────────
# LLM CALL 3: Check for contradictions
# ─────────────────────────────────────────────

def check_contradictions(config: dict) -> list:
    """
    Checks the parsed config for contradictions or logical problems.
    Returns list of contradictions found, or empty list if none.
    
    Example contradiction:
    "teams of 3" but "need ML expert, frontend, backend, DevOps" = 4 roles for 3 people
    
    CALLED BY: main.py in POST /api/events/configure
    CALLS: call_llm() from llm_service.py
    """
    
    system = """You are an event configuration reviewer.
Check for logical contradictions or impossible requirements.
If no contradictions: respond with exactly: NO_CONTRADICTIONS
If contradictions found: list them numbered, nothing else."""

    user = f"""Check this event configuration for contradictions:

{json.dumps(config, indent=2)}

Common contradictions to check:
- Scoring weights that don't add up to 100%
- Team size too small for the number of required skill roles
- Stages that reference things not defined elsewhere
- Progression top_n larger than expected number of teams

If no contradictions: NO_CONTRADICTIONS
If contradictions found: numbered list only."""

    response = call_llm(system, user, max_tokens=300)
    
    # Remove <think> tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    if "NO_CONTRADICTIONS" in response.upper():
        return []
    
    lines = response.strip().split("\n")
    contradictions = []
    for line in lines:
        line = line.strip()
        if line and "NO_CONTRADICTIONS" not in line.upper():
            cleaned = re.sub(r'^[\d]+[.)]\s*', '', line).strip()
            if cleaned:
                contradictions.append(cleaned)
    
    return contradictions


# ─────────────────────────────────────────────
# LLM CALL 4: Generate human-readable summary
# Shown to committee to confirm before going live
# ─────────────────────────────────────────────

def generate_config_summary(config: dict) -> str:
    """
    Generates a plain English summary of the parsed configuration
    so the committee can confirm the system understood correctly.
    
    CALLED BY: main.py after config is validated and saved
    CALLS: call_llm() from llm_service.py
    """
    
    system = """You are an event coordinator assistant.
Summarise an event configuration in clear, friendly plain English.
Write it as a confirmation to the committee — "Here is what I understood about your event..."
Keep it to 5-8 bullet points. Be specific with numbers."""

    user = f"""Summarise this event configuration for the committee to confirm:

{json.dumps(config, indent=2)}

Write a confirmation summary starting with:
"Here is what I understood about your event:"
Then bullet points covering: team size and rules, stages/timeline, 
evaluation criteria and weights, how many teams progress, anomaly threshold."""

    response = call_llm(system, user, max_tokens=500)
    
    # Remove <think> tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    
    return response


# ─────────────────────────────────────────────
# COMBINED PIPELINE
# main.py calls only this one function
# It runs all 3 checks and returns a clear status
# ─────────────────────────────────────────────

def process_event_description(description: str) -> dict:
    """
    Full pipeline:
    1. Parse description into JSON config
    2. Find gaps (missing info)
    3. Check contradictions
    4. If clean, generate summary for committee to confirm
    
    Returns:
    {
      "status": "READY" / "INCOMPLETE" / "CONTRADICTIONS",
      "config": {...},           # the parsed config
      "questions": [...],        # gap questions (if INCOMPLETE)
      "contradictions": [...],   # contradictions (if CONTRADICTIONS)
      "summary": "..."           # human readable summary (if READY)
    }
    
    CALLED BY: main.py in POST /api/events/configure
    """
    
    # STEP 1: Parse
    config = parse_event_description(description)
    
    # STEP 2: Find gaps
    gaps = find_config_gaps(config)
    if gaps:
        return {
            "status": "INCOMPLETE",
            "config": config,
            "questions": gaps,
            "contradictions": [],
            "summary": None
        }
    
    # STEP 3: Check contradictions
    contradictions = check_contradictions(config)
    if contradictions:
        return {
            "status": "CONTRADICTIONS",
            "config": config,
            "questions": [],
            "contradictions": contradictions,
            "summary": None
        }
    
    # STEP 4: All good — generate summary
    summary = generate_config_summary(config)
    
    return {
        "status": "READY",
        "config": config,
        "questions": [],
        "contradictions": [],
        "summary": summary
    }