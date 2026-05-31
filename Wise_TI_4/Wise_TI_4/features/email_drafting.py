import json
import re
from llm_service import call_llm

# ─────────────────────────────────────────────
# HELPER: Safe JSON Parser
# Smaller LLMs sometimes add extra text around JSON.
# This extracts it safely.
# ─────────────────────────────────────────────

def safe_parse_json(text: str) -> dict:
    """
    Tries to extract JSON from LLM response even if model adds extra text.
    Qwen3 sometimes adds <think> tags or extra explanation — this handles that.
    """
    import re
    
    # Remove <think>...</think> blocks that Qwen3 sometimes adds
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
    
    # Complete fallback — use raw text as email body
    return {
        "subject": "Event Update",
        "body": text.strip()
    }
# ─────────────────────────────────────────────
# EMAIL 1: Team Assignment Welcome
# Sent to each participant after team approval
# ─────────────────────────────────────────────

def draft_team_assignment_email(
    participant_name: str,
    team_name: str,
    teammates: list,        # list of {"name": str, "skills": str}
    problem_statement: str,
    start_date: str,
    deadline: str
) -> dict:
    """
    Drafts a welcome email for a participant after they've been assigned a team.
    
    Returns: {"subject": "...", "body": "..."}
    
    CALLED BY: main.py in POST /api/teams/{team_id}/approve
               (triggered when committee approves team compositions)
    CALLS: call_llm() from llm_service.py
    """
    
    teammates_text = ", ".join([f"{t['name']} ({t['skills']})" for t in teammates])
    
    system = """You are an event coordinator. Draft warm, professional hackathon emails.
IMPORTANT: Respond ONLY with a JSON object. No explanation, no markdown.
Format: {"subject": "email subject here", "body": "full email body here"}"""
    
    user = f"""Draft a welcome + team assignment email for:
- Participant: {participant_name}
- Team: {team_name}
- Teammates: {teammates_text}
- Problem statement: {problem_statement}
- Event starts: {start_date}
- Submission deadline: {deadline}

The email should be warm, encouraging, and include all key details.
Return ONLY JSON: {{"subject": "...", "body": "..."}}"""
    
    response = call_llm(system, user, max_tokens=600)
    return safe_parse_json(response)


# ─────────────────────────────────────────────
# EMAIL 2: Evaluation Request to Judge
# Sent to judges when evaluation phase begins
# ─────────────────────────────────────────────

def draft_evaluation_request_email(
    judge_name: str,
    teams_to_evaluate: list,   # list of team name strings
    eval_deadline: str,
    criteria: list,            # list of {"name": str, "weight": int}
    portal_link: str
) -> dict:
    """
    Drafts an email asking a judge to evaluate their assigned teams.
    
    Returns: {"subject": "...", "body": "..."}
    
    CALLED BY: main.py in POST /api/evaluation/start
               (triggered when committee starts evaluation phase)
    CALLS: call_llm() from llm_service.py
    """
    
    teams_text = ", ".join(teams_to_evaluate)
    criteria_text = "\n".join([f"- {c['name']}: {c['weight']} points" for c in criteria])
    
    system = """You are an event coordinator. Draft professional judge notification emails.
IMPORTANT: Respond ONLY with a JSON object. No explanation, no markdown.
Format: {"subject": "email subject here", "body": "full email body here"}"""
    
    user = f"""Draft an evaluation request email for:
- Judge: {judge_name}
- Teams to evaluate: {teams_text}
- Evaluation deadline: {eval_deadline}
- Scoring criteria:
{criteria_text}
- Portal link: {portal_link}

Be professional and clear about responsibilities and deadline.
Return ONLY JSON: {{"subject": "...", "body": "..."}}"""
    
    response = call_llm(system, user, max_tokens=600)
    return safe_parse_json(response)


# ─────────────────────────────────────────────
# EMAIL 3: Deadline Reminder
# Sent automatically X hours before deadline
# ─────────────────────────────────────────────

def draft_deadline_reminder_email(
    recipient_name: str,
    recipient_type: str,    # "participant" or "judge"
    team_name: str,
    deadline: str,
    hours_remaining: int
) -> dict:
    """
    Drafts a deadline reminder email.
    
    Returns: {"subject": "...", "body": "..."}
    
    CALLED BY: main.py in POST /api/communications/send-reminders
               (can be triggered manually or on a schedule)
    CALLS: call_llm() from llm_service.py
    """
    
    system = """You are an event coordinator. Draft concise deadline reminder emails.
IMPORTANT: Respond ONLY with a JSON object. No explanation, no markdown.
Format: {"subject": "email subject here", "body": "full email body here"}"""
    
    user = f"""Draft a deadline reminder email for:
- Name: {recipient_name}
- Role: {recipient_type}
- Team/context: {team_name}
- Deadline: {deadline}
- Time remaining: {hours_remaining} hours

Be friendly but create appropriate urgency. Keep it brief.
Return ONLY JSON: {{"subject": "...", "body": "..."}}"""
    
    response = call_llm(system, user, max_tokens=400)
    return safe_parse_json(response)


# ─────────────────────────────────────────────
# EMAIL 4: Results Notification
# Sent after scores are finalized and approved
# ─────────────────────────────────────────────

def draft_results_email(
    participant_name: str,
    team_name: str,
    score: float,
    rank: int,
    qualified: bool,
    next_round_date: str = None,
    confirmation_link: str = None
) -> dict:
    """
    Drafts a results email for a participant.
    Different content depending on whether they qualified.
    
    Returns: {"subject": "...", "body": "..."}
    
    CALLED BY: main.py in POST /api/results/publish
               (triggered after committee approves final results)
    CALLS: call_llm() from llm_service.py
    """
    
    if qualified:
        outcome_text = f"qualified for the next round (Rank #{rank})"
        extra = f"Next round: {next_round_date}\nConfirmation link: {confirmation_link}"
    else:
        outcome_text = f"not selected for the next round (Rank #{rank})"
        extra = "Encourage them and thank them for participating."
    
    system = """You are an event coordinator. Draft empathetic, professional results emails.
IMPORTANT: Respond ONLY with a JSON object. No explanation, no markdown.
Format: {"subject": "email subject here", "body": "full email body here"}"""
    
    user = f"""Draft a results email for:
- Participant: {participant_name}
- Team: {team_name}
- Score: {score}/100
- Outcome: {outcome_text}
- Additional context: {extra}

Be warm and genuine regardless of outcome.
Return ONLY JSON: {{"subject": "...", "body": "..."}}"""
    
    response = call_llm(system, user, max_tokens=600)
    return safe_parse_json(response)
