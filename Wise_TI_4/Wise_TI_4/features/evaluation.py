from llm_service import call_llm

# ─────────────────────────────────────────────
# PART 1: EVALUATION GUIDE GENERATION
# ─────────────────────────────────────────────

def generate_evaluation_guide(
    judge_name: str,
    team_name: str,
    project_description: str,
    team_skills: list,          # ["ML", "Backend", "Frontend"]
    criteria: list,             # [{"name": "Innovation", "weight": 30}, ...]
    minutes_per_team: int
) -> str:
    """
    Generates a custom evaluation guide for ONE judge evaluating ONE team.
    Each judge gets a different guide tailored to the specific team's project.
    
    Returns: markdown-formatted guide as plain string
    
    CALLED BY: main.py in POST /api/evaluation/start
               (called once per judge-team assignment)
    CALLS: call_llm() from llm_service.py
    """
    
    criteria_text = "\n".join([f"- {c['name']}: {c['weight']} points" for c in criteria])
    skills_text = ", ".join(team_skills)
    
    system = """You are a hackathon judging coordinator. 
Generate structured, practical evaluation guides that help judges score consistently.
Use clear headers and bullet points."""
    
    user = f"""Generate a complete evaluation guide for:

Judge: {judge_name}
Team: {team_name}
Project: {project_description}
Team skills: {skills_text}
Time available: {minutes_per_team} minutes per team

Scoring criteria:
{criteria_text}

Create a guide with these exact sections:

1. WHAT TO LOOK FOR
For each criterion, list 3-4 specific things to observe.
Format: "Innovation: Look for X, Y, Z"

2. QUESTIONS TO ASK  
Three questions tailored to this specific team's project.
Format: "1. Question one?"
       "2. Question two?"
       "3. Question three?"

3. RED FLAGS
Three warning signs specific to this type of project.
Format: "- Red flag one"
       "- Red flag two"
       "- Red flag three"

4. SCORING DISTRIBUTION
Show how to distribute points within each criterion.
Format: "Innovation (40 points): 10-15 for X, 15-25 for Y, 25-40 for Z"

IMPORTANT: Use plain text only. No markdown tables, no asterisks for bold, 
no hash headers, no pipe characters. Just clean readable text with dashes and numbers."""
    
    # Returns markdown text saved to evaluation_guides table
    # Shown to judge when they open their evaluation portal
    return call_llm(system, user, max_tokens=900)


# ─────────────────────────────────────────────
# PART 2: ANOMALY DETECTION
# Pure math — no LLM
# ─────────────────────────────────────────────

def detect_anomaly(scores: list, threshold: float) -> dict:
    """
    Checks if any judge's score deviates too far from the average.
    
    scores    = [{"judge_name": "Dr. Mehta", "judge_id": 1, "score": 85}, ...]
    threshold = maximum allowed deviation (set by committee in event config)
    
    Returns: {
        "is_anomaly": bool,
        "average": float,
        "flagged": [{"judge_name": ..., "score": ..., "deviation": ...}]
    }
    
    CALLED BY: detect_and_explain_anomaly() below
    NO LLM USED HERE — pure arithmetic
    """
    
    if not scores:
        return {"is_anomaly": False, "average": 0, "flagged": []}
    
    values = [s["score"] for s in scores]
    average = sum(values) / len(values)
    
    flagged = []
    for s in scores:
        deviation = abs(s["score"] - average)
        if deviation > threshold:
            flagged.append({
                "judge_name": s["judge_name"],
                "judge_id":   s["judge_id"],
                "score":      s["score"],
                "deviation":  round(deviation, 2)
            })
    
    return {
        "is_anomaly": len(flagged) > 0,
        "average":    round(average, 2),
        "flagged":    flagged
    }


# ─────────────────────────────────────────────
# PART 3: ANOMALY EXPLANATION
# LLM explains what the math found
# ─────────────────────────────────────────────

def explain_anomaly(
    team_name: str,
    scores: list,
    average: float,
    flagged: list
) -> str:
    """
    After detect_anomaly() finds a problem, this explains it in plain English.
    
    Returns: explanation string saved to anomalies table
             shown on committee dashboard with approve/resolve buttons
    
    CALLED BY: detect_and_explain_anomaly() below
    CALLS: call_llm() from llm_service.py
    """
    
    scores_text = "\n".join([f"- {s['judge_name']}: {s['score']}/100" for s in scores])
    flagged_text = "\n".join([f"- {f['judge_name']}: {f['score']} (deviation: {f['deviation']} points from average)" for f in flagged])
    
    system = """You are an event management assistant. 
Write neutral, factual anomaly summaries for committee review.
Never assume bad intent. Be concise and helpful."""
    
    user = f"""A score anomaly was detected for {team_name}.

All scores received:
{scores_text}

Panel average: {average}

Scores that exceeded the anomaly threshold:
{flagged_text}

Please write:
1. A 2-3 sentence neutral summary of the anomaly
2. Three possible reasons this divergence may have occurred (do not assume bad faith)
3. A recommended action for the committee"""
    
    return call_llm(system, user, max_tokens=400)


# ─────────────────────────────────────────────
# COMBINED FUNCTION
# main.py calls only this one — it handles everything
# ─────────────────────────────────────────────

def detect_and_explain_anomaly(
    team_name: str,
    scores: list,
    threshold: float
) -> dict:
    """
    Full anomaly pipeline. Detects mathematically, then explains with LLM if needed.
    
    Returns: {
        "is_anomaly": bool,
        "average": float,
        "flagged": list,
        "explanation": str or None
    }
    
    CALLED BY: main.py in POST /api/scores/submit
               (called every time a score is submitted and all judges have scored)
    """
    
    result = detect_anomaly(scores, threshold)
    
    if result["is_anomaly"]:
        explanation = explain_anomaly(
            team_name=team_name,
            scores=scores,
            average=result["average"],
            flagged=result["flagged"]
        )
        result["explanation"] = explanation
    else:
        result["explanation"] = None
    
    return result
