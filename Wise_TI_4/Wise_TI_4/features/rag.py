from llm_service import call_llm

# ─────────────────────────────────────────────
# RAG: Committee Q&A
#
# Flow:
# 1. Committee types a question
# 2. We search the DB for relevant info (retrieval)
# 3. We inject that info into the LLM prompt (augmented)
# 4. LLM answers using that context (generation)
#
# CALLED BY: main.py in POST /api/committee/ask
# ─────────────────────────────────────────────

def get_relevant_context(question: str, db, event_id: int) -> str:
    """
    Searches the database for information relevant to the question.
    Uses keyword matching — simple but effective for a hackathon.
    
    Returns: a block of text with all relevant data
             this gets injected into the LLM prompt
    
    CALLED BY: answer_question() below
    READS FROM: all database tables via the db session
    """
    
    from database import Team, Participant, Score, Anomaly, Communication, Judge, EvaluationGuide, Event
    
    question_lower = question.lower()
    context_parts = []
    
    # ── EVENT INFO ──
    # Always include basic event info
    event = db.query(Event).filter(Event.id == event_id).first()
    if event:
        context_parts.append(f"Event: {event.name} | Current stage: {event.current_stage} | Team size: {event.team_size}")
    
    # ── TEAM INFO ──
    # Include if question mentions teams, groups, members, composition
    if any(word in question_lower for word in ["team", "group", "member", "composition", "who", "assign"]):
        teams = db.query(Team).filter(Team.event_id == event_id).all()
        for team in teams:
            members = db.query(Participant).filter(Participant.team_id == team.id).all()
            member_names = ", ".join([m.name for m in members])
            context_parts.append(
                f"Team '{team.name}': members = [{member_names}] | status = {team.status} | rationale = {team.rationale}"
            )
    
    # ── PARTICIPANT INFO ──
    # Include if question mentions specific participants
    if any(word in question_lower for word in ["participant", "student", "person", "candidate"]):
        participants = db.query(Participant).filter(Participant.event_id == event_id).all()
        for p in participants:
            context_parts.append(
                f"Participant: {p.name} | Skills: {p.skills} | Institution: {p.institution} | Team ID: {p.team_id}"
            )
    
    # ── SCORE INFO ──
    # Include if question mentions scores, evaluation, results, leaderboard
    if any(word in question_lower for word in ["score", "result", "evaluat", "rank", "leaderboard", "point", "judge"]):
        scores = db.query(Score).filter(Score.event_id == event_id).all()
        judges = {j.id: j.name for j in db.query(Judge).filter(Judge.event_id == event_id).all()}
        teams  = {t.id: t.name for t in db.query(Team).filter(Team.event_id == event_id).all()}
        for s in scores:
            context_parts.append(
                f"Score: Judge '{judges.get(s.judge_id, 'Unknown')}' gave Team '{teams.get(s.team_id, 'Unknown')}' a score of {s.score}"
            )
    
    # ── ANOMALY INFO ──
    # Include if question mentions anomaly, discrepancy, flag, hold, diverge
    if any(word in question_lower for word in ["anomal", "discrepan", "flag", "hold", "diverge", "unusual"]):
        anomalies = db.query(Anomaly).filter(Anomaly.event_id == event_id).all()
        teams = {t.id: t.name for t in db.query(Team).filter(Team.event_id == event_id).all()}
        for a in anomalies:
            context_parts.append(
                f"Anomaly for Team '{teams.get(a.team_id, 'Unknown')}': status = {a.status} | results held = {a.results_held} | explanation = {a.explanation}"
            )
    
    # ── COMMUNICATION INFO ──
    # Include if question mentions email, communication, sent, pending
    if any(word in question_lower for word in ["email", "communicat", "sent", "pending", "notif", "message"]):
        comms = db.query(Communication).filter(Communication.event_id == event_id).all()
        for c in comms:
            context_parts.append(
                f"Communication: type = {c.type} | recipient_type = {c.recipient_type} | status = {c.status} | created = {c.created_at}"
            )
    
    # ── PENDING APPROVALS ──
    # Always include if question mentions pending, approve, waiting
    if any(word in question_lower for word in ["pending", "approv", "wait", "review"]):
        pending_teams = db.query(Team).filter(Team.event_id == event_id, Team.status == "PENDING_APPROVAL").count()
        pending_comms = db.query(Communication).filter(Communication.event_id == event_id, Communication.status == "DRAFT").count()
        pending_anomalies = db.query(Anomaly).filter(Anomaly.event_id == event_id, Anomaly.status == "PENDING_REVIEW").count()
        context_parts.append(
            f"Pending approvals: {pending_teams} teams, {pending_comms} communications, {pending_anomalies} anomalies"
        )
    
    # If nothing matched, return all basic info
    if len(context_parts) <= 1:
        teams = db.query(Team).filter(Team.event_id == event_id).all()
        context_parts.append(f"Total teams formed: {len(teams)}")
        participants = db.query(Participant).filter(Participant.event_id == event_id).all()
        context_parts.append(f"Total participants: {len(participants)}")
    
    return "\n".join(context_parts)


def answer_question(question: str, db, event_id: int) -> str:
    """
    Full RAG pipeline for committee Q&A.
    
    1. Retrieves relevant context from DB
    2. Injects it into LLM prompt
    3. Returns LLM's answer
    
    CALLED BY: main.py in POST /api/committee/ask
    CALLS: get_relevant_context() + call_llm()
    """
    
    # STEP 1: RETRIEVAL — find relevant data
    context = get_relevant_context(question, db, event_id)
    
    # STEP 2: AUGMENTED GENERATION — inject context into prompt
    system = """You are an intelligent event management assistant for a hackathon organizing committee.
You answer questions about the event using ONLY the context provided below.
Be concise, direct, and helpful.
If the context doesn't contain enough information to answer, say so clearly.
Never make up information that isn't in the context."""
    
    user = f"""Context from the event database:
{context}

Committee question: {question}

Answer based only on the context above."""
    
    # STEP 3: LLM answers using the context
    return call_llm(system, user, max_tokens=500)
