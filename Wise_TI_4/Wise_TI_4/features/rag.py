from llm_service import call_llm
from database import Team, Participant, Score, Anomaly, Communication, Judge, EvaluationGuide, Event

def get_relevant_context(question: str, db, event_id: int, user_context: dict = None):
    """
    Get relevant context based on the user's role.
    Different roles see different data.
    """
    question_lower = question.lower()
    user_role = user_context.get('user_role') if user_context else 'GUEST'
    reference_id = user_context.get('reference_id') if user_context else None
    user_email = user_context.get('user_email') if user_context else None
    
    context_parts = []
    
    # Event info (everyone can see basic info)
    event = db.query(Event).filter(Event.id == event_id).first()
    if event:
        context_parts.append(f"Event: {event.name} | Current stage: {event.current_stage}")
    
    # ════════════════════════════════════════════════════
    # ROLE-BASED DATA FILTERING
    # ════════════════════════════════════════════════════
    
    if user_role == 'COMMITTEE':
        # Committee gets full access
        # Teams
        if any(word in question_lower for word in ["team", "group", "member", "composition", "who", "assign"]):
            teams = db.query(Team).filter(Team.event_id == event_id).all()
            for team in teams:
                members = db.query(Participant).filter(Participant.team_id == team.id).all()
                member_names = ", ".join([m.name for m in members])
                context_parts.append(
                    f"Team '{team.name}': members = [{member_names}] | status = {team.status} | rationale = {team.rationale}"
                )
        
        # Participants
        if any(word in question_lower for word in ["participant", "student", "person", "candidate"]):
            participants = db.query(Participant).filter(Participant.event_id == event_id).all()
            for p in participants:
                context_parts.append(
                    f"Participant: {p.name} | Email: {p.email} | Skills: {p.skills} | Institution: {p.institution} | Team: {p.team_id}"
                )
                
        # Scores
        if any(word in question_lower for word in ["score", "result", "evaluat", "rank", "leaderboard", "point", "judge"]):
            scores = db.query(Score).filter(Score.event_id == event_id).all()
            judges = {j.id: j.name for j in db.query(Judge).filter(Judge.event_id == event_id).all()}
            teams = {t.id: t.name for t in db.query(Team).filter(Team.event_id == event_id).all()}
            for s in scores:
                context_parts.append(
                    f"Score: Judge '{judges.get(s.judge_id, 'Unknown')}' gave Team '{teams.get(s.team_id, 'Unknown')}' a score of {s.score}"
                )
                
        # Anomalies
        if any(word in question_lower for word in ["anomal", "discrepan", "flag", "hold", "diverge", "unusual"]):
            anomalies = db.query(Anomaly).filter(Anomaly.event_id == event_id).all()
            teams = {t.id: t.name for t in db.query(Team).filter(Team.event_id == event_id).all()}
            for a in anomalies:
                context_parts.append(
                    f"Anomaly for Team '{teams.get(a.team_id, 'Unknown')}': status = {a.status} | results held = {a.results_held} | explanation = {a.explanation}"
                )
                
        # Communications
        if any(word in question_lower for word in ["email", "communicat", "sent", "pending", "notif", "message"]):
            comms = db.query(Communication).filter(Communication.event_id == event_id).all()
            for c in comms:
                context_parts.append(
                    f"Communication: type = {c.type} | recipient_type = {c.recipient_type} | status = {c.status} | created = {c.created_at}"
                )
                
        # Pending approvals
        if any(word in question_lower for word in ["pending", "approv", "wait", "review"]):
            pending_teams = db.query(Team).filter(Team.event_id == event_id, Team.status == "PENDING_APPROVAL").count()
            pending_comms = db.query(Communication).filter(Communication.event_id == event_id, Communication.status == "DRAFT").count()
            pending_anomalies = db.query(Anomaly).filter(Anomaly.event_id == event_id, Anomaly.status == "PENDING_REVIEW").count()
            context_parts.append(
                f"Pending approvals: {pending_teams} teams, {pending_comms} communications, {pending_anomalies} anomalies"
            )
    
    elif user_role == 'JUDGE':
        # Judge sees ONLY their own assigned teams and scores
        
        # Their scores
        if any(word in question_lower for word in ["score", "submission", "my"]):
            my_scores = db.query(Score).filter(
                Score.judge_id == reference_id,
                Score.event_id == event_id
            ).all()
            teams = {t.id: t.name for t in db.query(Team).filter(Team.event_id == event_id).all()}
            for s in my_scores:
                context_parts.append(
                    f"Your score for '{teams.get(s.team_id, 'Unknown')}': {s.score}"
                )
        
        # Their assigned teams (via guides)
        my_guides = db.query(EvaluationGuide).filter(
            EvaluationGuide.judge_id == reference_id,
            EvaluationGuide.event_id == event_id
        ).all()
        teams_map = {t.id: t.name for t in db.query(Team).filter(Team.event_id == event_id).all()}
        my_team_ids = list(set([g.team_id for g in my_guides]))
        
        if any(word in question_lower for word in ["team", "roster", "member", "assigned"]):
            for team_id in my_team_ids:
                team = db.query(Team).filter(Team.id == team_id).first()
                if team:
                    members = db.query(Participant).filter(Participant.team_id == team.id).all()
                    member_names = ", ".join([m.name for m in members])
                    context_parts.append(
                        f"Your assigned team '{team.name}': members = [{member_names}]"
                    )
        
        # Public event info
        if any(word in question_lower for word in ["event", "deadline", "when", "schedule"]):
            if event:
                context_parts.append(f"Event deadline: {event.name} - check key dates for specific times")
    
    elif user_role == 'PARTICIPANT':
        # Participant sees ONLY their own team's info
        
        # Get their team
        my_team = None
        if reference_id:
            me = db.query(Participant).filter(Participant.id == reference_id).first()
            if me and me.team_id:
                my_team = db.query(Team).filter(Team.id == me.team_id).first()
        
        # Their team info
        if my_team and any(word in question_lower for word in ["team", "my", "teammate", "member"]):
            teammates = db.query(Participant).filter(
                Participant.team_id == my_team.id,
                Participant.id != reference_id
            ).all()
            teammate_info = ", ".join([
                f"{t.name} ({t.skills})" for t in teammates
            ])
            context_parts.append(
                f"Your team '{my_team.name}': You are on this team. Teammates: [{teammate_info}]. Status: {my_team.status}"
            )
        
        # Their team's score (only if published - i.e., event is in RESULTS stage)
        if my_team and any(word in question_lower for word in ["score", "result", "grade"]):
            if event and event.current_stage == 'RESULTS':
                team_scores = db.query(Score).filter(Score.team_id == my_team.id).all()
                if team_scores:
                    avg = sum(s.score for s in team_scores) / len(team_scores)
                    context_parts.append(
                        f"Your team's average score: {avg:.1f}/100 (from {len(team_scores)} judges)"
                    )
            else:
                context_parts.append("Scores are not yet published. They will be available after the evaluation phase.")
        
        # Their own info
        if reference_id and any(word in question_lower for word in ["me", "my name", "profile", "who am i"]):
            me = db.query(Participant).filter(Participant.id == reference_id).first()
            if me:
                context_parts.append(
                    f"You are {me.name} from {me.institution}. Skills: {me.skills}. Experience: {me.experience} years."
                )
        
        # Public event info
        if any(word in question_lower for word in ["event", "deadline", "when", "schedule"]):
            if event:
                context_parts.append(f"Event: {event.name} - Current stage: {event.current_stage}")
    
    else:
        # GUEST or unknown - only public event info
        pass
    
    return "\n".join(context_parts) if context_parts else "No relevant information found for your query."


def answer_question(question: str, db, event_id: int, user_context: dict = None):
    """
    Answer a question with role-based access control.
    The LLM only sees data the user is allowed to access.
    """
    # Get user role
    user_role = user_context.get('user_role', 'GUEST') if user_context else 'GUEST'
    
    # Get context (filtered by role)
    context = get_relevant_context(question, db, event_id, user_context)
    
    # Build role-specific system prompt
    if user_role == 'COMMITTEE':
        access_msg = "You have full access to all event data."
    elif user_role == 'JUDGE':
        access_msg = "You can only see information about teams you are assigned to evaluate and your own scores."
    elif user_role == 'PARTICIPANT':
        access_msg = "You can ONLY see information about YOUR OWN team. You CANNOT see other teams' scores, members, or details."
    else:
        access_msg = "You have limited access to public event information only."
    
    system = f"""You are an intelligent event management assistant for a hackathon organizing system.
The current user is a {user_role}.
{access_msg}

CRITICAL RULES:
1. Answer ONLY using the context provided below
2. If the user asks about information they shouldn't access, say: "I can only help you with information relevant to your role."
3. Do NOT make up information
4. Do NOT share other users' private data (emails, scores, etc.)
5. If you don't know, say "I don't have that information" clearly
6. Be concise and helpful
7. Never reveal that information exists elsewhere in the system
8. Redirect to what the user CAN see if they ask about restricted data"""
    
    user_message = f"""Context (filtered for your role):
{context}

Question: {question}

Answer based ONLY on the context above. If the question is about information you cannot access, politely redirect."""
    
    response = call_llm(system, user_message, max_tokens=500)
    return response
