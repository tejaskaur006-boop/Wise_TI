from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from features.event_config_parser import process_event_description

# Import database
from database import (
    get_db, create_tables,
    Event, Participant, Team, Judge, Score, Anomaly, Communication, EvaluationGuide,Approval,
    get_all_teams, get_team_members, get_scores_for_team,
    get_pending_approvals, get_leaderboard, get_activity_log
)

# Import all LLM features
from features.team_formation  import form_teams, generate_team_rationale
from features.email_drafting  import (
    draft_team_assignment_email,
    draft_evaluation_request_email,
    draft_deadline_reminder_email,
    draft_results_email
)
from features.evaluation import generate_evaluation_guide, detect_and_explain_anomaly
from features.rag        import answer_question

app = FastAPI(title="EventFlow - Intelligent Event Orchestration System")

# Allow frontend (React) to talk to this backend
# When your frontend team builds the UI, this prevents CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Create all database tables when app starts
create_tables()


# ─────────────────────────────────────────────
# REQUEST BODY MODELS
# These define what JSON the frontend sends to each endpoint
# ─────────────────────────────────────────────

class ConfigureEventRequest(BaseModel):
    description: str
    event_id: int

class ClarifyConfigRequest(BaseModel):
    event_id: int
    original_description: str
    answers: str

class ConfirmConfigRequest(BaseModel):
    event_id: int
    confirmed: bool

class CreateEventRequest(BaseModel):
    name: str
    team_size: int = 3
    skill_balance: bool = True
    no_same_institution: bool = True
    anomaly_threshold: float = 20.0

class AddJudgeRequest(BaseModel):
    name: str
    email: str

class SubmitScoreRequest(BaseModel):
    judge_id: int
    team_id: int
    score: float
    criteria_scores: Optional[dict] = {}

class StartEvaluationRequest(BaseModel):
    criteria: list            # [{"name": "Innovation", "weight": 30}]
    minutes_per_team: int = 10

class AskQuestionRequest(BaseModel):
    question: str

class ResolveAnomalyRequest(BaseModel):
    action: str               # "use_average", "accept_flagged", "request_reevaluation"


# ─────────────────────────────────────────────
# SECTION 1: EVENT SETUP
# ─────────────────────────────────────────────

@app.post("/api/events")
def create_event(request: CreateEventRequest, db: Session = Depends(get_db)):
    """
    Creates a new event.
    
    FRONTEND: Called when committee fills out event setup form and clicks "Create Event"
    LLM USED: No
    DB: Writes to events table
    """
    event = Event(
        name=request.name,
        team_size=request.team_size,
        skill_balance=request.skill_balance,
        no_same_institution=request.no_same_institution,
        anomaly_threshold=request.anomaly_threshold,
        current_stage="PARTICIPANT_INTAKE"
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return {"event_id": event.id, "message": f"Event '{event.name}' created successfully"}


@app.get("/api/events/{event_id}")
def get_event(event_id: int, db: Session = Depends(get_db)):
    """
    Gets event details and configuration.
    
    FRONTEND: Called when committee dashboard loads to show current event state
    LLM USED: No
    DB: Reads from events table
    """
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return {
        "id": event.id,
        "name": event.name,
        "current_stage": event.current_stage,
        "team_size": event.team_size,
        "anomaly_threshold": event.anomaly_threshold
    }


# ─────────────────────────────────────────────
# SECTION 2: PARTICIPANT INTAKE
# ─────────────────────────────────────────────

@app.post("/api/events/{event_id}/participants/upload")
async def upload_participants(event_id: int, db: Session = Depends(get_db)):
    """
    Loads participants from a CSV file into the database.
    
    FRONTEND: Called when committee uploads participant CSV
              The CSV should have columns: name, email, skills, institution, experience
    LLM USED: No
    DB: Writes to participants table
    
    CSV FORMAT EXPECTED:
    name,email,skills,institution,experience
    Rahul Sharma,rahul@email.com,"ML,Python",IIT Delhi,3
    Priya Singh,priya@email.com,"Frontend,React",BITS Pilani,2
    """
    
    # NOTE FOR FRONTEND TEAM:
    # This endpoint needs multipart/form-data with a "file" field
    # For now, we load from a fixed path for testing
    # Frontend will send: FormData with file attached
    
    try:
        df = pd.read_csv("sample_participants.csv")
        
        count = 0
        for _, row in df.iterrows():
            participant = Participant(
                event_id=event_id,
                name=row["name"],
                email=row["email"],
                skills=row["skills"],
                institution=row["institution"],
                experience=int(row["experience"])
            )
            db.add(participant)
            count += 1
        
        db.commit()
        
        # Update event stage
        event = db.query(Event).filter(Event.id == event_id).first()
        event.current_stage = "TEAM_FORMATION"
        db.commit()
        
        return {"message": f"{count} participants loaded successfully", "count": count}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/events/{event_id}/participants")
def list_participants(event_id: int, db: Session = Depends(get_db)):
    """
    Returns all participants for an event.
    
    FRONTEND: Called to populate the participant list on the dashboard
    LLM USED: No
    DB: Reads from participants table
    """
    participants = db.query(Participant).filter(Participant.event_id == event_id).all()
    return [{
        "id": p.id,
        "name": p.name,
        "email": p.email,
        "skills": p.skills,
        "institution": p.institution,
        "experience": p.experience,
        "team_id": p.team_id
    } for p in participants]


# ─────────────────────────────────────────────
# SECTION 3: TEAM FORMATION
# ─────────────────────────────────────────────

@app.post("/api/events/{event_id}/teams/form")
def form_teams_endpoint(event_id: int, db: Session = Depends(get_db)):
    import traceback  # LINE 1 ADDED
    try:              # LINE 2 ADDED — everything below is just indented one level

        event = db.query(Event).filter(Event.id == event_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        participants = db.query(Participant).filter(
            Participant.event_id == event_id,
            Participant.team_id == None
        ).all()
        
        if not participants:
            raise HTTPException(status_code=400, detail="No unassigned participants found")
        
        rules = {
            "team_size": event.team_size,
            "skill_balance": event.skill_balance,
            "no_same_institution": event.no_same_institution
        }
        formed_teams = form_teams(participants, event.team_size, event.no_same_institution)
        
        saved_teams = []
        
        for team_data in formed_teams:
            team = Team(
                event_id=event_id,
                name=team_data["name"],
                status="PENDING_APPROVAL"
            )
            db.add(team)
            db.commit()
            db.refresh(team)
            
            members = team_data["members"]
            for member in members:
                participant = db.query(Participant).filter(Participant.id == member.id).first()
                participant.team_id = team.id
            db.commit()
            
            rationale = generate_team_rationale(
                team_name=team_data["name"],
                members=members,
                rules=rules
            )
            
            team.rationale = rationale
            db.commit()
            
            saved_teams.append({
                "team_id": team.id,
                "team_name": team.name,
                "rationale": rationale,
                "status": "PENDING_APPROVAL",
                "members": [{"name": m.name, "skills": m.skills} for m in members]
            })
        
        return {
            "message": f"{len(saved_teams)} teams formed, awaiting committee approval",
            "teams": saved_teams
        }

    except HTTPException:
        raise  # let FastAPI handle 404/400 normally
    except Exception as e:
        traceback.print_exc()  # LINE 3 ADDED — prints exact error in terminal
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/events/{event_id}/teams")
def list_teams(event_id: int, db: Session = Depends(get_db)):
    """
    Returns all teams for an event with their members and rationales.
    
    FRONTEND: Called to populate the teams section of the committee dashboard
    LLM USED: No
    DB: Reads from teams + participants tables
    """
    teams = db.query(Team).filter(Team.event_id == event_id).all()
    result = []
    for team in teams:
        members = db.query(Participant).filter(Participant.team_id == team.id).all()
        result.append({
            "id": team.id,
            "name": team.name,
            "status": team.status,
            "rationale": team.rationale,
            "members": [{"id": m.id, "name": m.name, "skills": m.skills, "institution": m.institution} for m in members]
        })
    return result


@app.post("/api/teams/{team_id}/approve")
def approve_team(team_id: int, db: Session = Depends(get_db)):
    """
    Committee approves a team composition.
    After approval, drafts the welcome email for each team member.
    
    FRONTEND: Called when committee clicks "Approve" on a team card
    LLM USED: YES — draft_team_assignment_email() called for each member
    DB: Updates team.status, writes to communications table
    
    APPROVAL GATE: Emails saved as DRAFT — another approval needed before sending
    """
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Update team status
    team.status = "APPROVED"
    approval = Approval(
    event_id=team.event_id,
    team_id=team.id,
    approval_type="TEAM_FORMATION",
    status="APPROVED",
    approved_by="Committee",
    remarks="Team approved after review"
)

    db.add(approval)
    # db.commit()
    
    # Get event and members
    event = db.query(Event).filter(Event.id == team.event_id).first()
    members = db.query(Participant).filter(Participant.team_id == team_id).all()
    
    # Draft a welcome email for each member (LLM)
    emails_drafted = 0
    for member in members:
        teammates = [
            {"name": m.name, "skills": m.skills}
            for m in members if m.id != member.id
        ]
        
        email = draft_team_assignment_email(
            participant_name=member.name,
            team_name=team.name,
            teammates=teammates,
            problem_statement="IoT Smart Energy Monitoring",  # update with real problem
            start_date="June 15, 2025",
            deadline="June 16, 2025 at 6:00 PM"
        )
        
        comm = Communication(
            event_id=team.event_id,
            recipient_id=member.id,
            recipient_type="PARTICIPANT",
            type="TEAM_ASSIGNMENT",
            subject=email["subject"],
            body=email["body"],
            status="DRAFT"
        )
        db.add(comm)
        emails_drafted += 1
    
    db.commit()
    return {"message": f"Team approved. {emails_drafted} welcome emails drafted, awaiting send approval"}


@app.post("/api/teams/{team_id}/reject")
def reject_team(team_id: int, db: Session = Depends(get_db)):
    """
    Committee rejects a team composition. Teams go back to unassigned.
    
    FRONTEND: Called when committee clicks "Reject" on a team card
    LLM USED: No
    DB: Updates team status, clears participant.team_id
    """
    team = db.query(Team).filter(Team.id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Unassign all members
    db.query(Participant).filter(Participant.team_id == team_id).update({"team_id": None})
    team.status = "REJECTED"
    approval = Approval(
    event_id=team.event_id,
    team_id=team.id,
    approval_type="TEAM_FORMATION",
    status="REJECTED",
    approved_by="Committee",
    remarks="Team rejected by committee"
)

    db.add(approval)
    db.commit()
    
    return {"message": "Team rejected. Members returned to unassigned pool."}



@app.get("/api/events/{event_id}/approvals")
def get_approvals(event_id: int, db: Session = Depends(get_db)):

    approvals = db.query(Approval).filter(
        Approval.event_id == event_id
    ).all()

    return [{
        "team_id": a.team_id,
        "approval_type": a.approval_type,
        "status": a.status,
        "approved_by": a.approved_by,
        "remarks": a.remarks,
        "created_at": str(a.created_at)
    } for a in approvals]

# ─────────────────────────────────────────────
# SECTION 4: COMMUNICATIONS
# ─────────────────────────────────────────────

@app.get("/api/events/{event_id}/communications")
def list_communications(event_id: int, db: Session = Depends(get_db)):
    """
    Returns all drafted emails for preview on the dashboard.
    
    FRONTEND: Called to show the email preview section where committee approves emails
    LLM USED: No
    DB: Reads from communications table
    """
    comms = db.query(Communication).filter(Communication.event_id == event_id).all()
    return [{
        "id": c.id,
        "type": c.type,
        "recipient_type": c.recipient_type,
        "subject": c.subject,
        "body": c.body,
        "status": c.status,
        "created_at": str(c.created_at)
    } for c in comms]


@app.post("/api/communications/{comm_id}/approve")
def approve_communication(comm_id: int, db: Session = Depends(get_db)):
    """
    Approves an email draft and marks it as SENT.
    In production, this would actually call SendGrid/email service here.
    
    FRONTEND: Called when committee clicks "Send" on an email preview
    LLM USED: No
    DB: Updates communication.status to SENT
    
    NOTE FOR YOUR TEAM: 
    To actually send emails, add SendGrid here:
    pip install sendgrid
    Then call sendgrid_client.send(message) before updating status
    """
    from datetime import datetime
    comm = db.query(Communication).filter(Communication.id == comm_id).first()
    if not comm:
        raise HTTPException(status_code=404, detail="Communication not found")
    
    comm.status = "SENT"
    comm.sent_at = datetime.utcnow()
    db.commit()
    
    return {"message": f"Email marked as sent to recipient {comm.recipient_id}"}


# ─────────────────────────────────────────────
# SECTION 5: JUDGES
# ─────────────────────────────────────────────

@app.post("/api/events/{event_id}/judges")
def add_judge(event_id: int, request: AddJudgeRequest, db: Session = Depends(get_db)):
    """
    Adds a judge to the event.
    
    FRONTEND: Called from judge management section of dashboard
    LLM USED: No
    DB: Writes to judges table
    """
    judge = Judge(event_id=event_id, name=request.name, email=request.email)
    db.add(judge)
    db.commit()
    db.refresh(judge)
    return {"judge_id": judge.id, "message": f"Judge {judge.name} added"}


@app.get("/api/events/{event_id}/judges")
def list_judges(event_id: int, db: Session = Depends(get_db)):
    """
    Returns all judges for an event.
    
    FRONTEND: Called to populate the judges section
    LLM USED: No
    DB: Reads from judges table
    """
    judges = db.query(Judge).filter(Judge.event_id == event_id).all()
    return [{"id": j.id, "name": j.name, "email": j.email} for j in judges]

@app.get("/api/judges/{judge_id}/evaluated-teams")
def get_evaluated_teams(judge_id: int, db: Session = Depends(get_db)):
    """
    Returns team IDs that this judge has already scored.
    """
    from database import Score
    
    scores = db.query(Score).filter(Score.judge_id == judge_id).all()
    evaluated_team_ids = list(set([s.team_id for s in scores]))
    
    return {
        "judge_id": judge_id,
        "evaluated_team_ids": evaluated_team_ids,
        "count": len(evaluated_team_ids)
    }

# ─────────────────────────────────────────────
# SECTION 6: EVALUATION
# ─────────────────────────────────────────────

@app.post("/api/events/{event_id}/evaluation/start")
def start_evaluation(event_id: int, request: StartEvaluationRequest, db: Session = Depends(get_db)):
    """
    Starts the evaluation phase:
    1. Generates evaluation guides for each judge-team pair (LLM)
    2. Drafts evaluation request emails for judges (LLM)
    
    FRONTEND: Called when committee clicks "Start Evaluation Phase"
    LLM USED: YES — generate_evaluation_guide() and draft_evaluation_request_email()
    DB: Writes to evaluation_guides and communications tables
    """
    judges = db.query(Judge).filter(Judge.event_id == event_id).all()
    teams  = db.query(Team).filter(Team.event_id == event_id, Team.status == "APPROVED").all()
    
    if not judges:
        raise HTTPException(status_code=400, detail="No judges added yet")
    if not teams:
        raise HTTPException(status_code=400, detail="No approved teams found")
    
    guides_created = 0
    guides_skipped = 0  # ✅ Track how many we skipped
    emails_drafted = 0
    
    for judge in judges:
        team_names = [t.name for t in teams]
        
        # Draft evaluation request email for this judge (LLM)
        portal_link = f"http://yourapp.com/judge/{judge.id}"
        email = draft_evaluation_request_email(
            judge_name=judge.name,
            teams_to_evaluate=team_names,
            eval_deadline="June 16, 2025 at 8:00 PM",
            criteria=request.criteria,
            portal_link=portal_link
        )
        
        comm = Communication(
            event_id=event_id,
            recipient_id=judge.id,
            recipient_type="JUDGE",
            type="EVAL_REQUEST",
            subject=email["subject"],
            body=email["body"],
            status="DRAFT"
        )
        db.add(comm)
        emails_drafted += 1
        
        # Generate evaluation guide for each team this judge will evaluate (LLM)
        for team in teams:
            members = db.query(Participant).filter(Participant.team_id == team.id).all()
            skills = list(set([s.strip() for m in members for s in m.skills.split(",")]))
            
            # ✅ CHECK: Don't create duplicate guide for same judge + team
            existing_guide = db.query(EvaluationGuide).filter(
                EvaluationGuide.judge_id == judge.id,
                EvaluationGuide.team_id == team.id
            ).first()
            
            if existing_guide:
                print(f"⏭️  Guide already exists for Judge {judge.id} + Team {team.id}, skipping")
                guides_skipped += 1
                continue  # Skip if already exists
            
            guide_content = generate_evaluation_guide(
                judge_name=judge.name,
                team_name=team.name,
                project_description="IoT Smart Energy Monitoring",
                team_skills=skills,
                criteria=request.criteria,
                minutes_per_team=request.minutes_per_team
            )
            
            guide = EvaluationGuide(
                event_id=event_id,
                judge_id=judge.id,
                team_id=team.id,
                content=guide_content
            )
            db.add(guide)
            db.commit()  # ✅ Commit after each guide
            guides_created += 1
            print(f"✅ Guide created for Judge {judge.name} + Team {team.name}")
    
    # Update event stage
    event = db.query(Event).filter(Event.id == event_id).first()
    event.current_stage = "EVALUATION"
    db.commit()
    
    return {
        "message": "Evaluation phase started",
        "guides_created": guides_created,
        "guides_skipped": guides_skipped,  # ✅ Show how many were duplicates
        "emails_drafted": emails_drafted
    }



@app.post("/api/scores/submit")
def submit_score(request: SubmitScoreRequest, db: Session = Depends(get_db)):
    """
    Judge submits a score for a team.
    After submission, checks if all judges have scored this team.
    If yes, runs anomaly detection automatically.
    
    FRONTEND: Called from judge portal when they submit their evaluation form
    LLM USED: YES — if anomaly detected, explain_anomaly() is called
    DB: Writes to scores table, possibly writes to anomalies table
    
    APPROVAL GATE: If anomaly detected, results are held until committee resolves it
    """
    # Save the score
    score = Score(
        event_id=db.query(Judge).filter(Judge.id == request.judge_id).first().event_id,
        team_id=request.team_id,
        judge_id=request.judge_id,
        score=request.score
    )
    db.add(score)
    db.commit()
    
    # Check if all judges have scored this team
    event_id = score.event_id
    total_judges = db.query(Judge).filter(Judge.event_id == event_id).count()
    team_scores  = db.query(Score).filter(Score.team_id == request.team_id).all()
    
    if len(team_scores) >= total_judges:
        # All judges done — run anomaly detection
        event = db.query(Event).filter(Event.id == event_id).first()
        team  = db.query(Team).filter(Team.id == request.team_id).first()
        judges_map = {j.id: j.name for j in db.query(Judge).filter(Judge.event_id == event_id).all()}
        
        scores_for_detection = [{
            "judge_name": judges_map.get(s.judge_id, "Unknown"),
            "judge_id": s.judge_id,
            "score": s.score
        } for s in team_scores]
        
        # detect_and_explain_anomaly() uses math first, then LLM if needed
        result = detect_and_explain_anomaly(
            team_name=team.name,
            scores=scores_for_detection,
            threshold=event.anomaly_threshold
        )
        
        if result["is_anomaly"]:
            anomaly = Anomaly(
                event_id=event_id,
                team_id=request.team_id,
                explanation=result["explanation"],
                results_held=True,
                status="PENDING_REVIEW"
            )
            db.add(anomaly)
            db.commit()
            
            return {
                "score_saved": True,
                "anomaly_detected": True,
                "message": "Score saved. Anomaly detected — results held for committee review.",
                "anomaly_explanation": result["explanation"]
            }
    
    return {"score_saved": True, "anomaly_detected": False, "message": "Score saved successfully"}


@app.get("/api/events/{event_id}/anomalies")
def list_anomalies(event_id: int, db: Session = Depends(get_db)):
    """
    Returns all anomalies for committee review.
    
    FRONTEND: Shown as alert cards on committee dashboard
    LLM USED: No
    DB: Reads from anomalies table
    """
    anomalies = db.query(Anomaly).filter(Anomaly.event_id == event_id).all()
    teams_map = {t.id: t.name for t in db.query(Team).filter(Team.event_id == event_id).all()}
    
    return [{
        "id": a.id,
        "team_name": teams_map.get(a.team_id, "Unknown"),
        "explanation": a.explanation,
        "status": a.status,
        "results_held": a.results_held
    } for a in anomalies]


@app.post("/api/anomalies/{anomaly_id}/resolve")
def resolve_anomaly(anomaly_id: int, request: ResolveAnomalyRequest, db: Session = Depends(get_db)):
    """
    Committee resolves a flagged anomaly.
    
    FRONTEND: Called when committee clicks "Resolve" on anomaly alert
    LLM USED: No
    DB: Updates anomaly.status, releases result hold
    """
    anomaly = db.query(Anomaly).filter(Anomaly.id == anomaly_id).first()
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    
    anomaly.status = "RESOLVED"
    anomaly.results_held = False
    db.commit()
    
    return {"message": f"Anomaly resolved with action: {request.action}"}


# ─────────────────────────────────────────────
# SECTION 7: DASHBOARD DATA
# ─────────────────────────────────────────────

@app.get("/api/events/{event_id}/dashboard")
def get_dashboard(event_id: int, db: Session = Depends(get_db)):
    """
    Returns everything the committee dashboard needs in one call.
    
    FRONTEND: Called when committee dashboard loads or refreshes
    LLM USED: No
    DB: Reads from all tables
    """
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    return {
        "event_name":       event.name,
        "current_stage":    event.current_stage,
        "pending_approvals": get_pending_approvals(db, event_id),
        "leaderboard":      get_leaderboard(db, event_id),
        "activity_log":     get_activity_log(db, event_id),
        "participant_count": db.query(Participant).filter(Participant.event_id == event_id).count(),
        "team_count":        db.query(Team).filter(Team.event_id == event_id).count(),
        "judge_count":       db.query(Judge).filter(Judge.event_id == event_id).count()
    }


# ─────────────────────────────────────────────
# SECTION 8: PARTICIPANT PORTAL
# ─────────────────────────────────────────────

@app.get("/api/participants/{participant_id}/status")
def get_participant_status(participant_id: int, db: Session = Depends(get_db)):
    """
    Returns a participant's current status for their read-only portal.
    
    FRONTEND: This is what the participant portal page shows
    LLM USED: No
    DB: Reads from participants, teams, scores, communications tables
    """
    participant = db.query(Participant).filter(Participant.id == participant_id).first()
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    event = db.query(Event).filter(Event.id == participant.event_id).first()
    
    result = {
        "name": participant.name,
        "event_name": event.name,
        "current_stage": event.current_stage,
        "team": None,
        "teammates": [],
        "key_dates": {
            "submission_deadline": "June 16, 2025 at 6:00 PM",
            "results_announcement": "June 17, 2025"
        }
    }
    
    if participant.team_id:
        team = db.query(Team).filter(Team.id == participant.team_id).first()
        teammates = db.query(Participant).filter(
            Participant.team_id == participant.team_id,
            Participant.id != participant_id
        ).all()
        
        result["team"] = {"name": team.name, "status": team.status}
        result["teammates"] = [{"name": t.name, "skills": t.skills} for t in teammates]
    
    return result


# ─────────────────────────────────────────────
# SECTION 9: RAG — COMMITTEE Q&A
# ─────────────────────────────────────────────

@app.post("/api/events/{event_id}/ask")
def ask_question(event_id: int, request: AskQuestionRequest, db: Session = Depends(get_db)):
    """
    Committee asks a natural language question about the event.
    RAG pipeline: retrieves relevant DB data → injects into prompt → LLM answers.
    
    FRONTEND: Called from the chat/Q&A box on the committee dashboard
    LLM USED: YES — answer_question() from rag.py
    DB: Reads from all tables (via rag.py)
    
    Example questions:
    - "Which teams haven't been evaluated yet?"
    - "Are there any pending anomalies?"
    - "What is the current leaderboard?"
    - "Which participants are in Team Orion?"
    """
    answer = answer_question(
        question=request.question,
        db=db,
        event_id=event_id
    )
    return {"question": request.question, "answer": answer}


# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # Run with: python main.py
    # Or:       uvicorn main:app --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



@app.post("/api/events/configure")
def configure_event_from_description(
    request: ConfigureEventRequest,
    db: Session = Depends(get_db)
):
    """
    Takes a free-text event description from the committee
    and runs the full LLM pipeline:
    1. Parse description into structured JSON
    2. Find missing information (gaps)
    3. Check for contradictions
    4. If clean, generate summary for committee to confirm

    FRONTEND: Called when committee submits their free-text
              event description in the configuration chat box
    LLM USED: YES — process_event_description() runs 3-4 LLM calls
    DB: Does NOT save yet — waits for committee confirmation

    Returns one of three statuses:
    - INCOMPLETE  → gaps found, returns questions for committee
    - CONTRADICTIONS → problems found, returns list of issues
    - READY → all good, returns summary for committee to confirm
    """
    import traceback
    try:
        # Verify event exists
        event = db.query(Event).filter(Event.id == request.event_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Run the full LLM pipeline
        # This makes 3-4 LLM calls — takes 1-3 minutes with Ollama
        result = process_event_description(request.description)

        # Store the parsed config temporarily even if not confirmed yet
        # So we can update it when committee answers gap questions
        if result["config"]:
            import json
            event.dynamic_config = json.dumps(result["config"])
            db.commit()

        return {
            "event_id": request.event_id,
            "status": result["status"],
            "config": result["config"],
            "questions": result["questions"],         # empty if READY
            "contradictions": result["contradictions"], # empty if READY
            "summary": result["summary"]              # only set if READY
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/events/configure/clarify")
def clarify_event_config(
    request: ClarifyConfigRequest,
    db: Session = Depends(get_db)
):
    """
    Called when committee answers the gap questions.
    Combines original description + answers and re-runs the pipeline.

    FRONTEND: Called when committee submits answers to gap questions
              shown after the first /configure call returned INCOMPLETE
    LLM USED: YES — re-runs full process_event_description()
    DB: Updates dynamic_config with improved config

    Example flow:
    1. Committee describes event → INCOMPLETE
       System asks: "What is the submission deadline?"
    2. Committee answers: "Deadline is 6pm on Day 2"
    3. This endpoint combines both and re-parses → hopefully READY
    """
    import traceback
    try:
        event = db.query(Event).filter(Event.id == request.event_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        # Combine original description with the new answers
        # so the LLM has full context when re-parsing
        combined_description = f"""Original description:
{request.original_description}

Additional information provided:
{request.answers}"""

        # Re-run the full pipeline with combined info
        result = process_event_description(combined_description)

        # Update the stored config
        if result["config"]:
            import json
            event.dynamic_config = json.dumps(result["config"])
            db.commit()

        return {
            "event_id": request.event_id,
            "status": result["status"],
            "config": result["config"],
            "questions": result["questions"],
            "contradictions": result["contradictions"],
            "summary": result["summary"]
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/events/configure/confirm")
def confirm_event_config(
    request: ConfirmConfigRequest,
    db: Session = Depends(get_db)
):
    """
    Committee confirms the summary is correct.
    This activates the dynamic config — event now runs
    entirely from what the committee described.

    FRONTEND: Called when committee clicks "Yes, this is correct"
              after seeing the summary from /configure
    LLM USED: NO
    DB: Marks dynamic_config as confirmed, updates event fields
        to match the dynamic config values
    """
    import json
    try:
        event = db.query(Event).filter(Event.id == request.event_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")

        if not request.confirmed:
            return {
                "message": "Configuration not confirmed. "
                           "Please re-describe your event or adjust the details."
            }

        if not event.dynamic_config:
            raise HTTPException(
                status_code=400,
                detail="No configuration found. "
                       "Please run /api/events/configure first."
            )

        # Load the saved config
        config = json.loads(event.dynamic_config)

        # Update the event's standard fields from dynamic config
        # So existing endpoints (team formation, evaluation, etc.)
        # automatically use the committee's custom values
        if config.get("team_size"):
            event.team_size = config["team_size"]

        if config.get("team_rules"):
            rules = config["team_rules"]
            if rules.get("skill_balance") is not None:
                event.skill_balance = rules["skill_balance"]
            if rules.get("no_same_institution") is not None:
                event.no_same_institution = rules["no_same_institution"]

        if config.get("evaluation", {}).get("anomaly_threshold"):
            event.anomaly_threshold = config["evaluation"]["anomaly_threshold"]

        if config.get("event_name"):
            event.name = config["event_name"]

        event.current_stage = "PARTICIPANT_INTAKE"
        db.commit()

        return {
            "message": "Event configured successfully from your description. "
                       "You can now upload participants and begin.",
            "event_id": event.id,
            "active_config": config
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/events/{event_id}/config")
def get_event_config(event_id: int, db: Session = Depends(get_db)):
    """
    Returns the current configuration for an event.
    Shows both standard fields and dynamic config if present.

    FRONTEND: Called to show committee what the current event config is
    LLM USED: NO
    DB: Reads from events table
    """
    import json
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    result = {
        "event_id": event.id,
        "event_name": event.name,
        "team_size": event.team_size,
        "skill_balance": event.skill_balance,
        "no_same_institution": event.no_same_institution,
        "anomaly_threshold": event.anomaly_threshold,
        "current_stage": event.current_stage,
        "is_dynamically_configured": event.dynamic_config is not None
    }

    if event.dynamic_config:
        result["dynamic_config"] = json.loads(event.dynamic_config)

    return result

@app.get("/api/judges/{judge_id}/guides")
def get_judge_guides(judge_id: int, db: Session = Depends(get_db)):
    """
    Returns all evaluation guides for a specific judge.
    FRONTEND: Called when judge portal loads to show guides.
    """
    guides = db.query(EvaluationGuide).filter(
        EvaluationGuide.judge_id == judge_id
    ).all()
    teams_map = {t.id: t.name for t in db.query(Team).all()}
    return [{
        "team_id": g.team_id,
        "team_name": teams_map.get(g.team_id, "Unknown"),
        "content": g.content
    } for g in guides]

# Add this near the other endpoints in main.py
@app.get("/api/debug/evaluations/{event_id}")
def debug_evaluations(event_id: int, db: Session = Depends(get_db)):
    """
    DEBUG ENDPOINT: Shows all judges and their assigned teams for an event.
    Visit: http://localhost:8000/api/debug/evaluations/1
    """
    from database import Judge, Team, EvaluationGuide
    
    judges = db.query(Judge).filter(Judge.event_id == event_id).all()
    teams = db.query(Team).filter(Team.event_id == event_id).all()
    approved_teams = [t for t in teams if t.status == "APPROVED"]
    guides = db.query(EvaluationGuide).filter(EvaluationGuide.event_id == event_id).all()
    
    result = {
        "event_id": event_id,
        "total_judges": len(judges),
        "total_teams": len(teams),
        "approved_teams": len(approved_teams),
        "total_guides": len(guides),
        "judges": [
            {
                "id": j.id,
                "name": j.name,
                "email": j.email,
                "guides_count": len([g for g in guides if g.judge_id == j.id]),
                "assigned_teams": [
                    {"team_id": g.team_id, "team_name": next((t.name for t in teams if t.id == g.team_id), "Unknown")}
                    for g in guides if g.judge_id == j.id
                ]
            }
            for j in judges
        ],
        "all_teams": [
            {"id": t.id, "name": t.name, "status": t.status}
            for t in teams
        ]
    }
    
    return result
