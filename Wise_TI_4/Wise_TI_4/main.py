from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from features.event_config_parser import process_event_description
import secrets
import string
import os
from datetime import datetime
from database import User, create_user, get_user_by_email, verify_user, generate_random_password, Team
# Add this with your other imports at the top of main.py
from werkzeug.security import generate_password_hash, check_password_hash
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
# Add this at the top with other imports (NOT inside the function)
from features.email_drafting import (
    draft_team_assignment_email,
    draft_evaluation_request_email,
    draft_deadline_reminder_email,
    draft_results_email,
    draft_welcome_credentials_email,      # ← Add this
    draft_progression_invitation_email,   # ← Add this
    draft_anomaly_reevaluation_email      # ← Add this
)
from dotenv import load_dotenv
load_dotenv()



# Committee code from environment variable (with fallback)
COMMITTEE_CODE = os.getenv("COMMITTEE_CODE", "TI2025HACK")

# ════════════════════════════════════════════════════
# EMAIL SENDING (with SendGrid + Mock Mode)
# ════════════════════════════════════════════════════

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "noreply@eventflow.com")

def actually_send_email(to_email: str, subject: str, body: str) -> bool:
    """
    Send email using SendGrid. Falls back to mock mode if no API key.
    Returns True if sent successfully (or mocked).
    """
    # Mock mode - just log it
    if not SENDGRID_API_KEY:
        print(f"\n{'='*60}")
        print(f"📧 [MOCK EMAIL]")
        print(f"   To: {to_email}")
        print(f"   Subject: {subject}")
        print(f"   Body: {body[:150]}...")
        print(f"{'='*60}\n")
        return True
    
    # Real SendGrid mode
    try:
        message = Mail(
            from_email=SENDER_EMAIL,
            to_emails=to_email,
            subject=subject,
            plain_text_content=body
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        
        if response.status_code == 202:
            print(f"📧 [EMAIL SENT] To: {to_email}")
            return True
        else:
            print(f"❌ [EMAIL FAILED] Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ [EMAIL ERROR] {str(e)}")
        return False



# Import database
from database import (
    get_db, create_tables,
    Event, Participant, Team, Judge, Score, Anomaly, Communication, EvaluationGuide, Approval,
    User,  # ← Add this
    create_user, get_user_by_email, verify_user,  # ← Add these
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
from llm_service import USE_CLOUD, OLLAMA_API_KEY

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


# ════════════════════════════════════════════════════
# AUTHENTICATION ENDPOINTS
# ════════════════════════════════════════════════════

# Environment variable for committee code
COMMITTEE_CODE = os.getenv("COMMITTEE_CODE", "TI2025HACK")

class LoginRequest(BaseModel):
    email: str
    password: str
    role: str  # 'COMMITTEE', 'PARTICIPANT', 'JUDGE'

class CommitteeCodeRequest(BaseModel):
    code: str

@app.post("/api/auth/check-setup")
def check_setup(db: Session = Depends(get_db)):
    """
    Check if any users exist (first-time setup check).
    """
    user_count = db.query(User).count()
    return {
        "needs_setup": user_count == 0,
        "user_count": user_count
    }

@app.post("/api/auth/committee-setup")
def committee_setup(request: CommitteeCodeRequest, db: Session = Depends(get_db)):
    """
    First-time committee setup. Verifies the committee code.
    Creates a placeholder committee user.
    """
    if request.code != COMMITTEE_CODE:
        raise HTTPException(status_code=401, detail="Invalid committee code")
    
    # Check if already set up
    existing = db.query(User).filter(User.role == "COMMITTEE").first()
    if existing:
        raise HTTPException(status_code=400, detail="Committee already exists. Please login.")
    
    # Create a special committee user (code-based, no password)
    # We use a dummy email since login is by code
    user = User(
        email="committee@eventflow.internal",
        password_hash=generate_password_hash(secrets.token_urlsafe(32)),
        role="COMMITTEE",
        reference_id=None
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {
        "message": "Committee setup complete",
        "user_id": user.id,
        "role": "COMMITTEE"
    }

@app.post("/api/auth/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Login endpoint.
    - Committee: verifies code
    - Participant/Judge: verifies email + password
    """
    print(f"🔐 Login attempt: role={request.role}, email={request.email}")
    
    if request.role == "COMMITTEE":
        # Committee uses code
        if request.password != COMMITTEE_CODE:
            print(f"❌ Invalid committee code")
            raise HTTPException(status_code=401, detail="Invalid committee code")
        
        user = db.query(User).filter(User.role == "COMMITTEE").first()
        if not user:
            print(f"❌ Committee not set up yet")
            raise HTTPException(status_code=404, detail="Committee not set up yet")
        
        user.last_login = datetime.utcnow()
        db.commit()
        
        print(f"✓ Committee login successful")
        return {
            "user_id": user.id,
            "email": "committee",
            "role": "COMMITTEE",
            "name": "Committee Member"
        }
    else:
        # Participant/Judge uses email + password
        user = verify_user(db, request.email, request.password)
        if not user:
            print(f"❌ Invalid credentials for {request.email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        if user.role != request.role:
            print(f"❌ Role mismatch: user is {user.role}, tried {request.role}")
            raise HTTPException(status_code=403, detail=f"This account is not a {request.role.lower()}")
        
        # Get name from reference
        name = "User"
        if user.role == "PARTICIPANT" and user.reference_id:
            p = db.query(Participant).filter(Participant.id == user.reference_id).first()
            if p: name = p.name
        elif user.role == "JUDGE" and user.reference_id:
            j = db.query(Judge).filter(Judge.id == user.reference_id).first()
            if j: name = j.name
        
        user.last_login = datetime.utcnow()
        db.commit()
        
        print(f"✓ {user.role} login successful: {name}")
        return {
            "user_id": user.id,
            "email": user.email,
            "role": user.role,
            "name": name,
            "reference_id": user.reference_id
        }



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
async def upload_participants(event_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload participants from CSV with validation.
    """
    
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    # Read file
    contents = await file.read()
    
    try:
        import io
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    
    # Validate required columns
    required_cols = ['name', 'email', 'skills', 'institution', 'experience']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400, 
            detail=f"Missing required columns: {', '.join(missing_cols)}"
        )
    
    count = 0
    errors = []
    
    for idx, row in df.iterrows():
        try:
            # Validate email format
            email = str(row['email']).strip()
            if '@' not in email or '.' not in email:
                errors.append(f"Row {idx+2}: Invalid email '{email}'")
                continue
            
            # Validate experience is numeric
            try:
                exp = int(row['experience'])
            except (ValueError, TypeError):
                errors.append(f"Row {idx+2}: Experience must be a number")
                continue
            
            # Validate name
            name = str(row['name']).strip()
            if not name:
                errors.append(f"Row {idx+2}: Name is required")
                continue
            
            # Create participant
            participant = Participant(
                event_id=event_id,
                name=name,
                email=email,
                skills=str(row['skills']).strip(),
                institution=str(row['institution']).strip(),
                experience=exp
            )
            db.add(participant)
            db.flush()
            
            # Auto-create user account
            try:
                random_password = generate_random_password(8)
                user, plain_password = create_user(
                    db, email=email, password=random_password,
                    role="PARTICIPANT", reference_id=participant.id
                )
                
                # Draft welcome email
                email_content = draft_welcome_credentials_email(
                    participant_name=name, email=email, password=random_password
                )
                
                comm = Communication(
                    event_id=event_id,
                    recipient_id=participant.id,
                    recipient_type="PARTICIPANT",
                    type="WELCOME_CREDENTIALS",
                    subject=email_content["subject"],
                    body=email_content["body"],
                    status="DRAFT"
                )
                db.add(comm)
            except Exception as e:
                # Email might already exist
                pass
            
            count += 1
        except Exception as e:
            errors.append(f"Row {idx+2}: {str(e)}")
            continue
    
    db.commit()
    
    # Update event stage
    event = db.query(Event).filter(Event.id == event_id).first()
    if event:
        event.current_stage = "TEAM_FORMATION"
        db.commit()
    
    return {
        "message": f"{count} participants loaded successfully",
        "count": count,
        "errors": errors if errors else None
    }




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

@app.post("/api/events/{event_id}/teams/reform")
def reform_teams(event_id: int, db: Session = Depends(get_db)):
    """
    Re-form teams using ONLY unassigned participants.
    """
    import traceback
    try:
        event = db.query(Event).filter(Event.id == event_id).first()
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        # Archive old rejected teams
        rejected_teams = db.query(Team).filter(
            Team.event_id == event_id,
            Team.status == "REJECTED"
        ).all()
        
        # Get the members of rejected teams BEFORE we archive them
        rejected_member_ids = []
        for t in rejected_teams:
            members = db.query(Participant).filter(Participant.team_id == t.id).all()
            rejected_member_ids.extend([m.id for m in members])
        
        # Mark rejected teams as archived
        for t in rejected_teams:
            t.status = "ARCHIVED"
        db.commit()
        
        # ✅ Get ONLY unassigned participants
        unassigned = db.query(Participant).filter(
            Participant.event_id == event_id,
            Participant.team_id == None
        ).all()
        
        if not unassigned:
            raise HTTPException(
                status_code=400, 
                detail="No unassigned participants. All participants are already in teams."
            )
        
        # Categorize for better UX
        rejected_members = [p for p in unassigned if p.id in rejected_member_ids]
        never_assigned = [p for p in unassigned if p.id not in rejected_member_ids]
        
        # ✅ Count approved team members (for the breakdown)
        approved_team_members_excluded = db.query(Participant).filter(
            Participant.event_id == event_id,
            Participant.team_id != None,
            Participant.id.notin_(rejected_member_ids)
        ).count()
        
        # Form new teams from unassigned
        rules = {
            "team_size": event.team_size,
            "skill_balance": event.skill_balance,
            "no_same_institution": event.no_same_institution
        }
        
        formed_teams = form_teams(unassigned, event.team_size, event.no_same_institution)
        
        saved_teams = []
        for idx, team_data in enumerate(formed_teams):
            ref_name = f"{team_data['name']} (Reform {idx + 1})"
            
            team = Team(
                event_id=event_id,
                name=ref_name,
                status="PENDING_APPROVAL"
            )
            db.add(team)
            db.commit()
            db.refresh(team)
            
            for member in team_data["members"]:
                participant = db.query(Participant).filter(Participant.id == member.id).first()
                if participant:
                    participant.team_id = team.id
            db.commit()
            
            rationale = generate_team_rationale(
                team_name=team.name,
                members=team_data["members"],
                rules=rules
            )
            team.rationale = rationale
            db.commit()
            
            saved_teams.append({
                "team_id": team.id,
                "team_name": team.name,
                "status": team.status,
                "members": [m.name for m in team_data["members"]]
            })
        
        return {
            "message": f"Reformed {len(saved_teams)} teams from {len(unassigned)} unassigned participants",
            "teams": saved_teams,
            "breakdown": {
                "total_unassigned": len(unassigned),
                "from_rejected_teams": len(rejected_members),
                "never_assigned": len(never_assigned),
                "approved_team_members_excluded": approved_team_members_excluded  # ✅ Fixed
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



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
    Approve and ACTUALLY SEND an email to the recipient.
    """
    from datetime import datetime
    comm = db.query(Communication).filter(Communication.id == comm_id).first()
    if not comm:
        raise HTTPException(status_code=404, detail="Communication not found")
    
    # Don't re-send already sent emails
    if comm.status == "SENT":
        return {"message": "Email already sent", "already_sent": True}
    
    # Get the actual email address of the recipient
    recipient = None
    if comm.recipient_type == "PARTICIPANT":
        recipient = db.query(Participant).filter(Participant.id == comm.recipient_id).first()
    elif comm.recipient_type == "JUDGE":
        recipient = db.query(Judge).filter(Judge.id == comm.recipient_id).first()
    
    if not recipient or not recipient.email:
        raise HTTPException(status_code=400, detail="Recipient has no email address")
    
    # ACTUALLY SEND THE EMAIL (or mock if no API key)
    email_sent = actually_send_email(recipient.email, comm.subject, comm.body)
    
    # Update status
    comm.status = "SENT" if email_sent else "FAILED"
    comm.sent_at = datetime.utcnow()
    db.commit()
    
    if email_sent:
        return {
            "message": f"Email sent to {recipient.email}",
            "sent": True,
            "mock_mode": not bool(SENDGRID_API_KEY)
        }
    else:
        raise HTTPException(status_code=500, detail="Email delivery failed")



# ─────────────────────────────────────────────
# SECTION 5: JUDGES
# ─────────────────────────────────────────────

@app.post("/api/events/{event_id}/judges")
def add_judge(event_id: int, request: AddJudgeRequest, db: Session = Depends(get_db)):
    """
    Adds a judge AND auto-creates a user account.
    """
    
    
    # Create judge
    judge = Judge(event_id=event_id, name=request.name, email=request.email)
    db.add(judge)
    db.flush()
    
    # AUTO-CREATE USER ACCOUNT
    try:
        random_password = generate_random_password(8)
        user, plain_password = create_user(
            db,
            email=request.email,
            password=random_password,
            role="JUDGE",
            reference_id=judge.id
        )
        
        # Draft welcome email with credentials
        email_content = draft_welcome_credentials_email(
            participant_name=request.name,  # Reusing the function
            email=request.email,
            password=random_password
        )
        
        # Save email
        comm = Communication(
            event_id=event_id,
            recipient_id=judge.id,
            recipient_type="JUDGE",
            type="JUDGE_CREDENTIALS",
            subject=email_content["subject"],
            body=email_content["body"],
            status="DRAFT"
        )
        db.add(comm)
        
    except Exception as e:
        print(f"User creation skipped for {request.email}: {e}")
    
    db.commit()
    db.refresh(judge)
    
    return {
        "judge_id": judge.id, 
        "message": f"Judge {judge.name} added. Credentials email drafted."
    }




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
    Submit or UPDATE a judge's score for a team.
    If judge already scored this team, update the score instead of creating duplicate.
    """
    # Check if score already exists
    existing_score = db.query(Score).filter(
        Score.judge_id == request.judge_id,
        Score.team_id == request.team_id
    ).first()
    
    if existing_score:
        # Update existing score
        existing_score.score = request.score
        existing_score.submitted_at = datetime.utcnow()
        db.commit()
        return {
            "score_saved": True,
            "updated": True,
            "message": "Score updated successfully"
        }
    
    # Create new score
    judge = db.query(Judge).filter(Judge.id == request.judge_id).first()
    if not judge:
        raise HTTPException(status_code=404, detail="Judge not found")
    
    score = Score(
        event_id=judge.event_id,
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
    Resolve a flagged anomaly.
    Auto-transitions event to RESULTS stage when all anomalies are resolved.
    """
    
    
    anomaly = db.query(Anomaly).filter(Anomaly.id == anomaly_id).first()
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    
    # Look up team name
    team = db.query(Team).filter(Team.id == anomaly.team_id).first()
    team_name = team.name if team else f"Team #{anomaly.team_id}"
    
    # If requesting re-evaluation, send email to the judge whose score was flagged
    if request.action == "request_reevaluation":
        scores = db.query(Score).filter(Score.team_id == anomaly.team_id).all()
        
        if scores:
            avg = sum(s.score for s in scores) / len(scores)
            for s in scores:
                deviation = abs(s.score - avg)
                if deviation > 20:
                    judge = db.query(Judge).filter(Judge.id == s.judge_id).first()
                    if judge:
                        email = draft_anomaly_reevaluation_email(
                            judge_name=judge.name,
                            team_name=team_name,
                            anomaly_explanation=anomaly.explanation
                        )
                        
                        comm = Communication(
                            event_id=anomaly.event_id,
                            recipient_id=judge.id,
                            recipient_type="JUDGE",
                            type="ANOMALY_RESOLUTION",
                            subject=email["subject"],
                            body=email["body"],
                            status="DRAFT"
                        )
                        db.add(comm)
                        db.commit()
    
    # Mark anomaly as resolved
    anomaly.status = "RESOLVED"
    anomaly.results_held = False
    db.commit()
    
    # ✅ NEW: Check if all anomalies are now resolved → auto-transition to RESULTS
    event = db.query(Event).filter(Event.id == anomaly.event_id).first()
    if event:
        remaining_anomalies = db.query(Anomaly).filter(
            Anomaly.event_id == anomaly.event_id,
            Anomaly.status == "PENDING_REVIEW"
        ).count()
        
        # Also check if all approved teams have been scored
        approved_teams = db.query(Team).filter(
            Team.event_id == anomaly.event_id,
            Team.status == "APPROVED"
        ).all()
        
        teams_fully_scored = 0
        for t in approved_teams:
            scores = db.query(Score).filter(Score.team_id == t.id).all()
            judges = db.query(Judge).filter(Judge.event_id == anomaly.event_id).all()
            if len(scores) >= len(judges) and len(judges) > 0:
                teams_fully_scored += 1
        
        # If all anomalies resolved AND all teams scored → move to RESULTS
        if remaining_anomalies == 0 and len(approved_teams) > 0 and teams_fully_scored == len(approved_teams):
            if event.current_stage != "RESULTS":
                event.current_stage = "RESULTS"
                db.commit()
                print(f"✓ Event {event.id} auto-transitioned to RESULTS stage")
                
                # Auto-draft results emails for all participants
                for team in approved_teams:
                    team_scores = db.query(Score).filter(Score.team_id == team.id).all()
                    if team_scores:
                        avg = sum(s.score for s in team_scores) / len(team_scores)
                        
                        # Get rank
                        all_team_avgs = []
                        for t in approved_teams:
                            t_scores = db.query(Score).filter(Score.team_id == t.id).all()
                            if t_scores:
                                t_avg = sum(s.score for s in t_scores) / len(t_scores)
                                all_team_avgs.append({"team_id": t.id, "avg": t_avg, "name": t.name})
                        
                        all_team_avgs.sort(key=lambda x: x["avg"], reverse=True)
                        rank = next((i+1 for i, t in enumerate(all_team_avgs) if t["team_id"] == team.id), 0)
                        
                        # Draft email for each team member
                        members = db.query(Participant).filter(Participant.team_id == team.id).all()
                        for member in members:
                            try:
                                
                                email = draft_results_email(
                                    participant_name=member.name,
                                    team_name=team.name,
                                    score=avg,
                                    rank=rank,
                                    qualified=(rank <= 3),
                                    next_round_date="June 25, 2025",
                                    confirmation_link=f"http://localhost:5173/confirm/{member.id}"
                                )
                                
                                comm = Communication(
                                    event_id=event.id,
                                    recipient_id=member.id,
                                    recipient_type="PARTICIPANT",
                                    type="RESULTS",
                                    subject=email["subject"],
                                    body=email["body"],
                                    status="DRAFT"
                                )
                                db.add(comm)
                            except Exception as e:
                                print(f"Error drafting results email for {member.name}: {e}")
                
                db.commit()
    
    return {
        "message": f"Anomaly resolved with action: {request.action}",
        "auto_transitioned": event.current_stage == "RESULTS" if event else False
    }

@app.post("/api/events/{event_id}/publish-results")
def publish_results(event_id: int, db: Session = Depends(get_db)):
    """
    Manually publish results (bypasses auto-transition).
    Drafts results emails for all participants.
    """
    
    
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    approved_teams = db.query(Team).filter(
        Team.event_id == event_id,
        Team.status == "APPROVED"
    ).all()
    
    if not approved_teams:
        raise HTTPException(status_code=400, detail="No approved teams to publish results for")
    
    # Calculate rankings
    team_rankings = []
    for team in approved_teams:
        scores = db.query(Score).filter(Score.team_id == team.id).all()
        if scores:
            avg = sum(s.score for s in scores) / len(scores)
            team_rankings.append({"team": team, "avg": avg})
    
    team_rankings.sort(key=lambda x: x["avg"], reverse=True)
    
    # Draft results emails
    emails_created = 0
    for idx, item in enumerate(team_rankings):
        rank = idx + 1
        team = item["team"]
        avg_score = item["avg"]
        
        members = db.query(Participant).filter(Participant.team_id == team.id).all()
        for member in members:
            try:
                email = draft_results_email(
                    participant_name=member.name,
                    team_name=team.name,
                    score=avg_score,
                    rank=rank,
                    qualified=(rank <= 3),
                    next_round_date="June 25, 2025",
                    confirmation_link=f"http://localhost:5173/confirm/{member.id}"
                )
                
                comm = Communication(
                    event_id=event_id,
                    recipient_id=member.id,
                    recipient_type="PARTICIPANT",
                    type="RESULTS",
                    subject=email["subject"],
                    body=email["body"],
                    status="DRAFT"
                )
                db.add(comm)
                emails_created += 1
            except Exception as e:
                print(f"Error: {e}")
    
    db.commit()
    
    # Update event stage
    event.current_stage = "RESULTS"
    db.commit()
    
    return {
        "message": f"Results published! {emails_created} emails drafted.",
        "emails_created": emails_created,
        "leaderboard": [
            {"rank": i+1, "team_name": item["team"].name, "score": round(item["avg"], 2)}
            for i, item in enumerate(team_rankings)
        ]
    }


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
    Returns a participant's complete read-only status.
    """
    from database import Score, Anomaly
    
    participant = db.query(Participant).filter(Participant.id == participant_id).first()
    if not participant:
        raise HTTPException(status_code=404, detail="Participant not found")
    
    event = db.query(Event).filter(Event.id == participant.event_id).first()
    
    result = {
        "name": participant.name,
        "email": participant.email,
        "event_name": event.name if event else "Unknown",
        "current_stage": event.current_stage if event else "UNKNOWN",
        "team": None,
        "teammates": [],
        "key_dates": {
            "submission_deadline": "June 16, 2025 at 6:00 PM",
            "results_announcement": "June 17, 2025",
            "evaluation_deadline": "June 16, 2025 at 8:00 PM"
        },
        "score": None,
        "feedback": None,
        "progression": None
    }
    
    # Team info
    if participant.team_id:
        team = db.query(Team).filter(Team.id == participant.team_id).first()
        teammates = db.query(Participant).filter(
            Participant.team_id == participant.team_id,
            Participant.id != participant_id
        ).all()
        
        if team:
            result["team"] = {
                "id": team.id,
                "name": team.name,
                "status": team.status
            }
            result["teammates"] = [
                {"name": t.name, "skills": t.skills, "email": t.email}
                for t in teammates
            ]
            
            # Calculate team's average score
            scores = db.query(Score).filter(Score.team_id == team.id).all()
            if scores and team.status == "APPROVED":
                avg = sum([s.score for s in scores]) / len(scores)
                result["score"] = round(avg, 1)
                result["feedback"] = "Your team has been evaluated. See detailed feedback in your email."
    
    # ✅ NEW: Check if participant qualifies (top N from leaderboard)
    if event and result["score"]:
        # Get all approved teams with scores
        teams_with_scores = []
        approved_teams = db.query(Team).filter(
            Team.event_id == event.id,
            Team.status == "APPROVED"
        ).all()
        
        for t in approved_teams:
            t_scores = db.query(Score).filter(Score.team_id == t.id).all()
            if t_scores:
                avg = sum([s.score for s in t_scores]) / len(t_scores)
                teams_with_scores.append({"team_id": t.id, "avg": avg, "name": t.name})
        
        # Sort by average score descending
        teams_with_scores.sort(key=lambda x: x["avg"], reverse=True)
        
        # Top 3 qualify
        top_n = 3
        for rank, t in enumerate(teams_with_scores[:top_n], 1):
            if t["team_id"] == participant.team_id:
                result["progression"] = {
                    "qualified": True,
                    "rank": rank,
                    "score": result["score"],
                    "next_round": "Round 2",
                    "next_round_date": "June 20, 2025",
                    "confirmation_deadline": "June 18, 2025 at 12:00 PM",
                    "confirmation_status": None  # Frontend will manage this
                }
                break
    
    return result


# ─────────────────────────────────────────────
# SECTION 9: RAG — COMMITTEE Q&A
# ─────────────────────────────────────────────

class AskQuestionRequest(BaseModel):
    question: str
    user_id: Optional[int] = None
    user_role: Optional[str] = None
    user_email: Optional[str] = None
    reference_id: Optional[int] = None

@app.post("/api/events/{event_id}/ask")
def ask_question(event_id: int, request: AskQuestionRequest, db: Session = Depends(get_db)):
    # Build user context for role-aware filtering
    user_context = {
        'user_id': request.user_id,
        'user_role': request.user_role,
        'user_email': request.user_email,
        'reference_id': request.reference_id,
    }
    
    answer = answer_question(
        question=request.question,
        db=db,
        event_id=event_id,
        user_context=user_context
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

# ─────────────────────────────────────────────
# DEBUG: LLM STATUS CHECK
# ─────────────────────────────────────────────

@app.get("/api/llm-status")
def llm_status():
    """Check which LLM backend is currently active (cloud or local)."""
    return {
        "use_cloud": USE_CLOUD,
        "api_key_loaded": bool(OLLAMA_API_KEY),
        "model": "gpt-oss:20b (cloud)" if USE_CLOUD else "qwen3:0.6b (local)",
        "key_preview": OLLAMA_API_KEY[:8] + "..." if OLLAMA_API_KEY else None
    }
