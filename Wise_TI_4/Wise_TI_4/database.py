from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

# SQLite — creates a file called eventflow.db in your project folder
# No installation needed, no server needed, just a file
ENGINE = create_engine("sqlite:///eventflow.db", echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=ENGINE)

# ─────────────────────────────────────────────
# DATABASE TABLES
# Each class = one table in your database
# ─────────────────────────────────────────────

class Event(Base):
    """
    Stores the event configuration.
    One row = one event being managed.
    CONNECTED TO: main.py (create/read event), rag.py (reads for context)
    """
    __tablename__ = "events"
    
    id               = Column(Integer, primary_key=True, index=True)
    name             = Column(String)              # e.g. "TI Hackathon 2025"
    current_stage    = Column(String, default="PARTICIPANT_INTAKE")
    team_size        = Column(Integer, default=3)
    skill_balance    = Column(Boolean, default=True)
    no_same_institution = Column(Boolean, default=True)
    anomaly_threshold = Column(Float, default=20.0)  # score deviation threshold
    dynamic_config      = Column(Text, nullable=True)   
    created_at       = Column(DateTime, default=datetime.utcnow)


class Participant(Base):
    """
    Stores every participant loaded from the CSV.
    One row = one participant.
    CONNECTED TO: main.py (load CSV), team_formation.py (reads to form teams)
    """
    __tablename__ = "participants"
    
    id          = Column(Integer, primary_key=True, index=True)
    event_id    = Column(Integer, ForeignKey("events.id"))
    name        = Column(String)
    email       = Column(String)
    skills      = Column(String)       # stored as comma-separated: "ML,Python,React"
    institution = Column(String)
    experience  = Column(Integer)      # years of experience
    team_id     = Column(Integer, ForeignKey("teams.id"), nullable=True)


class Team(Base):
    """
    Stores formed teams.
    One row = one team.
    status flow: PENDING_APPROVAL → APPROVED → ANNOUNCED
    CONNECTED TO: main.py, team_formation.py, email_drafting.py, rag.py
    """
    __tablename__ = "teams"
    
    id          = Column(Integer, primary_key=True, index=True)
    event_id    = Column(Integer, ForeignKey("events.id"))
    name        = Column(String)        # e.g. "Team Orion"
    rationale   = Column(Text)          # LLM-generated explanation
    status      = Column(String, default="PENDING_APPROVAL")
    created_at  = Column(DateTime, default=datetime.utcnow)


class Judge(Base):
    """
    Stores judges for the event.
    CONNECTED TO: main.py (add judges), evaluation.py (reads judge info)
    """
    __tablename__ = "judges"
    
    id       = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, ForeignKey("events.id"))
    name     = Column(String)
    email    = Column(String)


class Score(Base):
    """
    Stores scores submitted by judges for each team.
    One row = one judge's score for one team.
    CONNECTED TO: main.py (submit score), evaluation.py (reads for anomaly check)
    """
    __tablename__ = "scores"
    
    id         = Column(Integer, primary_key=True, index=True)
    event_id   = Column(Integer, ForeignKey("events.id"))
    team_id    = Column(Integer, ForeignKey("teams.id"))
    judge_id   = Column(Integer, ForeignKey("judges.id"))
    score      = Column(Float)
    submitted_at = Column(DateTime, default=datetime.utcnow)


class Anomaly(Base):
    """
    Stores flagged score anomalies.
    status flow: PENDING_REVIEW → RESOLVED
    CONNECTED TO: evaluation.py (creates), main.py (resolves), rag.py (reads)
    """
    __tablename__ = "anomalies"
    
    id            = Column(Integer, primary_key=True, index=True)
    event_id      = Column(Integer, ForeignKey("events.id"))
    team_id       = Column(Integer, ForeignKey("teams.id"))
    explanation   = Column(Text)     # LLM-generated explanation
    results_held  = Column(Boolean, default=True)
    status        = Column(String, default="PENDING_REVIEW")
    created_at    = Column(DateTime, default=datetime.utcnow)


class Communication(Base):
    """
    Stores every email draft and its send status.
    status flow: DRAFT → APPROVED → SENT
    CONNECTED TO: email_drafting.py (creates drafts), main.py (approves/sends)
    """
    __tablename__ = "communications"
    
    id           = Column(Integer, primary_key=True, index=True)
    event_id     = Column(Integer, ForeignKey("events.id"))
    recipient_id = Column(Integer)    # participant or judge id
    recipient_type = Column(String)   # "PARTICIPANT" or "JUDGE"
    type         = Column(String)     # "TEAM_ASSIGNMENT", "EVAL_REQUEST", "RESULTS"
    subject      = Column(String)
    body         = Column(Text)
    status       = Column(String, default="DRAFT")
    created_at   = Column(DateTime, default=datetime.utcnow)
    sent_at      = Column(DateTime, nullable=True)

class Approval(Base):
    """
    Stores committee approval history for teams/results/communications.
    """

    __tablename__ = "approvals"

    id = Column(Integer, primary_key=True, index=True)

    event_id = Column(Integer, ForeignKey("events.id"))

    team_id = Column(Integer, ForeignKey("teams.id"))

    approval_type = Column(String)

    status = Column(String)

    approved_by = Column(String)

    remarks = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)


class EvaluationGuide(Base):
    """
    Stores LLM-generated evaluation guides for each judge-team pair.
    CONNECTED TO: evaluation.py (creates), main.py (reads for judge portal)
    """
    __tablename__ = "evaluation_guides"
    
    id        = Column(Integer, primary_key=True, index=True)
    event_id  = Column(Integer, ForeignKey("events.id"))
    judge_id  = Column(Integer, ForeignKey("judges.id"))
    team_id   = Column(Integer, ForeignKey("teams.id"))
    content   = Column(Text)    # LLM-generated markdown guide
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────
# DATABASE HELPER FUNCTIONS
# These are called by main.py to read/write data
# ─────────────────────────────────────────────

def get_db():
    """Returns a database session. Used in every endpoint in main.py"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """
    Creates all tables if they don't exist.
    Called once when the app starts.
    """
    Base.metadata.create_all(bind=ENGINE)


def get_event(db, event_id: int):
    return db.query(Event).filter(Event.id == event_id).first()


def get_all_participants(db, event_id: int):
    return db.query(Participant).filter(Participant.id == event_id).all()


def get_participants_without_team(db, event_id: int):
    """Used by team formation algorithm"""
    return db.query(Participant).filter(
        Participant.event_id == event_id,
        Participant.team_id == None
    ).all()


def get_all_teams(db, event_id: int):
    return db.query(Team).filter(Team.event_id == event_id).all()


def get_team_members(db, team_id: int):
    return db.query(Participant).filter(Participant.team_id == team_id).all()


def get_all_scores(db, event_id: int):
    return db.query(Score).filter(Score.event_id == event_id).all()


def get_scores_for_team(db, team_id: int):
    return db.query(Score).filter(Score.team_id == team_id).all()


def get_pending_approvals(db, event_id: int):
    """Returns all items waiting for committee approval"""
    pending_teams = db.query(Team).filter(
        Team.event_id == event_id,
        Team.status == "PENDING_APPROVAL"
    ).count()
    
    pending_comms = db.query(Communication).filter(
        Communication.event_id == event_id,
        Communication.status == "DRAFT"
    ).count()
    
    pending_anomalies = db.query(Anomaly).filter(
        Anomaly.event_id == event_id,
        Anomaly.status == "PENDING_REVIEW"
    ).count()
    
    return {
        "pending_teams": pending_teams,
        "pending_communications": pending_comms,
        "pending_anomalies": pending_anomalies,
        "total": pending_teams + pending_comms + pending_anomalies
    }


def get_leaderboard(db, event_id: int):
    """Returns teams sorted by average score"""
    teams = db.query(Team).filter(
        Team.event_id == event_id,
        Team.status == "APPROVED"
    ).all()
    
    leaderboard = []
    for team in teams:
        scores = get_scores_for_team(db, team.id)
        if scores:
            avg = sum([s.score for s in scores]) / len(scores)
            leaderboard.append({
                "team_id": team.id,
                "team_name": team.name,
                "average_score": round(avg, 2),
                "scores": [{"score": s.score} for s in scores],
                "num_scores": len(scores)
            })
    
    return sorted(leaderboard, key=lambda x: x["average_score"], reverse=True)


def get_activity_log(db, event_id: int):
    """Returns recent communications as activity log"""
    comms = db.query(Communication).filter(
        Communication.event_id == event_id
    ).order_by(Communication.created_at.desc()).limit(50).all()
    
    return [{
        "type": c.type,
        "recipient_type": c.recipient_type,
        "status": c.status,
        "created_at": str(c.created_at)
    } for c in comms]
