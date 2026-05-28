import random
from llm_service import call_llm

# ─────────────────────────────────────────────
# PART 1: PURE PYTHON ALGORITHM
# No LLM here — just logic to group participants
# ─────────────────────────────────────────────

def form_teams(participants: list, team_size: int, no_same_institution: bool) -> list:
    """
    Groups participants into balanced teams.
    
    participants = list of Participant objects from the database
    team_size    = number of people per team (set by committee)
    no_same_institution = whether two people from same college can be on same team
    
    Returns: list of teams, where each team is a list of participants
    
    CALLED BY: main.py in the POST /api/teams/form endpoint
    CALLS: generate_team_rationale() after forming each team
    """
    
    # Sort participants by skills so we can spread them evenly
    # This ensures each team gets different skill types
    sorted_participants = sorted(participants, key=lambda p: p.skills)
    
    teams = []
    used = set()  # track who has already been assigned
    team_number = 1
    
    # Team names — add more if you have many teams
    team_names = [
        "Team Orion", "Team Nova", "Team Zenith", "Team Phoenix",
        "Team Atlas", "Team Apex", "Team Helix", "Team Nexus",
        "Team Sigma", "Team Titan", "Team Vega", "Team Zephyr",
        "Team Echo", "Team Flux", "Team Prism", "Team Quasar",
        "Team Rigel", "Team Solar", "Team Terra", "Team Umbra"
    ]
    
    for i, anchor in enumerate(sorted_participants):
        if anchor.id in used:
            continue
        
        team = [anchor]
        used.add(anchor.id)
        
        # Try to fill the rest of the team
        for candidate in sorted_participants:
            if len(team) >= team_size:
                break
            if candidate.id in used:
                continue
            
            # Check institution constraint
            if no_same_institution:
                team_institutions = [m.institution for m in team]
                if candidate.institution in team_institutions:
                    continue
            
            # Check skill diversity — prefer different skills
            team_skills = []
            for m in team:
                team_skills.extend(m.skills.split(","))
            candidate_skills = candidate.skills.split(",")
            
            # Add if at least one skill is different
            if any(s.strip() not in team_skills for s in candidate_skills):
                team.append(candidate)
                used.add(candidate.id)
        
        # If we couldn't fill the team due to constraints,
        # fill with anyone remaining
        for candidate in sorted_participants:
            if len(team) >= team_size:
                break
            if candidate.id in used:
                continue
            team.append(candidate)
            used.add(candidate.id)
        
        if team:
            name = team_names[team_number - 1] if team_number <= len(team_names) else f"Team {team_number}"
            teams.append({
                "name": name,
                "members": team
            })
            team_number += 1
    
    return teams


# ─────────────────────────────────────────────
# PART 2: LLM RATIONALE GENERATION
# Called after algorithm forms teams
# ─────────────────────────────────────────────

def generate_team_rationale(team_name: str, members: list, rules: dict) -> str:
    """
    Takes a formed team and asks the LLM to explain why this grouping makes sense.
    
    team_name = "Team Orion"
    members   = list of Participant objects
    rules     = {"team_size": 3, "skill_balance": True, "no_same_institution": True}
    
    Returns: plain text rationale string (3-4 sentences)
    
    CALLED BY: main.py after form_teams() returns results
    CALLS: call_llm() from llm_service.py
    """
    
    # Build a readable description of each member
    members_text = ""
    for m in members:
        members_text += f"- {m.name} | Skills: {m.skills} | Institution: {m.institution} | Experience: {m.experience} years\n"
    
    # Build rules description
    rules_parts = []
    if rules.get("skill_balance"):
        rules_parts.append("balanced skill sets required")
    if rules.get("no_same_institution"):
        rules_parts.append("no two members from same institution")
    rules_text = ", ".join(rules_parts) if rules_parts else "standard grouping"
    
    system = "You are a hackathon coordinator assistant. Write clear, professional team rationales in exactly 3-4 sentences."
    
    user = f"""Team name: {team_name}
Members:
{members_text}
Rules applied: {rules_text}

Write a 3-4 sentence rationale explaining why this specific team composition makes sense. 
Focus on skill complementarity and how the members can work together effectively."""
    
    # LLM writes the rationale
    # This text gets saved to the database and shown on the committee dashboard
    return call_llm(system, user, max_tokens=250)
