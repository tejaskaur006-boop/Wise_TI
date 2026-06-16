# features/team_formation.py
import random
import traceback
from llm_service import call_llm

# ─────────────────────────────────────────────
# PART 1: PURE PYTHON ALGORITHM
# No LLM here — just logic to group participants
# ─────────────────────────────────────────────

def form_teams(participants: list, team_size: int, no_same_institution: bool) -> list:
    """
    Groups participants into balanced teams.
    """
    sorted_participants = sorted(participants, key=lambda p: p.skills)
    
    teams = []
    used = set()
    team_number = 1
    
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
        
        for candidate in sorted_participants:
            if len(team) >= team_size:
                break
            if candidate.id in used:
                continue
            
            if no_same_institution:
                team_institutions = [m.institution for m in team]
                if candidate.institution in team_institutions:
                    continue
            
            team_skills = []
            for m in team:
                team_skills.extend(m.skills.split(","))
            candidate_skills = candidate.skills.split(",")
            
            if any(s.strip() not in team_skills for s in candidate_skills):
                team.append(candidate)
                used.add(candidate.id)
        
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
# PART 2: LLM RATIONALE GENERATION (WITH FALLBACK)
# ─────────────────────────────────────────────

def generate_team_rationale(team_name: str, members: list, rules: dict) -> str:
    """
    Takes a formed team and asks the LLM to explain why this grouping makes sense.
    NOW WITH FALLBACK if LLM fails!
    """
    
    # Build a readable description of each member
    members_text = ""
    skills_set = set()
    institutions_set = set()
    for m in members:
        members_text += f"- {m.name} | Skills: {m.skills} | Institution: {m.institution} | Experience: {m.experience} years\n"
        for s in m.skills.split(","):
            skills_set.add(s.strip())
        institutions_set.add(m.institution)
    
    rules_parts = []
    if rules.get("skill_balance"):
        rules_parts.append("balanced skill sets required")
    if rules.get("no_same_institution"):
        rules_parts.append("no two members from same institution")
    rules_text = ", ".join(rules_parts) if rules_parts else "standard grouping"
    
    # Try LLM first
    print(f"\n🤖 Generating rationale for {team_name}...")
    print(f"   Members: {len(members)}")
    print(f"   Skills: {skills_set}")
    print(f"   Institutions: {institutions_set}")
    
    try:
        system = "You are a hackathon coordinator. Write a clear team rationale in 3-4 sentences. Return ONLY plain text, no JSON, no markdown."
        user = f"""Team: {team_name}
Members:
{members_text}
Rules: {rules_text}

Write a COMPLETE 3-4 sentence rationale (minimum 100 words) explaining why this specific team composition makes sense. 

CRITICAL INSTRUCTIONS:
- Do NOT stop mid-sentence under any circumstances
- Complete EVERY sentence you start
- Must include: (1) what skills each member brings, (2) how they complement each other, (3) why the institutional diversity is valuable
- End with a proper concluding sentence with a period
- Aim for 3-4 complete, well-formed sentences"""
        
        rationale = call_llm(system, user, max_tokens=400)
        
        print(f"   ✅ LLM returned ({len(rationale)} chars): {rationale[:100]}...")
        
        if rationale and len(rationale.strip()) > 20:
            return rationale.strip()
        else:
            print(f"   ⚠️ LLM response too short, using fallback")
    
    except Exception as e:
        print(f"   ❌ LLM call failed: {e}")
        traceback.print_exc()
    
    # FALLBACK: Generate manual rationale
    print(f"   📝 Using fallback rationale for {team_name}")
    
    skills_list = ", ".join(sorted(list(skills_set))[:5])  # Top 5 skills
    if len(skills_set) > 5:
        skills_list += f" and {len(skills_set) - 5} more"
    
    num_institutions = len(institutions_set)
    num_members = len(members)
    avg_experience = sum(m.experience for m in members) / len(members) if members else 0
    
    fallback = (
        f"Team {team_name} was formed to bring together {num_members} members with complementary "
        f"technical skills including {skills_list}. "
        f"The team represents {num_institutions} different institution{'s' if num_institutions > 1 else ''}, "
        f"ensuring diverse perspectives and backgrounds in line with the event rules. "
        f"With an average experience of {avg_experience:.1f} years, the members can collaborate effectively "
        f"on both technical implementation and creative problem-solving. "
        f"This balanced composition allows the team to handle full-stack development with a mix of "
        f"frontend, backend, and specialized capabilities."
    )
    
    return fallback
