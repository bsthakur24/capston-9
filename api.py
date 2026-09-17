"""FastAPI service exposing the recommendation engine and the AI agent.

Two endpoints:
  POST /recommend  - direct call into engine.recommend(), for callers that already know
                      current_skills/target_skills (e.g. a dashboard, or a tester bypassing
                      the agent).
  POST /chat        - the AI Agent: a learner sends free text, the agent extracts skills and
                      goals via Gemini tool calls into engine.py, and replies in prose.

Run with: .venv/Scripts/uvicorn api:app --reload
"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from google.genai import errors as genai_errors
from pydantic import BaseModel

import agent
import engine

app = FastAPI(title="Skill-Gap Course Recommendation API")

# In-memory session store: fine for a single-process demo/capstone deployment. A restart or a
# second worker process loses in-flight conversations - there is no persistence layer here.
_CHAT_SESSIONS = {}


class RecommendRequest(BaseModel):
    current_skills: list[str]
    target_skills: list[str]
    skill_weights: Optional[dict[str, float]] = None
    top_n: int = 10


@app.post("/recommend")
def recommend(request: RecommendRequest):
    recommendations, gap = engine.recommend(
        request.current_skills,
        request.target_skills,
        skill_weights=request.skill_weights,
        top_n=request.top_n,
    )
    return {
        "recommendations": recommendations.to_dict(orient="records"),
        "gap": {
            "missing": sorted(gap.missing),
            "teachable": sorted(gap.teachable),
            "unteachable": sorted(gap.unteachable),
            "ceiling": round(gap.ceiling, 2),
        },
    }


class PersonRecommendRequest(BaseModel):
    target_skills: list[str]
    skill_weights: Optional[dict[str, float]] = None
    top_n: int = 10


@app.post("/recommend/{person_id}")
def recommend_for_person(person_id: int, request: PersonRecommendRequest):
    try:
        recommendations, gap = engine.recommend_for_person(
            person_id,
            request.target_skills,
            skill_weights=request.skill_weights,
            top_n=request.top_n,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "recommendations": recommendations.to_dict(orient="records"),
        "gap": {
            "missing": sorted(gap.missing),
            "teachable": sorted(gap.teachable),
            "unteachable": sorted(gap.unteachable),
            "ceiling": round(gap.ceiling, 2),
        },
    }


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session = _CHAT_SESSIONS.get(request.session_id)
    if session is None:
        try:
            session = agent.new_chat()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        _CHAT_SESSIONS[request.session_id] = session

    try:
        reply = session.send(request.message)
    except genai_errors.ClientError as exc:
        # Surface Gemini's own status (commonly 429 - the free tier is 20 requests/day per
        # model) instead of a bare 500, since this is a routine, expected failure mode.
        raise HTTPException(status_code=exc.code or 502, detail=str(exc)) from exc

    return ChatResponse(reply=reply)


@app.delete("/chat/{session_id}")
def end_chat(session_id: str):
    _CHAT_SESSIONS.pop(session_id, None)
    return {"ok": True}
