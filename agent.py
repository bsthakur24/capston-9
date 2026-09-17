"""AI Training Agent (target.md deliverable 3): a conversational layer over engine.py.

The agent's only jobs are (1) turning a resume/conversation into current_skills, (2) grounding
a stated career goal into target_skills, (3) inferring skill_weights when the learner signals
emphasis, and (4) phrasing the result. It never re-derives the ranking, coverage or bridging
itself - those stay in engine.py, deterministic and already evaluated (PROJECT_LOG.md, Protocol
B). This keeps the system auditable: every number in a response traces back to engine.py, not to
the model's own arithmetic.

Two tools are exposed to the model:
  - lookup_role_skills: grounds a stated career goal in real peer data (engine.get_peer_skills_for_role)
    instead of letting the model invent a target skill list from its own knowledge.
  - recommend_courses: the actual ranking call (engine.recommend).

Tool calling is driven manually (see TrainingAgent.send) rather than via the SDK's automatic
function calling: google-genai==2.23.0's AFC loop closes its internal HTTP client after the
first tool round-trip, so a turn needing two sequential tool calls (lookup then recommend, which
is the agent's main path) fails with "Cannot send a request, as the client has been closed."
A manual loop - request, execute any function_call parts ourselves, send the results back,
repeat until the model stops calling tools - sidesteps the bug and is unaffected if a future SDK
release fixes it.

Requires GOOGLE_API_KEY in the environment - see .env.example.
"""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

import engine

load_dotenv()

DEFAULT_MODEL = "gemini-3.6-flash"
MAX_TOOL_ROUNDS = 6

SYSTEM_INSTRUCTION = """
You are a training recommendation assistant for employees at a company. You help a learner go
from their current skills to a career goal by recommending courses from the company's catalogue.

Rules:
- Extract `current_skills` from whatever the learner tells you about their background (a resume
  paste, a list, or plain conversation). Use the skill names as the learner phrases them - do not
  try to rename them to match a catalogue; that normalisation happens on the engine side.
- When the learner states a career goal rather than a skill list (e.g. "I want to become a data
  scientist"), call `lookup_role_skills` first to ground that goal in real peer data before
  proposing target skills. Do not invent a target skill list purely from your own knowledge.
- Only pass `skill_weights` when the learner actually signals that some target skills matter more
  than others (e.g. "mostly interested in machine learning, data viz is a bonus"). Leave it unset
  otherwise - the engine already defaults to treating every missing skill equally.
- Call `recommend_courses` once you have both current_skills and target_skills. If either is
  unclear, ask a short clarifying question instead of guessing.
- When you present results, explain each course using the `covers` and `coverage` fields you get
  back - never state a skill is covered or a percentage that didn't come from the tool result.
- If `gap.unteachable` is non-empty, tell the learner plainly which of their target skills the
  catalogue cannot currently teach, instead of silently dropping them.
"""


def recommend_courses(current_skills: list[str], target_skills: list[str],
                       skill_weights: dict[str, float] | None = None, top_n: int = 10) -> dict:
    """Rank the course catalogue for a learner and return a shortlist plus gap diagnostics.

    Args:
        current_skills: skills the learner already has, in their own words.
        target_skills: skills the learner wants to reach, in their own words.
        skill_weights: optional relative importance per target skill (default 1.0 each).
        top_n: how many courses to return.
    """
    recommendations, gap = engine.recommend(
        current_skills, target_skills, skill_weights=skill_weights, top_n=top_n
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


def lookup_role_skills(role: str, top_n: int = 8) -> list[str]:
    """Look up the most common catalogue-mapped skills among learners in a similar role.

    Grounds a stated career goal (e.g. "data scientist", "cloud security engineer") in what
    people already in that role have on file, instead of guessing a skill list.

    Args:
        role: a job title or paraphrased career goal.
        top_n: how many skills to return.
    """
    return engine.get_peer_skills_for_role(role, top_n=top_n)


TOOL_FUNCTIONS = {f.__name__: f for f in [recommend_courses, lookup_role_skills]}


def build_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )
    return genai.Client(api_key=api_key)


class TrainingAgent:
    """A multi-turn chat session that drives tool calls itself (see module docstring for why)."""

    def __init__(self, client: genai.Client | None = None, model: str = DEFAULT_MODEL):
        self._client = client or build_client()
        declarations = [
            types.FunctionDeclaration.from_callable(client=self._client, callable=fn)
            for fn in TOOL_FUNCTIONS.values()
        ]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[types.Tool(function_declarations=declarations)],
            # Disabled deliberately - see module docstring.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self._chat = self._client.chats.create(model=model, config=config)

    def send(self, message) -> str:
        """Send a user message, executing any tool calls the model makes, and return its reply."""
        for _ in range(MAX_TOOL_ROUNDS):
            response = self._chat.send_message(message)
            calls = [
                part.function_call
                for part in response.candidates[0].content.parts
                if part.function_call
            ]
            if not calls:
                return response.text

            message = [
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": TOOL_FUNCTIONS[call.name](**call.args)},
                )
                for call in calls
            ]

        raise RuntimeError(f"agent did not finish after {MAX_TOOL_ROUNDS} tool-call rounds")


def new_chat(client: genai.Client | None = None, model: str = DEFAULT_MODEL) -> TrainingAgent:
    return TrainingAgent(client=client, model=model)


if __name__ == "__main__":
    session = new_chat()
    print("Training agent ready. Ctrl+C to exit.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        print(f"\nagent> {session.send(user_input)}\n")
