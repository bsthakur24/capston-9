"""Dashboard (target.md deliverable 4): chat with the agent, call the engine directly, and
browse the catalogue/learner data - all in one Streamlit app.

Run with: .venv/Scripts/streamlit run dashboard.py
"""
import os

import pandas as pd
import streamlit as st
from google import genai
from google.genai import errors as genai_errors

import agent
import engine

st.set_page_config(page_title="Skill-Gap Course Recommender", layout="wide")


# ---------------------------------------------------------------- sidebar: Gemini API key

st.sidebar.header("Gemini API key")
st.sidebar.caption(
    "Only needed for the Chat tab. Kept in this browser session only - never written to disk."
)
env_key = os.environ.get("GOOGLE_API_KEY", "")
entered_key = st.sidebar.text_input(
    "GOOGLE_API_KEY", value=st.session_state.get("google_api_key", ""),
    type="password", placeholder="paste your key, or leave blank to use .env",
)
active_key = entered_key.strip() or env_key

if entered_key.strip() and entered_key.strip() != st.session_state.get("google_api_key"):
    st.session_state["google_api_key"] = entered_key.strip()
    st.session_state.pop("agent_session", None)  # rebuild the chat session with the new key

if active_key:
    st.sidebar.success("Key loaded from the box above" if entered_key.strip() else "Key loaded from .env")
else:
    st.sidebar.warning("No key set - the Chat tab will not work until one is provided.")


def get_agent_session():
    """One TrainingAgent per browser session, rebuilt only when the key changes."""
    if "agent_session" not in st.session_state:
        client = genai.Client(api_key=active_key)
        st.session_state["agent_session"] = agent.new_chat(client=client)
    return st.session_state["agent_session"]


# ---------------------------------------------------------------- tabs

tab_chat, tab_recommend, tab_explore = st.tabs(
    ["Chat with Agent", "Direct Recommend", "Catalogue & Learner Explorer"]
)


# ---------------------------------------------------------------- Chat with Agent

with tab_chat:
    st.subheader("Talk to the training agent")
    st.caption(
        "Describe your background and a career goal in plain language. The agent grounds your "
        "goal in real peer skill data, then calls the recommendation engine - it never invents "
        "course rankings itself."
    )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for role, text in st.session_state["chat_messages"]:
        with st.chat_message(role):
            st.markdown(text)

    prompt = st.chat_input("e.g. I know SQL and Tableau, I want to become a data scientist")
    if prompt:
        if not active_key:
            st.error("Set a Gemini API key in the sidebar first.")
        else:
            st.session_state["chat_messages"].append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = get_agent_session().send(prompt)
                    except genai_errors.ClientError as exc:
                        reply = (
                            f"Gemini API error ({exc.code}): {exc.message}\n\n"
                            "If this is a 429, the free tier caps requests per day - wait for "
                            "the quota to reset or use a different key."
                        )
                    except genai_errors.ServerError as exc:
                        reply = (
                            f"Gemini API error ({exc.code}): {exc.message}\n\n"
                            "The model is temporarily overloaded on Google's side - wait a "
                            "moment and try sending your message again."
                        )
                    except RuntimeError as exc:
                        reply = str(exc)
                st.markdown(reply)
            st.session_state["chat_messages"].append(("assistant", reply))

    if st.session_state["chat_messages"] and st.button("Clear conversation"):
        st.session_state["chat_messages"] = []
        st.session_state.pop("agent_session", None)
        st.rerun()


# ---------------------------------------------------------------- Direct Recommend

with tab_recommend:
    st.subheader("Call the engine directly")
    st.caption("No LLM involved - bypasses Gemini entirely, so this never uses your API quota.")

    mode = st.radio("Learner input", ["Type skills", "Look up a stored person_id"], horizontal=True)

    col1, col2 = st.columns(2)

    if mode == "Type skills":
        with col1:
            current_text = st.text_area(
                "Current skills (one per line)", "sql\ntableau\ndata analysis", height=140
            )
        current_skills = [s.strip() for s in current_text.splitlines() if s.strip()]
        person_id = None
    else:
        with col1:
            person_id = st.number_input(
                "person_id", min_value=int(engine.user_profiles["person_id"].min()),
                max_value=int(engine.user_profiles["person_id"].max()), value=1, step=1,
            )
            match = engine.user_profiles.loc[engine.user_profiles["person_id"] == person_id]
            if match.empty:
                st.warning("No profile with that person_id.")
            else:
                st.text(f"name: {match.iloc[0]['name']}")
                st.text(f"career context: {match.iloc[0]['career_context']}")
        current_skills = None

    with col2:
        target_text = st.text_area(
            "Target skills (one per line)", "machine learning\ndeep learning\nstatistics", height=140
        )
    target_skills = [s.strip() for s in target_text.splitlines() if s.strip()]

    with st.expander("Optional: per-skill weights (overrides the default of 1.0 each)"):
        weights_text = st.text_area(
            "One 'skill: weight' per line, e.g. 'deep learning: 3.0'", "", height=80
        )
        skill_weights = {}
        for line in weights_text.splitlines():
            if ":" in line:
                skill, _, weight = line.partition(":")
                try:
                    skill_weights[skill.strip()] = float(weight.strip())
                except ValueError:
                    st.warning(f"Could not parse weight on line: {line!r}")
        skill_weights = skill_weights or None

    top_n = st.slider("How many courses", min_value=1, max_value=20, value=10)

    if st.button("Recommend", type="primary"):
        try:
            if mode == "Type skills":
                if not current_skills or not target_skills:
                    st.error("Enter at least one current skill and one target skill.")
                    st.stop()
                recommendations, gap = engine.recommend(
                    current_skills, target_skills, skill_weights=skill_weights, top_n=top_n
                )
            else:
                if not target_skills:
                    st.error("Enter at least one target skill.")
                    st.stop()
                recommendations, gap = engine.recommend_for_person(
                    int(person_id), target_skills, skill_weights=skill_weights, top_n=top_n
                )
        except KeyError as exc:
            st.error(str(exc))
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Missing skills", len(gap.missing))
            m2.metric("Teachable", len(gap.teachable))
            m3.metric("Unteachable", len(gap.unteachable))
            m4.metric("Best possible coverage", f"{gap.ceiling:.0%}")

            if gap.unteachable:
                st.info(f"No course in the catalogue teaches: {', '.join(sorted(gap.unteachable))}")

            st.dataframe(recommendations, width="stretch", hide_index=True)


# ---------------------------------------------------------------- Catalogue & Learner Explorer

with tab_explore:
    st.subheader("Catalogue & learner data")

    explore_tab1, explore_tab2, explore_tab3 = st.tabs(
        ["Courses", "Learner profiles", "Skill vocabulary"]
    )

    with explore_tab1:
        c1, c2, c3 = st.columns(3)
        title_query = c1.text_input("Search title or skills contains")
        difficulty_filter = c2.selectbox(
            "Difficulty", ["(any)"] + sorted(engine.course_profiles["difficulty"].dropna().unique())
        )
        org_filter = c3.selectbox(
            "Organization", ["(any)"] + sorted(engine.course_profiles["organization"].dropna().unique())
        )

        courses = engine.course_profiles
        if title_query:
            mask = (
                courses["title"].str.contains(title_query, case=False, na=False)
                | courses["skills_normalized"].str.contains(title_query, case=False, na=False)
            )
            courses = courses[mask]
        if difficulty_filter != "(any)":
            courses = courses[courses["difficulty"] == difficulty_filter]
        if org_filter != "(any)":
            courses = courses[courses["organization"] == org_filter]

        st.caption(f"{len(courses)} of {len(engine.course_profiles)} courses")
        st.dataframe(
            courses[["course_id", "title", "organization", "difficulty", "type", "duration",
                     "ratings", "skills_normalized"]].head(200),
            width="stretch", hide_index=True,
        )

    with explore_tab2:
        person_query = st.number_input(
            "Look up a person_id", min_value=int(engine.user_profiles["person_id"].min()),
            max_value=int(engine.user_profiles["person_id"].max()), value=1, step=1, key="explore_person",
        )
        profile = engine.user_profiles.loc[engine.user_profiles["person_id"] == person_query]
        if profile.empty:
            st.warning("No profile with that person_id.")
        else:
            row = profile.iloc[0]
            st.write(f"**{row['name']}**  (person_id={person_query})")
            st.write(f"Career context: {row['career_context']}")
            st.write(f"Skills on file ({row['skill_count']}): {row['skills_text']}")
            st.write(f"Education: {row['education_text'] or '(none on file)'}")

        st.divider()
        st.caption(f"{len(engine.user_profiles):,} learner profiles total")
        st.dataframe(
            engine.user_profiles[["person_id", "name", "career_context", "skill_count",
                                   "experience_count", "education_count"]].head(200),
            width="stretch", hide_index=True,
        )

    with explore_tab3:
        skill_vocab_path = engine.READY_DIR / "skill_vocabulary.csv"
        if skill_vocab_path.exists():
            skill_vocab = pd.read_csv(skill_vocab_path)
            skill_query = st.text_input("Search skill vocabulary")
            filtered = skill_vocab
            if skill_query:
                filtered = filtered[filtered["skill"].str.contains(skill_query, case=False, na=False)]
            st.caption(f"{len(filtered):,} of {len(skill_vocab):,} distinct skills")
            st.dataframe(
                filtered.sort_values("user_frequency", ascending=False).head(200),
                width="stretch", hide_index=True,
            )
        else:
            st.warning(f"{skill_vocab_path} not found - run notebook 02 to generate it.")
