"""Hybrid course recommendation engine, extracted from 03_recommendation_engine.ipynb.

Importable version of the same logic the notebook demonstrates and evaluates, so the AI
Agent (agent.py) and API (api.py) can call `recommend()` / `recommend_for_person()` directly
instead of re-running a notebook per request. See PROJECT_LOG.md and README.md for the design
rationale behind each signal, the skill bridge, and the two evaluation protocols - none of that
is repeated here, only the runtime path.
"""
import ast
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
READY_DIR = PROJECT_ROOT / "data" / "recommendation_ready"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# A learner skill maps onto a catalogue skill only above this cosine similarity.
# Below it we would be inventing a match - see the calibration table in notebook 03, Step 2.
BRIDGE_THRESHOLD = 0.60

# Bayesian prior strength for course quality, in units of reviews.
QUALITY_PRIOR_REVIEWS = 50

# Hybrid score weights. Must sum to 1.
WEIGHTS = {
    "coverage": 0.40,
    "semantic": 0.25,
    "tfidf": 0.15,
    "quality": 0.12,
    "difficulty": 0.08,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

# MMR trade-off: 1.0 = pure relevance, 0.0 = pure diversity.
MMR_LAMBDA = 0.75
SAME_PROVIDER_PENALTY = 0.85

# Role-group size floor for grounding a career goal in peer data (get_peer_skills_for_role).
MIN_ROLE_GROUP = 20

DISPLAY_COLUMNS = ["course_id", "title", "organization", "difficulty", "ratings"]

RANDOM_SEED = 42


# ---------------------------------------------------------------- data + indices

user_profiles = pd.read_csv(READY_DIR / "user_profiles.csv")
course_profiles = pd.read_csv(READY_DIR / "course_profiles.csv")

# Guard: notebook 02 removes the 3x stacked duplicates. If this fires, re-run 02.
_fingerprint = (
    user_profiles["name"].fillna("~").astype(str)
    + "||"
    + user_profiles["profile_text"].fillna("~").astype(str)
)
_repeat = len(user_profiles) / _fingerprint.nunique()
assert _repeat < 1.05, (
    f"learner profiles still contain ~{_repeat:.1f}x duplicates - re-run 02_recommendation_data.ipynb"
)

course_profiles["skills_list"] = course_profiles["skills_list"].map(
    lambda value: ast.literal_eval(value) if isinstance(value, str) else []
)
_empty_columns = [c for c in course_profiles.columns if course_profiles[c].isna().all()]
if _empty_columns:
    course_profiles = course_profiles.drop(columns=_empty_columns)

CATALOGUE_SKILLS = sorted({s for row in course_profiles["skills_list"] for s in row})
SKILL_INDEX = {skill: i for i, skill in enumerate(CATALOGUE_SKILLS)}

COURSE_SKILL_MATRIX = np.zeros((len(course_profiles), len(CATALOGUE_SKILLS)), dtype=np.float32)
for _row, _skills in enumerate(course_profiles["skills_list"]):
    for _skill in _skills:
        COURSE_SKILL_MATRIX[_row, SKILL_INDEX[_skill]] = 1.0

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
CATALOGUE_SKILL_EMBEDDINGS = embedding_model.encode(
    CATALOGUE_SKILLS, normalize_embeddings=True, show_progress_bar=False
)


def _course_semantic_text(row):
    skills = ", ".join(row["skills_list"])
    description = " ".join(str(row.get("course_description_clean", "")).split()[:60])
    return f"A course teaching {skills}. {row['title']}. {description}".strip()


course_profiles["semantic_text"] = course_profiles.apply(_course_semantic_text, axis=1)
COURSE_EMBEDDINGS = embedding_model.encode(
    course_profiles["semantic_text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=False,
    batch_size=64,
)

SKILL_SEPARATOR = " ||| "


def _skill_tokenizer(text):
    return [token for token in text.split(SKILL_SEPARATOR) if token]


tfidf_vectorizer = TfidfVectorizer(
    tokenizer=_skill_tokenizer, preprocessor=None, token_pattern=None, lowercase=False
)
COURSE_TFIDF = tfidf_vectorizer.fit_transform(
    course_profiles["skills_list"].map(SKILL_SEPARATOR.join)
)

_ratings = course_profiles["ratings"].astype(float)
_reviews = course_profiles["review_count"].fillna(0).astype(float)
_prior_mean = _ratings.mean()
_smoothed = (
    (_reviews / (_reviews + QUALITY_PRIOR_REVIEWS)) * _ratings
    + (QUALITY_PRIOR_REVIEWS / (_reviews + QUALITY_PRIOR_REVIEWS)) * _prior_mean
)
QUALITY_SCORE = ((_smoothed - _smoothed.min()) / (_smoothed.max() - _smoothed.min())).to_numpy()
course_profiles["quality_score"] = QUALITY_SCORE

DIFFICULTY_RANK = {"beginner": 0, "mixed": 1, "intermediate": 1, "advanced": 2}
COURSE_DIFFICULTY = (
    course_profiles["difficulty"].fillna("mixed").str.lower().str.strip()
    .map(DIFFICULTY_RANK).fillna(1).to_numpy()
)

POPULARITY_TOP = course_profiles.iloc[np.argsort(-QUALITY_SCORE)]["course_id"].tolist()


# ---------------------------------------------------------------- skill bridge

def normalize_skill(skill):
    """Same normalisation as notebook 02, so both sides share one surface form."""
    if skill is None or (isinstance(skill, float) and np.isnan(skill)):
        return None

    skill = str(skill).lower().strip()
    skill = skill.replace("&", " and ").replace("/", " ").replace("-", " ")
    skill = re.sub(r"[^a-z0-9+#.\s]", " ", skill)
    skill = re.sub(r"\s+", " ", skill).strip()

    return skill or None


_bridge_cache = {}


def bridge_skills(skills, threshold=BRIDGE_THRESHOLD, explain=False):
    """Map arbitrary skill strings onto the course-catalogue skill space.

    Returns the set of matched catalogue skills, or - when `explain` is True - a DataFrame
    showing how each input skill was resolved.
    """
    normalized = [normalize_skill(skill) for skill in skills]
    normalized = [skill for skill in normalized if skill]

    resolved, unknown = {}, []
    for skill in normalized:
        if skill in SKILL_INDEX:
            resolved[skill] = (skill, 1.0, "exact")
        elif skill in _bridge_cache:
            resolved[skill] = _bridge_cache[skill]
        else:
            unknown.append(skill)

    if unknown:
        vectors = embedding_model.encode(
            unknown, normalize_embeddings=True, show_progress_bar=False
        )
        similarity = vectors @ CATALOGUE_SKILL_EMBEDDINGS.T
        best = similarity.argmax(axis=1)
        for skill, index, score in zip(unknown, best, similarity.max(axis=1)):
            score = float(score)
            match = (
                (CATALOGUE_SKILLS[index], score, "semantic")
                if score >= threshold
                else (None, score, "unmapped")
            )
            _bridge_cache[skill] = match
            resolved[skill] = match

    if explain:
        return pd.DataFrame(
            [
                {"input": skill, "mapped_to": match, "similarity": round(score, 3), "method": how}
                for skill, (match, score, how) in resolved.items()
            ]
        ).sort_values(["method", "similarity"], ascending=[True, False])

    return {match for match, _, _ in resolved.values() if match is not None}


def _bridged_unmapped(skills):
    """Same resolution as bridge_skills, but also reports what did not clear the threshold."""
    explained = bridge_skills(skills, explain=True)
    if explained.empty:
        return set(), []
    mapped = set(explained.loc[explained["method"] != "unmapped", "mapped_to"])
    unmapped = sorted(explained.loc[explained["method"] == "unmapped", "input"])
    return mapped, unmapped


# ---------------------------------------------------------------- skill gap

class SkillGap:
    """The learner's missing skills, their weights, and what the catalogue can actually teach."""

    def __init__(self, current_skills, target_skills, skill_weights=None):
        self.current = bridge_skills(current_skills)
        self.target = bridge_skills(target_skills)
        self.missing = self.target - self.current

        # Weights arrive keyed by the agent's surface form, so bridge them too.
        bridged_weights = {}
        for raw, weight in (skill_weights or {}).items():
            for mapped in bridge_skills([raw]):
                bridged_weights[mapped] = float(weight)

        self.weights = {skill: bridged_weights.get(skill, 1.0) for skill in self.missing}
        self.teachable = {s for s in self.missing if s in SKILL_INDEX}
        self.unteachable = self.missing - self.teachable

        self.total_weight = sum(self.weights.values())
        self.teachable_weight = sum(self.weights[s] for s in self.teachable)

    @property
    def ceiling(self):
        """Best coverage score any single course could achieve for this learner."""
        if self.total_weight == 0:
            return 0.0
        return self.teachable_weight / self.total_weight

    @property
    def is_empty(self):
        return len(self.missing) == 0

    def vector(self):
        """Weighted gap vector over the catalogue skill space."""
        vector = np.zeros(len(CATALOGUE_SKILLS), dtype=np.float32)
        for skill in self.teachable:
            vector[SKILL_INDEX[skill]] = self.weights[skill]
        return vector

    def __repr__(self):
        return (
            f"SkillGap(missing={len(self.missing)}, teachable={len(self.teachable)}, "
            f"unteachable={len(self.unteachable)}, ceiling={self.ceiling:.2f})"
        )


# ---------------------------------------------------------------- signals

def coverage_signal(gap):
    if gap.total_weight == 0:
        return np.zeros(len(course_profiles), dtype=np.float32)
    return (COURSE_SKILL_MATRIX @ gap.vector()) / gap.total_weight


def tfidf_signal(gap):
    if not gap.teachable:
        return np.zeros(len(course_profiles), dtype=np.float32)
    query = tfidf_vectorizer.transform([SKILL_SEPARATOR.join(sorted(gap.teachable))])
    return cosine_similarity(query, COURSE_TFIDF).ravel()


def semantic_signal(gap):
    if not gap.missing:
        return np.zeros(len(course_profiles), dtype=np.float32)
    query = f"A course teaching {', '.join(sorted(gap.missing))}."
    vector = embedding_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return np.clip(cosine_similarity(vector, COURSE_EMBEDDINGS).ravel(), 0.0, 1.0)


def difficulty_signal(gap):
    """1.0 for an exact level match, 0.5 one level away, 0.2 two levels away."""
    if not gap.target:
        return np.full(len(course_profiles), 0.5, dtype=np.float32)

    held = len(gap.target & gap.current) / len(gap.target)
    learner_level = 0 if held < 0.34 else (1 if held < 0.67 else 2)

    distance = np.abs(COURSE_DIFFICULTY - learner_level)
    return np.select([distance == 0, distance == 1], [1.0, 0.5], default=0.2).astype(np.float32)


def score_catalogue(gap):
    """All five signals plus the hybrid score, one row per course."""
    signals = pd.DataFrame(
        {
            "coverage": coverage_signal(gap),
            "semantic": semantic_signal(gap),
            "tfidf": tfidf_signal(gap),
            "quality": QUALITY_SCORE,
            "difficulty": difficulty_signal(gap),
        },
        index=course_profiles.index,
    )
    signals["hybrid_score"] = sum(signals[name] * weight for name, weight in WEIGHTS.items())
    return signals


def mmr_rerank(candidate_index, scores, top_n, lambda_=MMR_LAMBDA):
    """Greedy MMR over course embeddings. Returns positional indices, best first.

    Redundancy is the larger of content similarity and a flat penalty for reusing a provider,
    since two different Google certificates can be near-duplicates in intent while sitting far
    enough apart in embedding space to both survive.
    """
    candidates = list(candidate_index)
    if not candidates:
        return []

    embeddings = COURSE_EMBEDDINGS[candidates]
    content_similarity = embeddings @ embeddings.T

    providers = course_profiles.loc[candidates, "organization"].fillna("").to_numpy()
    same_provider = (providers[:, None] == providers[None, :]).astype(np.float64)

    redundancy_matrix = np.maximum(content_similarity, same_provider * SAME_PROVIDER_PENALTY)
    score = np.asarray([scores[i] for i in candidates], dtype=np.float64)

    selected = [int(score.argmax())]
    while len(selected) < min(top_n, len(candidates)):
        redundancy = redundancy_matrix[:, selected].max(axis=1)
        mmr = lambda_ * score - (1.0 - lambda_) * redundancy
        mmr[selected] = -np.inf
        selected.append(int(mmr.argmax()))

    return [candidates[i] for i in selected]


# ---------------------------------------------------------------- public interface

def recommend(current_skills, target_skills, skill_weights=None, top_n=10,
              diversify=True, explain=True):
    """Rank the catalogue for one learner.

    Returns (recommendations, gap). `gap` carries the diagnostics - which target skills the
    catalogue cannot teach, and the best coverage any course could reach.
    """
    gap = SkillGap(current_skills, target_skills, skill_weights)
    signals = score_catalogue(gap)

    # Keep anything with evidence from at least one signal; fall back to the whole catalogue.
    has_evidence = (
        (signals["coverage"] > 0) | (signals["tfidf"] > 0) | (signals["semantic"] >= 0.40)
    )
    candidates = signals.index[has_evidence]
    if len(candidates) < top_n:
        candidates = signals.index

    if diversify:
        order = mmr_rerank(candidates, signals["hybrid_score"].to_dict(), top_n)
    else:
        order = signals.loc[candidates, "hybrid_score"].nlargest(top_n).index.tolist()

    result = course_profiles.loc[order, DISPLAY_COLUMNS].copy()
    result.insert(0, "rank", range(1, len(order) + 1))
    result["covers"] = [
        ", ".join(sorted(gap.teachable & set(course_profiles.loc[i, "skills_list"]))) or "-"
        for i in order
    ]
    if explain:
        # `difficulty` is already a display column holding the course's level, so the signal
        # is surfaced under a distinct name rather than overwriting it.
        for column in ["coverage", "semantic", "tfidf", "quality", "difficulty", "hybrid_score"]:
            label = "difficulty_fit" if column == "difficulty" else column
            result[label] = signals.loc[order, column].round(3).to_numpy()

    return result.reset_index(drop=True), gap


def recommend_for_person(person_id, target_skills, **kwargs):
    """Derive `current_skills` from a stored learner profile, then recommend."""
    matches = user_profiles.loc[user_profiles["person_id"] == person_id]
    if matches.empty:
        raise KeyError(f"person_id {person_id} not found")

    profile = matches.iloc[0]
    current = ast.literal_eval(profile["skills"]) if isinstance(profile["skills"], str) else []
    return recommend(current, target_skills, **kwargs)


# ---------------------------------------------------------------- peer-grounded goal lookup
#
# Supports the AI Agent's target_skills step: rather than asking an LLM to invent a skill list
# for a stated career goal, ground it in what people already in that role actually have on file.
# This is the same peer-group idea PROJECT_LOG.md's Protocol B evaluates (7.65x lift over
# popularity), reused here as a lookup instead of an evaluation.

def primary_role(career_context):
    """Last title in a pipe-joined career context, e.g. 'sr. python developer | python developer'."""
    if not isinstance(career_context, str) or not career_context.strip():
        return None
    return career_context.split("|")[-1].strip() or None


_BRIDGED_SKILLS_CACHE = {}


def bridged_skills_for(profile):
    """Bridged catalogue skills for one learner profile, memoised by person_id."""
    person_id = int(profile["person_id"])
    if person_id not in _BRIDGED_SKILLS_CACHE:
        raw = ast.literal_eval(profile["skills"]) if isinstance(profile["skills"], str) else []
        _BRIDGED_SKILLS_CACHE[person_id] = frozenset(bridge_skills(raw))
    return _BRIDGED_SKILLS_CACHE[person_id]


_role_profiles = user_profiles.copy()
_role_profiles["role"] = _role_profiles["career_context"].map(primary_role)
_role_profiles = _role_profiles.dropna(subset=["role"])
_role_sizes = _role_profiles["role"].value_counts()
ELIGIBLE_ROLES = _role_sizes[_role_sizes >= MIN_ROLE_GROUP].index.tolist()
_ROLE_EMBEDDINGS = (
    embedding_model.encode(ELIGIBLE_ROLES, normalize_embeddings=True, show_progress_bar=False)
    if ELIGIBLE_ROLES else np.zeros((0, 384), dtype=np.float32)
)


def get_peer_skills_for_role(stated_role, top_n=8):
    """Nearest known job-title cluster to `stated_role`, and its most common bridged skills.

    `stated_role` is free text from a learner's stated goal (e.g. "data scientist" or
    "I want to move into cloud security"). Returns [] if no role group is large enough to trust.
    """
    if not ELIGIBLE_ROLES:
        return []

    query_embedding = embedding_model.encode(
        [stated_role], normalize_embeddings=True, show_progress_bar=False
    )
    similarities = (query_embedding @ _ROLE_EMBEDDINGS.T).ravel()
    nearest_role = ELIGIBLE_ROLES[int(similarities.argmax())]

    peers = _role_profiles[_role_profiles["role"] == nearest_role]
    counter = Counter()
    for _, profile in peers.iterrows():
        counter.update(bridged_skills_for(profile))

    return [skill for skill, _ in counter.most_common(top_n)]
