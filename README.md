# Skill-Gap Course Recommendation Engine

Recommends Coursera courses that close the gap between a learner's **current skills** and the
**target skills** required for their career goal.

The engine is built to sit behind an AI Agent: the agent infers `current_skills`, `target_skills`
and optional per-skill importance weights from a conversation or a resume, and the engine turns
those into a ranked, explainable course list.

```python
recommendations, gap = recommend(
    current_skills=["sql", "tableau", "data analysis"],
    target_skills=["machine learning", "deep learning", "statistics"],
    skill_weights={"deep learning": 3.0},
    top_n=10,
)
```

---

## Pipeline

```
data/raw/                01_data_cleaning.ipynb      standardise, de-orphan, type-coerce
   |                              |
   v                              v
data/cleaned/            02_recommendation_data.ipynb  reshape to profiles, repair data quality
   |                              |
   v                              v
data/recommendation_ready/  03_recommendation_engine.ipynb  five signals -> hybrid score -> MMR
                                  |
                                  v
                          engine.py  (03's logic, extracted so agent.py/api.py can import it)
                                  |
                    +-------------+-------------+
                    v                           v
             agent.py + api.py             04_analytics_report.ipynb
             (AI Agent, FastAPI)           (business-facing metrics, charts)
                    |
                    v
             dashboard.py (Streamlit: chat, direct recommend, data explorer)
```

Run the notebooks in order. Notebook 02 falls back to rebuilding its inputs from
`data/recommendation_ready/` if `data/cleaned/` is absent, so the pipeline is runnable even
without the original raw files.

`data/raw/` is empty in this checkout, so **notebook 01 ships with its outputs cleared** — it
needs the five raw CSVs to run. Notebooks 02 and 03 are committed with their outputs intact and
were executed against the data in `data/recommendation_ready/`.

See **[`PROJECT_LOG.md`](PROJECT_LOG.md)** for the change-by-change record — what was done, the
reason for each change, and the outstanding task list.

| Notebook / module | Purpose |
|---|---|
| `01_data_cleaning.ipynb` | Column/text normalisation, referential integrity, numeric coercion, duplicate-person diagnostic |
| `02_recommendation_data.ipynb` | Learner and course profiles, skill vocabulary, **the two data repairs below** |
| `03_recommendation_engine.ipynb` | Skill bridge, five ranking signals, hybrid score, MMR, evaluation |
| `04_analytics_report.ipynb` | Business-facing metrics and charts: skill supply/demand gaps, goal-teachability distribution, Protocol A/B recall vs. baselines, MMR trade-off |
| `engine.py` | `03`'s logic, extracted into an importable module so `agent.py`/`api.py`/`dashboard.py` share one scoring path instead of re-running a notebook |
| `agent.py` | AI Training Agent - Gemini tool-calling over `engine.py` (`GOOGLE_API_KEY` required, see `.env.example`) |
| `api.py` | FastAPI service: `/recommend`, `/recommend/{person_id}`, `/chat` |
| `dashboard.py` | Streamlit app: chat with the agent, call the engine directly, browse the catalogue/learner data |

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

The first run downloads the `all-MiniLM-L6-v2` sentence-transformer (~90 MB) from Hugging Face.
Everything runs on CPU; no GPU is required.

---

## Data repairs

Two problems in the prepared data were material enough to invalidate downstream statistics.

**1. The learner population was stacked three times.** Every profile appeared three times under
three different `person_id`s — the raw export is three concatenated copies, with ids offset by a
constant 18,311. Because the ids differed, de-duplicating on `person_id` kept all three copies.

```
learner profiles   54,933  ->  18,194   (3.02x)
```

Left uncorrected this inflates every skill-frequency count threefold and puts the same person on
both sides of any train/test split.

**2. A third of the skill vocabulary was resume prose.** Skills harvested from resumes included
entire bullet points, e.g. `"maintain multiple database environments redshift rds in aws"`.

The course catalogue supplies a ground truth for the shape of a real skill — all 323 catalogue
skills are 1–5 words, averaging 2.0 — so the filter combines a length bound with rules for
bullet-leading verbs and prose connectives. It is tuned to keep **100% of the 323 known-good
catalogue skills** as a positive control.

```
distinct skills   201,934  ->  132,943
```

---

## How the engine ranks

Five signals, each on a 0–1 scale that does not depend on the candidate set:

| Signal | Weight | Notes |
|---|---|---|
| Weighted skill-gap coverage | 0.40 | fraction of the learner's weighted missing skills the course teaches |
| Semantic similarity | 0.25 | MiniLM embeddings, symmetric query/document phrasing |
| TF-IDF similarity | 0.15 | skill-level tokens, so IDF rewards rare skills |
| Course quality | 0.12 | Bayesian-smoothed rating |
| Difficulty fit | 0.08 | course level vs. inferred learner level |

The top candidates are then re-ordered with **MMR** (λ = 0.75). Redundancy is the larger of
content similarity and a flat penalty for reusing a provider, since two Google certificates can be
near-duplicates in intent while sitting far enough apart in embedding space for both to survive.
On the worked example this lowers mean pairwise similarity in the top 10 from 0.541 to 0.523 and
raises distinct providers from 7 to 8, costing 0.006 of mean hybrid score.

Note that the `rank` column reflects MMR order, so `hybrid_score` is deliberately not monotonic
down the list.

### The learner→catalogue skill bridge

The learner vocabulary has 132,943 distinct skills; the catalogue has 323. Only **237** match
exactly, which leaves the average profile matching ~1.5 catalogue skills — far too sparse for
skill-gap matching to work.

The bridge resolves each skill in three stages: normalise → exact match → nearest catalogue
neighbour by embedding cosine, accepted only above `BRIDGE_THRESHOLD = 0.60`. A skill with no
close neighbour is reported as *unmapped* rather than forced onto a wrong match. This is what
replaces the 12 hand-written aliases in the original draft.

### Honest coverage accounting

If a target skill is taught by no course, it stays in the coverage denominator. Every call also
returns `gap.ceiling` — the best coverage any single course could possibly achieve — so a 0.4
score against a ceiling of 0.4 is recognisable as a perfect result rather than a mediocre one.

### Why there is no `MinMaxScaler`

The original engine min-max scaled the signals *after* filtering to the candidate pool. That made
scores incomparable between learners (the pool's best course is forced to exactly 1.0 regardless
of how good it actually is) and discarded the fixed interpretation of coverage. Since every signal
is already 0–1 and query-independent, they are now combined directly.

---

## Evaluation

There is no interaction log, so relevance is simulated. Notebook 03 runs **two** protocols,
because the obvious one flatters the engine.

**Protocol A — retrieval check.** Hide a third of a learner's skills and hand *those same skills*
to the engine as the target. This is close to tautological — the engine optimises coverage of
exactly what it was asked for — so a high score is the expected outcome. It is a correctness test
that catches a broken ranker, bridge or index, not evidence of recommendation quality.

**Protocol B — peer-target prediction.** The engine never sees the held-out skills. The target is
built from the learner's **peers**: the most common skills among other people with the same job
title. Recall of the hidden skills then measures genuine generalisation — *given who this person
is and what their role demands, can we surface what they are missing?*

Both run against two baselines (highest-rated-always, and random), over 150 learners each.

| recall@10 | Protocol A (retrieval) | Protocol B (peer targets) |
|---|---|---|
| **engine** | **0.979** | **0.750** |
| random | 0.374 | 0.378 |
| popularity | 0.110 | 0.098 |
| lift vs popularity | 8.90× | 7.65× |

**Only Protocol B should be read as recommendation quality.** Protocol A's 0.979 is the expected
result of asking the engine to find precisely what it was told to find; its value is that the
baselines stay low, which shows a 404-course catalogue does not make the task trivial.

Protocol B is the real number: **0.750 recall of skills the engine never saw**, with targets
assembled from other people in the same role — 7.65× the popularity baseline and roughly double
random.

---

## Limits

* **The catalogue is small.** 404 courses covering 323 distinct skills. Many realistic learner
  goals are simply not teachable by it; `gap.ceiling` exists to make that visible per request.
* **Quality is a weak signal by construction.** Ratings are compressed (mean 4.68, σ 0.17) and
  `review_count` is missing for 75% of courses. `course_students_enrolled` is empty for all 404
  and is dropped.
* **The evaluation measures skill retrieval, not learning outcomes.** It cannot capture teaching
  quality, prerequisites or ordering.
* **Signal weights are reasoned defaults, not learned.** With real interaction data they should be
  fit rather than assumed.
* **Notebook 02's fallback loader is lossy.** Rebuilding from `data/recommendation_ready/` cannot
  recover institution, firm or location. Re-run `01` from `data/raw/` for full fidelity.
