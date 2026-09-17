# Presentation Script — AI Powered Course Recommendation Engine

Speaker notes for each slide, in order. Read the **cue** for what's on screen, then the **script**
for what to say. Numbers in scripts match the deck exactly — no rounding, no rephrasing of stats.

---

## Slide 1 — Title

**On screen:** AI Powered Course Recommendation Engine · tagline · Team-9 (5 names)

**Script:**
> Good [morning/afternoon] everyone. We're Team 9, and today we're presenting our capstone
> project: an AI-powered course recommendation engine. In one line — it matches employees to
> the right courses, and proves why, every time. Over the next few minutes we'll walk through
> the problem we set out to solve, how the system works end-to-end, the data issues we had to
> fix before any of it could be trusted, and the results we measured.

---

## Slide 2 — Agenda

**On screen:** 7-item agenda, Business Problem → Analytics/Limits/Roadmap

**Script:**
> Here's the roadmap for the talk. We'll start with the business problem — why generic training
> doesn't work. Then the solution and architecture — five deliverables built around one shared
> engine. We'll cover the data quality issues we found and fixed, how the recommendation engine
> actually ranks courses, and how we evaluated it honestly using two different protocols.
> Finally, we'll show the AI agent, API and dashboard that serve it, the business-facing
> analytics, and close with limits and what's next.

---

## Slide 3 — The Problem

**On screen:** "Training is one-size-fits-all" · pull-quote · TODAY vs THE GOAL panels

**Script:**
> The starting point is a real business problem: training departments need customized learning
> plans based on each employee's individual needs and career goals. Today, that's not what
> happens — the same catalogue gets pushed at everyone. That causes four things: low engagement
> because irrelevant courses feel like noise, training spend that's misaligned with actual career
> needs, no visibility into *why* a course was recommended, and no way to see which skill gaps
> the catalogue can't even address. So the goal we set for this system was simple to state and
> hard to build: recommend the right course to the right person, explain why, and expose the
> gaps training can't yet fill.

---

## Slide 4 — Five Deliverables

**On screen:** 5 cards — Recommendation Engine, User Profiling, AI Agent, Dashboard, Analytics Report

**Script:**
> We didn't just build a model — we shipped five deliverables around it. First, the
> recommendation engine itself, a hybrid explainable scoring system in `engine.py`. Second, a
> user profiling module that builds learner profiles out of raw HR and resume data. Third, an AI
> training agent that lets someone describe their goals in plain conversation, using Gemini
> tool-calling. Fourth, a Streamlit dashboard with chat, direct recommend, and a data explorer.
> And fifth, a business-facing analytics report. Together, the expected impact is higher course
> engagement and completion, a genuinely personalized learning experience, and clear visibility
> into skill development.

---

## Slide 5 — End-to-End Architecture

**On screen:** offline pipeline → engine.py core → agent/api/dashboard serving layer

**Script:**
> Here's how it all fits together. Offline, we have a data pipeline: raw CSVs get cleaned,
> turned into learner and course profiles, and then notebook 03 designs the five ranking signals
> and the MMR re-ranking. That logic gets extracted into one shared module, `engine.py` — this is
> the core design decision in the whole project. Every consumer downstream — the AI agent, the
> FastAPI service, and the Streamlit dashboard — calls that *same* scoring implementation. So the
> ranking logic can never drift apart between the notebook, the agent, the API, and the UI.

---

## Slide 6 — Data Pipeline Stages

**On screen:** table — Clean / Profile / Design / Report / Runtime / Agent / API / UI

**Script:**
> Breaking that pipeline down stage by stage: notebook 01 cleans and normalizes the raw data.
> Notebook 02 builds learner and course profiles and does the two data repairs we're about to
> cover. Notebook 03 designs the skill bridge, the five signals, the hybrid score, and MMR.
> Notebook 04 turns all of that into business-facing analytics. And `engine.py` is notebook 03's
> logic extracted into an importable module, so the agent, the API, and the dashboard all share
> one scoring path instead of re-running a notebook.

---

## Slide 7 — Data Quality Repair 1: 3× Duplication

**On screen:** 54,933 → 18,194 · 3.02× reduction

**Script:**
> Before we could trust any statistic, we had to fix the data. The first issue: every learner
> appeared *three times* in the raw export, under three different person IDs offset by a constant
> 18,311 — with every other column byte-identical. A naive `drop_duplicates` on person ID
> couldn't catch this, because the IDs were different. Left uncorrected, this would have inflated
> every skill-frequency statistic threefold, and worse — it could put the same person on both
> sides of a train/test split, making any accuracy number meaningless. Our fix: de-duplicate on a
> content fingerprint — name plus profile text — instead of the untrustworthy ID. That took us
> from 54,933 raw rows down to 18,194 true learners, a 3.02x reduction, and we added an automated
> diagnostic so this can never slip through again.

---

## Slide 8 — Data Quality Repair 2: Resume Prose

**On screen:** 201,934 → 132,943 · 100% catalogue retention

**Script:**
> The second issue was in the skill vocabulary itself. A third of the "skills" harvested from
> resumes were actually entire bullet points of prose — things like "maintain multiple database
> environments redshift rds in aws" — which will never match a real course. To fix this, we used
> the 404-course catalogue as ground truth for what a real skill looks like: one to five words,
> averaging two. We built a length-and-grammar filter and tuned it against a positive control — it
> had to keep 100% of the 323 known-good catalogue skills. That took the vocabulary from 201,934
> distinct entries down to 132,943 real skills.

---

## Slide 9 — Learner → Catalogue Skill Bridge

**On screen:** flow: raw skill → normalise → exact match → semantic match · two stat cards

**Script:**
> Even with clean skills, we had a matching problem: 132,943 learner skills against only 323
> catalogue skills. Exact string matching alone found just 237 overlaps — leaving the average
> learner profile matched to only about one and a half catalogue skills, far too sparse to do
> gap analysis. So we built a three-stage bridge: normalize the skill text, try an exact match,
> and if that fails, embed it with MiniLM and compare against all 323 catalogue skills — accepting
> the nearest neighbor only above a cosine similarity of 0.60. If nothing clears that bar, the
> skill is reported as unmapped rather than forced onto a wrong match. That bridge took matched
> catalogue skills per profile from 1.47 up to 7.23, and cut zero-match profiles from 32.7% down
> to 6.7%.

---

## Slide 10 — Five Ranking Signals

**On screen:** table — coverage 0.40, semantic 0.25, TF-IDF 0.15, quality 0.12, difficulty 0.08

**Script:**
> Once skills are bridged, the engine scores every candidate course on five signals, each on a
> zero-to-one scale. The biggest weight, 0.40, is weighted skill-gap coverage — how much of the
> learner's *weighted* missing skills the course teaches. Then semantic similarity at 0.25, using
> MiniLM embeddings; TF-IDF similarity at 0.15, which rewards rare skills; course quality at 0.12,
> using a Bayesian-smoothed rating so a 4.9 star from three reviewers can't outrank a 4.6 star
> from thousands; and difficulty fit at 0.08. Because all five signals are independently 0 to 1
> and don't depend on the candidate pool, we sum them directly — no MinMaxScaler. An earlier
> version scaled after filtering to the candidate pool, which forced the best course in *any* pool
> to a score of 1.0 regardless of how good it actually was — that made scores incomparable between
> learners, so we removed it.

---

## Slide 11 — MMR Re-Ranking for Diversity

**On screen:** before/after MMR stats — similarity 0.541→0.523, providers 7→8

**Script:**
> Two providers, Google and IBM, hold about a quarter of the catalogue, and their specializations
> are often near-duplicates. Ranking on score alone gave repetitive top-10 lists. So we re-rank
> the top candidates with Maximal Marginal Relevance, lambda of 0.75, where redundancy is defined
> as the *larger* of embedding similarity or a same-provider penalty — because two courses from
> the same provider can be near-duplicate in intent while sitting far apart in embedding space.
> The result: mean pairwise similarity in the top 10 dropped from 0.541 to 0.523, distinct
> providers went from 7 to 8, at a cost of just 0.006 off the mean hybrid score.

---

## Slide 12 — Honest Coverage Accounting

**On screen:** gap.ceiling example — 0.40 score vs 0.40 ceiling

**Script:**
> One design principle we're proud of: if a target skill is taught by *no* course in the
> catalogue, it still counts against the coverage denominator — it doesn't quietly disappear from
> the math. And every result also returns `gap.ceiling`, the best coverage any single course could
> possibly achieve for that learner. So if you see a score of 0.40 against a ceiling of 0.40,
> that's not a mediocre result — it's a *perfect* one, correctly labeled as such instead of looking
> like a weak recommendation.

---

## Slide 13 — Evaluation Methodology

**On screen:** Protocol A (retrieval check) vs Protocol B (peer-target prediction)

**Script:**
> There's no real interaction log, so we had to simulate relevance carefully — and we used two
> protocols on purpose, because the obvious one flatters the engine. Protocol A hides a third of
> a learner's skills and hands those *same* skills back as the target — this is close to
> tautological, since the engine is optimizing for exactly what it was told to find. Its value is
> as a correctness check: confirming the ranker, the bridge, and the index aren't broken. Protocol
> B is the honest one: the target is built from the learner's peers — the most common skills among
> other people with the same job title — and the engine never sees the held-out skills. That tests
> genuine generalization: given who this person is and what their role demands, can the system
> actually surface what they're missing?

---

## Slide 14 — Evaluation Results

**On screen:** table — Engine 0.979/0.750, Random 0.374/0.378, Popularity 0.110/0.098

**Script:**
> Here are the numbers, across 150 learners, measuring recall at 10. On Protocol A, the engine
> scores 0.979. On Protocol B — the honest test — it scores 0.750. Compare that to baselines: a
> random baseline scores 0.374 and 0.378, and a popularity baseline scores just 0.110 and 0.098.
> That's a lift of 7.65x over popularity on Protocol B. And that's the headline number we want you
> to remember: 0.750 recall on skills the engine never saw, using peer-derived targets — that's
> real recommendation quality, not an artifact of the test setup.

---

## Slide 15 — The AI Training Agent

**On screen:** flow — learner message → lookup_role_skills() → recommend_courses()

**Script:**
> On top of the engine sits a conversational AI agent. The core design principle here: the LLM
> never invents a ranking or a skill list on its own — it only extracts what the learner said and
> grounds every claim by calling into `engine.py`. So if a learner says "I know SQL and Tableau,
> and I want to become a data scientist," the agent extracts the current skills, calls
> `lookup_role_skills` to get peer-derived target skills for that role, then calls
> `recommend_courses` to get a ranked list with gap diagnostics. Every number in the agent's reply
> traces back to deterministic code, not the model's own arithmetic. We did hit a real engineering
> bug here: the Gemini SDK's automatic function-calling closed its internal HTTP client after the
> first tool call, breaking any turn that needed two sequential tool calls — which is the agent's
> main path. We fixed it by driving the tool-call loop manually instead of relying on the SDK's
> automatic handling.

---

## Slide 16 — API & Dashboard

**On screen:** FastAPI endpoints table · Streamlit dashboard's 3 tabs

**Script:**
> All of this is exposed two ways. A FastAPI service with three endpoints: a direct
> `/recommend` call that bypasses the LLM entirely and costs no API quota, a per-person version
> that pulls stored skills automatically, and a `/chat` endpoint that puts the AI agent behind
> HTTP, with Gemini rate-limit errors surfaced as proper HTTP status codes instead of a bare 500.
> And a Streamlit dashboard with three tabs: chat with the agent, a direct-recommend form for
> bypassing Gemini, and a catalogue and learner explorer for browsing the underlying data.

---

## Slide 17 — Analytics Report Findings

**On screen:** Catalogue blind spots · Goal teachability 93.5% · Independent re-verification

**Script:**
> Because our analytics notebook imports `engine.py` directly, every business chart traces to the
> exact code path the agent and API use — nothing is recomputed separately. Three findings stand
> out. First, catalogue blind spots: of the 15 most common learner-side skills, 13 — things like
> HTML, CSS, jQuery, Git, JIRA — are taught by *zero* courses in the catalogue. That's a direct,
> actionable list for what to add next. Second, goal teachability: across 355 realistic target
> sets built the same way Protocol B builds them, the mean ceiling is 93.5% — most goals are fully
> teachable by what we already have. Third, we independently re-ran Protocol A and B through
> `engine.py` on a fresh sample and reproduced almost identical recall — confirming that
> extracting the logic out of the notebook changed nothing about the scoring.

---

## Slide 18 — Technology Stack

**On screen:** table — Python, pandas, sentence-transformers, scikit-learn, Gemini, FastAPI, Streamlit

**Script:**
> On the technology side: Python 3.13, pandas and numpy for data processing, sentence-transformers
> with the all-MiniLM-L6-v2 model for semantic embeddings, scikit-learn's TF-IDF vectorizer for
> the classical NLP signal, a custom greedy MMR implementation for diversification, Google Gemini
> via the google-genai SDK for the agent, FastAPI and Pydantic for the API, and Streamlit for the
> dashboard. Worth calling out: everything runs on CPU — no GPU required anywhere in this system.

---

## Slide 19 — Engineering Challenges (1/2)

**On screen:** table — 5 challenges and resolutions

**Script:**
> A few of the harder engineering problems we solved. The 3x duplicated learner population that
> was invisible to a naive de-dupe — solved with content-fingerprint deduplication and an
> automated diagnostic. The resume-prose skill problem — solved with a length-and-grammar filter
> validated against a positive control. Only 237 of 132,943 skills matching the catalogue exactly
> — solved with the three-stage semantic bridge. MinMaxScaler making scores incomparable across
> learners — solved by redesigning all five signals to be independently 0 to 1. And MMR using
> embedding similarity alone actually *reducing* provider diversity — solved by redefining
> redundancy as the max of content similarity and a same-provider penalty.

---

## Slide 20 — Engineering Challenges (2/2)

**On screen:** table — 4 more challenges and resolutions

**Script:**
> A few more. Our first evaluation attempt was near-tautological — a 0.979 headline recall that
> actually meant nothing — so we added Protocol B as the honest quality measure. We had a subtle
> bug where a `difficulty` signal column silently overwrote a `difficulty` display column — fixed
> by renaming the signal output to `difficulty_fit`. The Gemini SDK's automatic function-calling
> breaking on two sequential tool calls — fixed with a manual tool-call loop. And notebooks 01 and
> 02 being unrunnable out of the box because they referenced missing data folders — fixed by
> rebuilding the data layout and adding a fallback loader.

---

## Slide 21 — Known Limits

**On screen:** 5 limits — catalogue size, quality signal, evaluation scope, signal weights, persistence

**Script:**
> We want to be upfront about the limits of this system. The catalogue is small — 404 courses
> covering 323 skills — so many realistic learner goals simply aren't teachable by it yet; that's
> exactly why we expose `gap.ceiling` per request instead of hiding it. Course quality is a weak
> signal by construction, since ratings are heavily compressed and review counts are missing for
> most courses. Our evaluation measures skill retrieval, not learning outcomes — it can't capture
> teaching quality or prerequisites. The five signal weights are reasoned defaults, not learned
> from data. And there's no persistence yet for embeddings or chat sessions — everything
> re-encodes or resets on every run.

---

## Slide 22 — Roadmap / Future Work

**On screen:** 8 numbered future-work items

**Script:**
> Looking ahead, eight concrete next steps: tune the bridge threshold against hand-labeled skill
> pairs instead of an eyeballed 0.60; learn the hybrid weights with a grid search against Protocol
> B recall instead of reasoning them by hand; start using course duration and type, which we
> collect but don't use yet; handle cold start explicitly for the 381 learners with zero valid
> skills; and — probably the single highest-leverage change — expand the catalogue itself, since
> 404 courses over 323 skills is the binding constraint on quality. We'd also like to persist
> embeddings and chat sessions so they survive a restart, formally evaluate the agent's free-text
> skill extraction, and add automated regression tests for the three assertions that already
> caught real bugs during development.

---

## Slide 23 — Summary: Key Numbers

**On screen:** 8 stat cards — 18,194 learners · 0.750 recall · 7.65× lift · 93.5% teachability

**Script:**
> To close, the numbers that matter most. We took 54,933 duplicated rows down to 18,194 real
> learner profiles, and 201,934 noisy skill strings down to 132,943 real skills. The skill bridge
> took matched skills per profile from 1.47 up to 7.23. And the headline result: 0.750 recall at
> 10 on Protocol B — real recommendation quality on skills the engine never saw — a 7.65x lift
> over the popularity baseline, with 93.5% mean goal teachability across realistic targets. All
> five planned deliverables — engine, profiling, agent, dashboard, and analytics report — shipped.

---

## Slide 24 — Thank You / Q&A

**On screen:** Thank You · Questions & Discussion

**Script:**
> Thank you. That's the AI-powered course recommendation engine — from raw, messy HR data all the
> way to an explainable, evaluated, and served recommendation system. We're happy to take any
> questions, whether that's on the data repairs, the ranking math, the evaluation protocols, or
> where we're taking this next.

---

*Total estimated runtime at a natural speaking pace: ~14–18 minutes for all 24 slides, before Q&A.*
