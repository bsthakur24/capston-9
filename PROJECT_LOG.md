# Project Log — Skill-Gap Course Recommendation Engine

The working record for this project: every change made, **why** it was made, and what is still
outstanding. `README.md` describes what the system *is*; this file tracks how it got there and
where it is going.

**Convention:** every entry states the *reason* for the change, not just the change. Items move
from Part B to Part A only when verified — a task is not "done" because code was written, it is
done when it ran and the result was checked.

**Status:** 13 of 13 planned tasks complete, plus the AI Agent, API, dashboard and analytics
report (Phase 6). 11 open items in Part B (B1.1 and B4.3 are now resolved; B2 shrank from 4 items
to 2).
**Last updated:** 2026-09-16

---

# Part A — Completed

## Phase 0 · Assessment

### A0.1 — Audit the existing three notebooks
**Reason:** Nothing could be fixed responsibly before understanding what the pipeline did and
which of its claims held up.

**Found:** a data-cleaning notebook, a profile-building notebook, and an engine combining
skill-gap coverage, TF-IDF and sentence embeddings into a hybrid score. Working, but with an
empty Step 6, twelve hard-coded skill aliases, and an engine that never touched `user_profiles`.

---

## Phase 1 · Environment

### A1.1 — Build a project virtualenv
**Reason:** The notebooks had been executed on two different kernels (`my_venv 3.12`,
`.venv 3.10`), neither present on this machine, and the active Python 3.13 had no `torch` or
`sentence-transformers`. The semantic signal — a third of the engine — could not run at all.

**Done:** `.venv` on Python 3.13 with CPU-only torch. Verified `torch 2.14.0+cpu`,
`sentence-transformers 6.0.1`, and a successful `all-MiniLM-L6-v2` encode.

### A1.2 — Pin `requirements.txt`
**Reason:** The original file listed nine bare package names with no versions and a missing
newline (`ipykernel` was glued to the previous line). Unpinned dependencies plus a notebook
pipeline means results are not reproducible.

**Done:** All versions pinned to what is actually installed, with the CPU-torch index URL
documented as an install note.

---

## Phase 2 · Data quality

### A2.1 — Detect and remove the 3× stacked learner population
**Reason:** Every `user_frequency` in `skill_vocabulary.csv` was divisible by 3, which is not a
coincidence anyone should accept. Investigating showed each person appears **three times** under
three different `person_id`s, offset by exactly 18,311, with all other columns byte-identical.
Notebook 01's `drop_duplicates("person_id")` could not see it because the IDs differ.

**Why it mattered:** it inflated every frequency statistic threefold and would have put the same
person on both sides of any train/test split, making offline evaluation meaningless.

**Done:** content fingerprint (`name` + `profile_text`), keep lowest `person_id`.
**Result:** 54,933 → **18,194** learners (3.02×). Minimum skill frequency dropped from 3 to 1,
confirming the inflation is gone.

### A2.2 — Add a duplicate-person diagnostic to notebook 01
**Reason:** A2.1 fixes the symptom in notebook 02, but the defect originates upstream. If new raw
data arrives with the same problem, it should be caught at the cleaning stage rather than
rediscovered.

**Done:** `person_fingerprint()` reports a repeat factor and warns above 1.05×.
**Verified:** on synthetic raw data containing a planted 3× stack, it reported exactly `3.00x`.

### A2.3 — Filter resume prose out of the skill vocabulary
**Reason:** ~⅓ of "skills" were whole resume bullets — e.g. *"maintain multiple database
environments redshift rds in aws"*. These can never match a course and bloat the vocabulary.

**Why this filter:** the course catalogue is a ground truth for the shape of a real skill — all
323 catalogue skills are 1–5 words, averaging 2.0. Length bounds plus rules for bullet-leading
verbs and prose connectives follow from that evidence rather than from taste.

**Done:** `is_valid_skill()` in notebook 02, gated by a **positive control** that must keep all
323 known-good catalogue skills.
**Result:** 201,934 → **132,943** skills, positive control **100%**. An earlier draft scored 99.7%
by wrongly rejecting `writing` and `managed services`; restricting the verb rule to phrases of 3+
words fixed it.

---

## Phase 3 · Recommendation engine

### A3.1 — Replace hard-coded aliases with a learner→catalogue skill bridge
**Reason:** the twelve hand-written aliases could not generalise across a 132,943-skill learner
vocabulary. Measurement showed the real cost: only 237 skills match the catalogue exactly, so the
average profile matched **1.47** catalogue skills and 32.7% matched *nothing*. The skill-gap
signal was empty for most learners — the engine's primary signal was barely functioning.

**Done:** three-stage bridge — normalise → exact match → nearest catalogue neighbour by embedding
cosine, accepted only above a calibrated 0.60 threshold. Unresolvable skills are reported as
`unmapped` rather than forced onto a wrong match.
**Result:** **1.47 → 7.23** catalogue skills per profile; zero-match profiles **32.7% → 6.7%**.

### A3.2 — Implement Step 6, weighted skill-gap scoring
**Reason:** Step 6 was an empty cell with a markdown note deferring it to the AI Agent. Without
it, every missing skill counted equally, so a course teaching one incidental skill ranked level
with one teaching the learner's central goal.

**Done:** `SkillGap.weights`, exposed through `recommend(skill_weights=...)`, defaulting to 1.0.
Weights are bridged through the same canonicalisation as the skills themselves, so the agent can
pass its own surface forms.

### A3.3 — Fix the coverage denominator
**Reason:** the original divided by the number of gap skills *present in the catalogue*. A learner
with a five-skill gap where only two are teachable would see a course covering both scored as
100% coverage — flattering and wrong.

**Done:** all gap skills stay in the denominator; `gap.ceiling` separately reports the best
coverage any course could achieve, so a 0.4 against a ceiling of 0.4 is legible as a perfect
result.

### A3.4 — Remove `MinMaxScaler`
**Reason:** it was fit on the *filtered candidate pool*, which forced that pool's best course to
exactly 1.0 regardless of quality. Scores became incomparable between learners, and coverage —
already a fraction with a fixed meaning — was rescaled against whatever else happened to be in the
pool, discarding that meaning.

**Done:** deleted. Every signal is now constructed on a 0–1 scale independent of the candidate
set, so they combine directly.

### A3.5 — Add the two prepared-but-unused metadata signals
**Reason:** Step 2 of the original notebook promised a "metadata representation" of difficulty and
ratings; nothing downstream ever read those columns.

**Done:** Bayesian-smoothed quality and difficulty fit. Smoothing was necessary because
`review_count` is missing for 75% of courses — a 4.9 from three reviewers would otherwise outrank
one from thousands. Both carry low weight (0.12 / 0.08) because ratings are compressed
(σ = 0.17) and the catalogue is 73% Beginner; they break ties rather than drive ranking.
**Also:** dropped `course_students_enrolled`, which is empty for all 404 courses.

### A3.6 — MMR re-ranking with provider-aware redundancy
**Reason:** two providers hold a quarter of the catalogue (Google 61, IBM 53) and specialisations
are often near-duplicates, so pure score ordering returns the same course repeatedly.

**Correction during the work:** the first MMR implementation used embedding similarity alone and
*reduced* provider diversity (7 → 6). Two certificates from one provider can be near-duplicates in
intent while sitting far enough apart in embedding space for both to survive. Redundancy is now
the larger of content similarity and a flat same-provider penalty.
**Result:** pairwise similarity 0.541 → 0.523, distinct providers 7 → **8**, costing 0.006 of mean
hybrid score.

### A3.7 — Connect the engine to real learner profiles
**Reason:** the engine ran entirely off two hard-coded lists and never read `user_profiles.csv`,
so the 18,194 profiles were decorative.

**Done:** `recommend_for_person(person_id, target_skills)` derives current skills from the stored
profile through the bridge. The explicit-skills `recommend()` remains the agent-facing entry point.

### A3.8 — Fix the `difficulty` column collision
**Reason:** the explain block wrote a `difficulty` signal column over the `difficulty` display
column, so the output table showed `1.0` / `0.5` where the course's level belonged.
**Done:** the signal is surfaced as `difficulty_fit`. Verified the table now shows
`Beginner` / `Intermediate`.

---

## Phase 4 · Evaluation

### A4.1 — Replace a near-tautological evaluation with two honest protocols
**Reason:** the first evaluation hid a third of a learner's skills and then handed **those same
skills** to the engine as its target. The engine optimises coverage of exactly what it is asked
for, so recall@10 = 0.979 was an arithmetic consequence of the setup, not a finding. Publishing it
as a headline result would have been misleading.

**Done:** two explicitly labelled protocols.
* **Protocol A (retrieval check)** — the original design, relabelled as a correctness test. Its
  value is that the baselines stay low, showing a 404-course catalogue does not make the task
  trivial.
* **Protocol B (peer-target prediction)** — the engine never sees the held-out skills; the target
  is built from the most common skills of *other people with the same job title*. This asks a
  question with a real answer.

**Result (150 learners each):**

| recall@10 | Protocol A | Protocol B |
|---|---|---|
| engine | 0.979 | **0.750** |
| random | 0.374 | 0.378 |
| popularity | 0.110 | 0.098 |
| lift vs popularity | 8.90× | **7.65×** |

Only Protocol B's margin should be read as recommendation quality. A2.1 is load-bearing here:
without de-duplication a learner's own triplicate copies would sit in their own peer group.

---

## Phase 5 · Reproducibility and documentation

### A5.1 — Make notebooks 01 and 02 runnable again
**Reason:** both read from `data/raw/` and `data/cleaned/`, neither of which existed — only the
final outputs had been carried over as loose files in the project root. The pipeline could not be
re-run at all.

**Done:** `data/` layout created; loose CSVs moved to `data/recommendation_ready/`; notebook 02
gained a fallback that reconstructs the long tables from the published profiles when
`data/cleaned/` is absent. Notebook 02 now executes end-to-end.

### A5.2 — Smoke-test notebook 01 against synthetic raw data
**Reason:** `data/raw/` is empty, so notebook 01's code path was unverifiable. Shipping unverified
edits is how regressions reach a submission.

**Value confirmed:** the test caught two real regressions introduced by my own rewrite — replacing
the original cell 0 dropped the `re` and `unicodedata` imports, and the notebook crashed on the
first `clean_text()` call. Both fixed; the notebook then ran end-to-end in an isolated sandbox.

### A5.3 — Clear notebook 01's stale outputs
**Reason:** it carried outputs from the original author's run interleaved with my new cells, which
have none. A half-populated notebook misrepresents what was actually executed.
**Done:** outputs cleared, with the reason documented in `README.md`. Notebooks 02 and 03 ship
executed.

### A5.4 — Write `README.md`
**Reason:** no documentation existed. A capstone that cannot be explained to a reader is
incomplete regardless of how the code performs.
**Done:** pipeline diagram, setup, both data repairs, ranking design, evaluation results, limits.

---

## Phase 6 · Serving layer, AI Agent, dashboard, analytics report

### A6.1 — Extract the engine from notebook 03 into `engine.py`
**Reason:** the AI Agent and an API both need to call the scoring path as a function; a notebook
cannot be imported or called per-request. Splitting it out also means notebook 03, `agent.py`,
`api.py` and `dashboard.py` all share exactly one implementation instead of four copies drifting
apart.

**Done:** `engine.py` mirrors notebook 03's logic (bridge, five signals, MMR, `recommend()` /
`recommend_for_person()`) plus one addition, `get_peer_skills_for_role()`, which grounds a stated
career goal in real peer data (the same idea Protocol B evaluates) instead of letting an LLM
invent a target skill list. Verified by smoke test: `recommend()` and the peer lookup both return
sensible output on import.

### A6.2 — Build the AI Training Agent (`agent.py`) on Gemini
**Reason:** `target.md` deliverable 3. Closes B2.1-B2.4 below: the agent extracts
`current_skills` from free text, grounds `target_skills` via `lookup_role_skills`, infers
`skill_weights` only when the learner signals emphasis, and calls `recommend_courses` - all as
Gemini tool calls into `engine.py`, so the model never re-derives a ranking itself.

**Bug found and worked around:** `google-genai==2.23.0`'s automatic function-calling loop closes
its internal HTTP client after the first tool round-trip, so any turn needing two sequential tool
calls (the agent's main path: look up peer skills, then recommend) failed with `"Cannot send a
request, as the client has been closed."` Bisected by testing one tool alone, two tools with one
call, and two tools with a forced chain - only the chained case failed. Fixed by driving tool
calls manually (`TrainingAgent.send`) instead of relying on the SDK's AFC, which sidesteps the bug
entirely.

**Also found:** the model named in earlier guidance, `gemini-2.5-flash`, is retired; the API's own
error message pointed to `gemini-3.6-flash`, now the default.

**Verified:** a live two-turn conversation ("I want to become a data scientist" -> peer lookup ->
10-course shortlist; then "I mostly care about python" -> re-ranked with `skill_weights`) produced
correct, traceable tool calls and results.

### A6.3 — FastAPI service (`api.py`)
**Reason:** `/recommend` and `/recommend/{person_id}` expose the engine directly (no LLM, no API
quota cost); `/chat` exposes the agent over HTTP with in-memory session state.

**Done and verified live:** `POST /recommend` returned correct scored results over real HTTP.
`POST /chat` surfaced a real finding - see A6.2's bug - and now converts a Gemini `ClientError`
(e.g. 429 quota exhaustion, which we hit live during testing: the free tier is 20
requests/day/model) into a proper HTTP status instead of a bare 500.

### A6.4 — Streamlit dashboard (`dashboard.py`)
**Reason:** `target.md` deliverable 4. Three tabs: chat with the agent (with a sidebar field to
paste a session-only Gemini key, never written to disk), a direct-recommend form that bypasses
Gemini entirely, and a catalogue/learner/skill-vocabulary explorer.

**Done and verified:** `streamlit.testing.v1.AppTest` executed the full script headlessly with
zero exceptions across all 6 tabs (a plain HTTP GET does not run a Streamlit script - only a real
session does, so this was the correct way to verify it). Caught and fixed a real deprecation
(`use_container_width` was already past Streamlit's removal date).

### A6.5 — Analytics report (`04_analytics_report.ipynb`)
**Reason:** `target.md` deliverable 5. Quantifies what the engine achieves against the real data
rather than restating design intent - imports `engine.py` directly so every figure traces to the
same code path the agent and API call.

**Findings that were not already documented:**
* **Catalogue blind spots:** of the 15 most common learner-side skills, 13 are taught by *zero*
  courses (`html`, `css`, `jquery`, `java`, `ajax`, `xml`, `mysql`, `html5`, `git`, `bootstrap`,
  `oracle`, `json`, `css3`, `jsp`, `jira`). Directly actionable for what to add to the catalogue.
* **Goal teachability:** sampling 355 (role, learner) target-skill sets built the same way
  Protocol B builds them, mean `gap.ceiling` is 93.5% and 93.5% are fully teachable. This is
  better than the "404 courses, 323 skills" framing alone suggests, precisely *because*
  peer-derived targets concentrate on skills the catalogue already covers - it is not evidence
  against B3.5 (catalogue expansion still matters most for less-common goals).
* Re-ran Protocol A/B evaluation independently through `engine.py` (150 learners, fresh sample):
  0.979 / 0.749 recall@10, 8.90x / 7.64x lift - matches the original notebook 03 run to within
  sampling noise, cross-verifying the extraction in A6.1 changed nothing about the scoring.

**Caveat carried into the report itself:** `data/cleaned/` was still empty at execution time, so
these figures come from `data/recommendation_ready/`'s fallback-snapshot rebuild (see B1.2), not a
fresh pass over the raw CSVs now sitting in `data/raw/`. Re-running 01 -> 02 -> 03 -> 04 in order
will regenerate every number in the report from real raw data.

---

# Part B — Outstanding

Ordered by priority. Sizes are rough: **S** ≈ under an hour, **M** ≈ a few hours, **L** ≈ a day+.

## B1 — Blocking / correctness

### B1.1 · Obtain the raw source files · **S** · DONE, blocks B1.2
`data/raw/` now has all five files: `01_people.csv`, `03_education.csv`, `04_experience.csv`,
`05_person_skills.csv`, and `coursera_data.csv` (the last one rebuilt from a differently-shaped
source - `Metadata` split into `Difficulty`/`Type`/`Duration`, `Review Count` extracted from
`"4.8(20K reviews)"` - `course_url` deliberately omitted rather than faked, since an all-empty
column would make notebook 01's dedup step treat every course as a duplicate of the first). Two
unrelated files that arrived alongside these (an HR employee-master table and a training-log
table, neither read by any notebook) were removed rather than left to cause confusion.
**Why it matters:** notebook 02's fallback loader is lossy — it cannot recover institution, firm
or location, and derives record counts from de-duplicated lists rather than raw rows. B1.2 is
still open: nothing has actually run notebook 01 against these files yet.

### B1.2 · Run the full pipeline 01 → 02 → 03 on real raw data · **S** · blocked by B1.1
Confirm the repeat-factor diagnostic fires at 3.00× on the genuine input and that row counts match
what the fallback path produced.

### B1.3 · Surface unmapped skills to the caller · **S**
`bridge_skills()` classifies a skill as `unmapped` when nothing clears the 0.60 threshold, but
`SkillGap` discards that — verified: `unmapped` appears nowhere in the `SkillGap` class. A learner
asking for a skill the system did not understand gets silence.
**Proposal:** track `gap.unmapped_inputs` and report it alongside `gap.unteachable`, which is
already exposed. The two are different failures: *we did not understand you* vs *no course teaches
this*.

---

## B2 — The AI Agent — DONE (see A6.2); two follow-ups remain

B2.1-B2.4 are resolved by `agent.py`: the contract is the three Gemini tool-call arguments
(`current_skills`, `target_skills`, `skill_weights`) plus `lookup_role_skills` grounding a stated
goal in Protocol B's peer-skill baseline instead of an LLM guess. What's left is refinement, not
the core capability:

### B2.5 · Evaluate agent-extracted skills against stored-profile ground truth · **M**
`agent.py`'s free-text extraction (B2.2) has only been checked by reading its output and judging
it sensible - never compared against a learner's actual stored `skills` list the way the engine
itself is evaluated (Protocol A/B). Without that, there's no measured error rate for "the agent
misread what I said."

### B2.6 · Persist chat sessions · **S**
`api.py`'s `_CHAT_SESSIONS` and `dashboard.py`'s equivalent are in-memory only - a restart loses
every in-flight conversation. Fine for a demo, not for anything beyond it.

---

## B3 — Model quality

### B3.1 · Tune the bridge threshold against labelled pairs · **M**
0.60 was chosen by inspecting a calibration table — defensible but not measured. `ms sql server
2008 r2 → sql` scores 0.491 and is currently rejected, which is arguably a miss.
**Proposal:** hand-label a few hundred pairs, then pick the threshold on precision/recall instead
of by eye.

### B3.2 · Fit the hybrid weights instead of asserting them · **L**
0.40 / 0.25 / 0.15 / 0.12 / 0.08 are reasoned defaults. Protocol B gives an objective to optimise
against, so at minimum run a grid search; with real interaction data, learn them.

### B3.3 · Use `duration` and `type` · **S**
Verified unused — zero occurrences in notebook 03. `duration` (3 buckets) and `type`
(Course / Specialization / Professional Certificate) are clean and complete for all 404 courses. A
learner wanting a quick upskill and one wanting a certificate should not get identical lists.

### B3.4 · Handle cold start · **S**
`recommend()` is untested for a learner with no skills on file. 381 profiles had zero valid skills
after filtering. Coverage and TF-IDF fall to zero and ranking degrades to quality-plus-semantics —
intended behaviour, but unverified and undocumented.

### B3.5 · Expand the catalogue · **L**
404 courses over 323 skills is the binding constraint. Many realistic goals are simply not
teachable, which is why `gap.ceiling` exists. More courses would likely beat any modelling change
in this list.

---

## B4 — Engineering

### B4.1 · Persist embeddings and the bridge cache · **S**
Verified: no `np.save`, `pickle` or `to_parquet` anywhere. Every run of notebook 03 re-encodes 323
skill embeddings, 404 course embeddings, and re-bridges thousands of learner skills — the peer
evaluation is the slowest part of the notebook almost entirely because of this.

### B4.2 · Add regression tests · **M**
Notebooks-only means nothing is automatically tested. The highest-value assertions are the ones
that already caught real bugs: the skill-filter positive control (must stay 100%), the duplicate
check (must stay ~1.00×), and coverage never exceeding `ceiling`.
**Note:** this conflicts with the notebooks-only structure you chose — it needs either a small
test file or in-notebook `assert` cells. Flagging rather than deciding.

### B4.3 · `target.md` — RESOLVED
No longer empty: it holds the business scenario and the five deliverables (Recommendation Engine,
User Profiling Module, AI Training Agent, Dashboard, Analytics Report) that Phase 3, notebook 02,
A6.2, A6.4 and A6.5 were each built against. All five now exist in at least a first version.

---

## B5 — Known limits (accepted, not tasks)

Recorded so they are not rediscovered as bugs:

* **Ratings are weak by construction** — mean 4.68, σ 0.17, and `review_count` missing for 75% of
  courses. Bayesian smoothing manages this; it cannot manufacture signal that is not there.
* **Protocol B measures skill retrieval, not learning outcomes.** It cannot capture teaching
  quality, prerequisites or ordering, and it assumes a learner's peers are a reasonable proxy for
  what they need next.
* **`rank` follows MMR order**, so `hybrid_score` is deliberately not monotonic down the shortlist.
* **The bridge can over-generalise.** `t sql → sql` (0.792) loses specificity. This is the
  intended trade against a 237-skill exact overlap, but it is a real loss.
