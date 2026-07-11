# PROJECT_LOG.md — Fraud Detection MLOps Build

## Goal
Transform the cloned MLOps_Project baseline into a production-grade Credit Card
Fraud Detection engine. Priority: deep, defensible understanding over breadth.

## Ground Rules & Phase Roadmap
(Read this section first in any new session — it stands in for the original
briefing, so it should not need to be re-pasted or re-explained.)

**Operating rules, apply to every phase:**
1. Do not execute a phase until the user gives its exact trigger phrase — never assume or self-trigger.
2. Session start: read this file's Current Status + Known Gaps before doing anything else. Session end: update Current Status, Architecture Notes, Decisions Log, and the handoff note.
3. End of every phase: quiz the user — ask them to explain the phase's key design decisions/tradeoffs back in their own words (not a recap of my explanation). Log anything they can't articulate cleanly under Known Gaps. Don't block the next phase on this — the user controls the trigger — just log it honestly.
4. Log every nontrivial engineering decision into the Decisions Log as it's made, not batched at phase-end.
5. For fast-moving-API libraries (SHAP, EvidentlyAI, MLflow), flag explicitly if uncertain of current API shape rather than assuming — ask the user to verify against docs.
6. Prefer small, reviewable diffs over full-file rewrites — one function/module at a time for substantial changes.

**Phase roadmap:**
- **Phase 1 — Codebase Interpretation & Diagnostics.** ✅ Done — see Architecture Notes below.
- **Phase 2 — Domain Migration & Core Feature Engineering.** Trigger: `"Proceed to Phase 2"`. Replace the loan pipeline with the anonymized ULB/Kaggle Credit Card Fraud dataset (`creditcard.csv`: `Time`, `Amount`, PCA features `V1`-`V28`, target `Class`, ~0.17% positive class). Rebuild `prediction_model/processing/data_handling.py` + `preprocessing.py` (schema validation, robust feature scaling, zero leakage) and `prediction_model/training_pipeline.py` (cost-sensitive XGBoost tuned for the imbalance, optimize PR-AUC not ROC-AUC/F1). Also reconcile `prediction_model/pipeline.py` vs `training_pipeline.py`'s duplicated preprocessing (Phase 1 flagged this as a drift risk — don't reintroduce it). Update `prediction_model/config/config.py` (FEATURES/TARGET/paths) accordingly. Deliverable: rebuilt files + Decisions Log entries justifying imbalance-handling method and validation split strategy.
- **Phase 3a — SHAP-Based Local Explainability.** Trigger: `"Proceed to Phase 3a"`. Add local SHAP explainability to the FastAPI `/predict` route so flagged transactions return which features drove the decision, not just the label. Key open decisions to resolve during planning: where SHAP computation happens (inline vs. cached) and its latency-under-load cost, which SHAP explainer fits XGBoost specifically, whether explanations are computed for every prediction or only flagged ones, and response shape. Deliverable: hardened `/predict` endpoint + Decisions Log entries, especially on the latency tradeoff. Scoped as a one-day session.
- **Phase 3b — Drift Monitoring & Observability.** Trigger: `"Proceed to Phase 3b"`. Add EvidentlyAI-based drift monitoring on inference inputs, connected to Prometheus/Grafana. Key open decisions: what baseline drift is measured against (expected: training distribution), whether this is scheduled/automated or manual, whether drift→alert is in scope this session (drift→retrain is explicitly deferred, log as a known future step same as the per-request MLflow redesign), and how/whether this connects to 3a's SHAP output. Deliverable: hardened `/metrics` + drift pipeline, Decisions Log entries on baseline definition and automation scope. Scoped as a one-day session, run the day after 3a.
- **Phase 4 — Hardening & System Architecture Grill.** Trigger: `"Start Interview Drill"`. Become a cynical, aggressive Principal Interviewer — one question at a time on SHAP latency, metric cascading failures, drift-automation loops, architecture bottlenecks; critique harshly, then show the senior-engineer answer. Re-run periodically as a retention check, not just once at the end.

## Current Status
- Phase: 2 code complete AND fully verified end-to-end (training + pytest + live serve
  request all confirmed working). Only the Phase 2 checkpoint quiz remains before Phase 3
  can be triggered.
- Last updated: 2026-07-10
- Runtime blockers — all resolved this session:
  1. ~~`creditcard.csv` must be placed locally~~ — done (Kaggle/ULB `mlg-ulb/creditcardfraud`).
  2. ~~`env/` venv incompatible (Python 3.14, empty)~~ — recreated as a **Python 3.10.11** venv
     at the same path (`env/`), pinned `requirements.txt` installed cleanly into it.
  3. ~~Baseline MLflow EC2 server unreachable~~ — confirmed unreachable (connection timed out);
     `TRACKING_URI` correctly falls back to local `file:./mlruns`.
  4. **New, now fixed**: `pytest -v tests/` initially failed with
     `ModuleNotFoundError: No module named 'prediction_model'` — `tests/` has no `__init__.py`,
     so pytest's default "prepend" import mode used `tests/` (not the repo root) as the
     sys.path base. Fixed with a new `pytest.ini` (`[pytest]\npythonpath = .`), pytest 7.4.2's
     built-in fix for exactly this. No code changes needed; `python -m prediction_model.training_pipeline`
     (module form, not `python path/to/script.py`) was the equivalent fix for running the
     training script directly.
- **Training verified** (`python -m prediction_model.training_pipeline`, 20 Hyperopt trials,
  ~6 min): `scale_pos_weight = 577.3` (394 frauds / 227,451 legit in train — matches the
  Decisions Log's ~578 estimate). Best holdout metrics: **`pr_auc = 0.8693`**,
  `roc_auc = 0.9809`, `precision = 0.8632`, `recall = 0.8367`, `f1 = 0.8497`. Strong result
  given the ~0.17% base rate — validates the `scale_pos_weight` + PR-AUC-tuned-CV approach.
- **Tests**: `pytest -v tests/` — confirmed **6 passed** after the `pytest.ini` fix
  (schema accept/reject/type gates, feature contract, split-ratio + determinism checks).
- **Pre-existing baseline gap found while serve-testing**: `main.py` imports `boto3` (used by
  `upload_to_s3`) but it was never listed in `requirements.txt` — only worked in Docker before
  because `dvc[s3]` pulled it in transitively. Not a Phase 2 regression; added `boto3` to
  `requirements.txt` and it now installs/runs standalone.
- **Serve smoke test confirmed**: `python main.py` + `POST /prediction_api` with the dataset's
  first row (a known-legit transaction) → `200 OK`, `{"status": "Legitimate"}` — correct.
  End-to-end (train → test → serve) is now fully verified locally.
- Still to do: the Phase 2 checkpoint quiz (malformed-body/422 check is a nice-to-have, not
  blocking).
- Session handoff note: read Ground Rules & Phase Roadmap + Architecture Notes (incl. the
  Phase 2 addendum) + Known Gaps. Environment is now fully working end-to-end locally — next
  session can go straight to the serve smoke test and checkpoint quiz, then Phase 3 (SHAP +
  drift) once triggered. Docker/CI + per-request MLflow redesign remain deferred as scoped.

## Architecture Notes (fill in during Phase 1, keep updated)
- **Pipeline orchestration**: No orchestrator (no Airflow/Kubeflow/ZenML).
  `prediction_model/training_pipeline.py` is a flat script: load `train.csv`
  → Hyperopt search (5 evals) over XGBoost params → each trial rebuilds an
  inline sklearn `Pipeline` (DomainProcessing → MeanImputer → ModeImputer →
  DropColumns → CustomLabelEncoder → LogTransforms → MinMaxScaler →
  XGBoostClassifier) → autologs to MLflow. `prediction_model/pipeline.py`
  defines a second, hand-duplicated copy of the same preprocessing steps
  (minus the classifier) — not imported by training_pipeline.py, so the two
  can drift out of sync silently.
- **Config management**: Single flat module `prediction_model/config/config.py`
  — no env-based overrides, no dev/stage/prod separation. Hardcodes: MLflow
  `TRACKING_URI` to a raw EC2 hostname, S3 `S3_BUCKET`/`FOLDER` names,
  feature lists (`FEATURES`, `NUM_FEATURES`, `CAT_FEATURES`, etc.), and
  `MODEL_NAME`. Every module (`main.py`, `predict.py`, `training_pipeline.py`,
  `pipeline.py`) imports this module directly — it functions as global state.
- **Artifact registry / versioning**: DVC for data (`prediction_model/datasets.dvc`,
  S3 remote `loanprediction`, pulled via `dvc pull --force` in the Docker
  build). MLflow for models/experiments — but no MLflow Model Registry stages
  (Staging/Production) are used; `predict.py` instead re-queries
  `mlflow.search_runs(...).order_by(['metrics.f1_score DESC'])` and loads
  `runs:/<best_run_id>/Loanprediction-model` **on every single prediction
  call** (single, UI, and batch endpoints) — no caching, no pinned model
  version, no promotion gate.
- **Logging**: None. No `logging` module anywhere in the repo — the only
  diagnostic output is a stray `print(df)` in `main.py`'s `batch_predict`.
  No request/response audit trail, no structured logs for a
  fraud-relevant compliance trail.
- **Monitoring layer**: `prometheus_fastapi_instrumentator.Instrumentator()`
  exposes `/metrics` on the FastAPI app (HTTP/infra metrics only — request
  count/latency, nothing about model output distribution or class balance)
  → scraped by Prometheus → visualized in Grafana per the README screenshots.
  Separately, `drift_monitoring/app_v1.py` is a standalone Streamlit app
  (not scheduled, not containerized alongside the API) that pulls the
  batch-endpoint's S3-logged inputs (`datadrift/<date>/...`) plus a
  `baseline.csv`, and renders EvidentlyAI `DataDriftPreset`/
  `DataQualityPreset` HTML reports on manual demand — no automated
  drift→alert or drift→retrain loop exists yet.

### Data flow (end to end)
1. **Data versioning** — raw data DVC-tracked, S3 remote, pulled in Docker build.
2. **Training** — `training_pipeline.py` trains XGBoost via Hyperopt (5 evals),
   logs every trial + the fitted pipeline artifact to MLflow experiment
   `loan_prediction_model`.
3. **CI/CD** (`.github/workflows/main.yml`) — on push to `main`: Docker build
   (which itself runs `dvc pull`, re-executes `training_pipeline.py`, and
   runs pytest inside the image) → push to ECR → `kubectl apply` to EKS.
4. **Serving** (`main.py`) — `/prediction_api`, `/prediction_ui`,
   `/batch_prediction` → `predict.py` → live MLflow lookup + model reload per
   request → prediction. Batch endpoint also uploads raw input to S3 for
   later drift review.
5. **Monitoring** — `/metrics` → Prometheus → Grafana (infra only); drift/
   quality checked manually via the Streamlit app against Evidently.

### Top 3 structural vulnerabilities (flagged for Phase 2/3 hardening)
1. **Per-request MLflow model resolution, no registry stages** — `predict.py`
   hits the MLflow tracking server + artifact store (S3) on every request to
   find and reload the "best" model; couples serving latency/availability to
   MLflow uptime, and makes production model selection an unpinned, ungated
   hot-swap (any new run with a marginally higher F1 goes live instantly).
2. **Silent failure modes instead of loud validation** — `CustomLabelEncoder`
   maps unseen categories to `NaN` via `.map()` with no explicit handling;
   `LogTransforms` runs `np.log` on raw values with no zero/negative guard
   (`-inf`/`NaN`); Pydantic fields are untyped `str` with no enum
   constraints. A malformed input silently shifts the prediction instead of
   raising — dangerous for a fraud decision path.
3. **Config/secrets/train-serve skew** — hardcoded MLflow URI/S3 names with
   no env separation; AWS credentials baked into Docker build ARGs/ENV
   (visible in image layers); `pipeline.py` and `training_pipeline.py`
   maintain two independently hand-written copies of the same preprocessing
   steps.

Secondary (not in top 3, but worth remembering): `batch_predict` checks that
required columns are *present* but never slices the DataFrame to
`config.FEATURES`, so extra input columns ride into `model.predict` and can
misalign the training-time column count; the only test
(`tests/test_prediction.py`) hard-asserts one fixed row predicts `'Y'` — it
breaks the moment the "best run" model changes, so it's not a real
regression gate.

### Phase 2 addendum — fraud architecture (2026-07-08)
- **Domain**: `creditcard.csv` — `Time`, 28 PCA components `V1`–`V28`, `Amount`, target
  `Class` (1=fraud, ~0.17%). All-numeric, so the loan-era categorical machinery is gone.
- **Config** (`config/config.py`): `DATA_FILE='creditcard.csv'`, `TARGET='Class'`, 30-item
  `FEATURES`, `ROBUST_SCALE_FEATURES=['Amount','Time']`, plus `TEST_SIZE/RANDOM_STATE/CV_FOLDS/
  MAX_EVALS`, `EXPERIMENT_NAME='fraud_detection_model'`, `MODEL_NAME='/fraud-detection-model'`.
  `TRACKING_URI` now env-overridable, local `file:./mlruns` default.
- **Preprocessing** (`processing/preprocessing.py`): reduced to `MeanImputer` (defensive) +
  new `RobustScalerTransformer` (median/IQR on Amount/Time; V1–V28 already PCA-scaled). Removed
  `ModeImputer`/`DropColumns`/`DomainProcessing`/`CustomLabelEncoder`/`LogTransforms` — this
  eliminates the Phase-1 silent-`.map()`-NaN and unguarded-`np.log` vulns outright (no
  categoricals in this data). Input hardening now lives in the typed Pydantic schema.
- **Single-source pipeline** (`pipeline.py`): `build_pipeline(classifier=None)` is the one
  definition of preprocessing; `training_pipeline.py` appends the classifier to it (baseline
  duplication removed).
- **Split** (`processing/data_handling.py`): `stratified_split(df, ...)` — the single, named,
  dedicated holdout function, separate from the StratifiedKFold folds used inside Hyperopt.
- **Training** (`training_pipeline.py`): cost-sensitive XGBoost (`scale_pos_weight`, no
  `use_label_encoder` — removed in xgb≥1.6), Hyperopt/TPE (`max_evals=20`) minimizing
  `-mean(PR-AUC)` over StratifiedKFold; final refit on X_train, evaluated once on the holdout;
  logs `pr_auc` (primary) + roc_auc/precision/recall/f1 and the model artifact.
- **Serving**: `schemas.py` (new) holds the typed `TransactionFeatures` model (imported by
  `main.py`; kept separate so the test gate needn't import FastAPI/mlflow/boto3). `predict.py`
  selects the best run by `metrics.pr_auc`, outputs `fraud`/`legit`, and slices batch input to
  `FEATURES`. `main.py` endpoints use the new schema and return `Fraud`/`Legitimate`.
  (Per-request MLflow resolution redesign still deferred to Phase 3.)

## Engineering Decisions Log
(One entry per nontrivial choice. This is your interview cheat sheet — keep it precise.)

| Date | Decision | Why | Tradeoff / Alternative Considered |
|---|---|---|---|
| 2026-07-07 | Diagnosed baseline repo instead of assuming it's production-ready | Repo is a loan-approval demo, not built for a fraud SLA; interviewers will probe exactly these gaps | Could have skipped diagnostics and gone straight to Phase 2, but would've inherited the per-request MLflow lookup and silent-NaN bugs unknowingly |
| 2026-07-07 | Will replace, not patch, `predict.py`'s per-request MLflow lookup in a later phase | Live `search_runs`+`load_model` per call is a latency/availability P0 for real-time fraud scoring | Alternative: cache model in memory with TTL/webhook refresh, or pin to an MLflow Registry "Production" stage — decide in Phase 2/3 |
| 2026-07-07 | Closed Phase 1 with one logged gap (NaN articulation) rather than re-quizzing same-session | User asked to close the session; ground rules log gaps but don't gate the phase trigger on resolving them | Alternative was to re-quiz immediately — deferred to "before Phase 4 drills" per Known Gaps instead |
| 2026-07-08 | Kept XGBoost (not LightGBM) for the fraud model | Baseline choice; strong tabular learner with native `scale_pos_weight` for imbalance; not worth re-litigating this phase | LightGBM can be faster on wide data, but no evidence it beats XGB here and migration adds cost |
| 2026-07-08 | PR-AUC as primary metric (over ROC-AUC/F1) — now enforced in **serving** too | At 0.17% positives ROC-AUC is optimistic (dominated by the huge true-negative mass); PR-AUC concentrates on the fraud (positive) class. Also changed `predict.py` best-run selection `f1_score`→`pr_auc` | F1 needs a fixed threshold; PR-AUC is threshold-free. Logged now because the change reaches serving, not just training |
| 2026-07-08 | Cost-sensitive XGBoost via `scale_pos_weight` = n_neg/n_pos (~578) | Reweights the loss toward the rare class without discarding or duplicating rows | SMOTE/under-sampling rejected — distort the base rate and can leak across folds; class weighting is simpler and fold-safe |
| 2026-07-08 | Stratified holdout + StratifiedKFold CV, averaging PR-AUC across folds | Preserves the ~0.17% rate in every split; averaging avoids tuning against one noisy holdout | Single-holdout tuning rejected as too high-variance with so few positives |
| 2026-07-08 | Rejected temporal split (reasoning logged explicitly) | Temporal splitting is the correct approach for production fraud in general (avoids look-ahead leakage, honors concept drift), BUT this dataset spans only ~2 days of PCA-anonymized features with no interpretable time-of-day/seasonal signal and only 492 frauds — a temporal cut mostly injects noise into the PR-AUC signal rather than buying real drift protection | If extended to a longer, real-world transaction stream, temporal splitting becomes the correct upgrade |
| 2026-07-08 | 5 CV folds | 492 frauds → ~394 in the 80% train → ~79 frauds per validation fold: enough positives for a stable per-fold PR-AUC | 10 folds → ~39 frauds/fold (noisier); 3 folds → coarser tuning signal |
| 2026-07-08 | `max_evals = 20` for the Hyperopt search | Bounded search given placement-season time constraints; enough to move meaningfully off defaults | Would raise to 100s+ (with early stopping) for a real deployment; 20 trades tuning depth for turnaround — same spirit as the fold-count sanity check |
| 2026-07-08 | Leakage prevention: scaler/imputer fit **inside** the sklearn Pipeline | The pipeline is refit within each CV fold and on X_train only for the final model, so median/IQR/means never see held-out rows | Fitting a scaler on the full dataset before splitting (a common bug) leaks the test distribution into training |
| 2026-07-08 | Local `creditcard.csv` over continuing the DVC/S3 flow | Prioritized modeling depth over infra continuity given placement-season time; the DVC/S3 pattern is already demonstrated in the baseline, so re-deriving it here adds no new understanding | Raw CSV stays local/untracked; Docker/CI (which `dvc pull`s) no longer builds cleanly — accepted, see Known Gaps |
| 2026-07-08 | Single-source preprocessing via `pipeline.build_pipeline()` | Removes the baseline's hand-duplicated preprocessing across `pipeline.py`/`training_pipeline.py` (the Phase-1 drift risk); training appends the classifier to the same builder | — |
| 2026-07-08 | Rewrote the only test into a schema/type + split gate | Closes the Phase-1 "not a real regression gate" vuln — the old test hard-asserted one row == 'Y' and broke on any best-run change | New gate is offline/deterministic (no MLflow, no CSV): Pydantic type validation + stratified-split invariants |
| 2026-07-10 | Added `pytest.ini` (`pythonpath = .`) instead of adding `tests/__init__.py` or a `conftest.py` | `tests/` has no `__init__.py`, so pytest's default "prepend" import mode roots sys.path at `tests/`, not the repo root, so `import prediction_model` failed; `pythonpath = .` is pytest 7.4.2's (already pinned) built-in fix for exactly this, needs no code/package changes | Alternative: add `tests/__init__.py` to make tests part of a package — rejected, doesn't by itself put the repo root on sys.path either without also relying on rootdir insertion behavior, so it's less direct than the ini option |
| | e.g. SHAP vs drift monitoring priority given time (Phase 3) | | |

## Checkpoint Tracker
(Only advance to the next phase if you can genuinely check these off — don't
self-grade generously.)

- [~] Phase 1 checkpoint: explained current architecture unaided, no notes —
      partial: nailed MLflow coupling/hot-swap risk, gap on NaN mechanism +
      fraud-specific reasoning (see Known Gaps)
- [ ] Phase 2 checkpoint: explained PR-AUC choice, imbalance method, leakage
      prevention unaided
- [ ] Phase 3a checkpoint: explained SHAP latency tradeoff, explainer choice,
      and response design unaided
- [ ] Phase 3b checkpoint: explained drift baseline definition and automation
      scope unaided
- [ ] Phase 4: first interview drill completed — date: ___
- [ ] Phase 4: re-drill after a few days (retention check) — date: ___

## Known Gaps / Open Questions
(Anything you couldn't explain cleanly at a checkpoint goes here until resolved.)
- Phase 1 quiz (2026-07-07): correctly explained MLflow per-request coupling
  and the ungated hot-swap risk unaided. On silent-NaN propagation: picked a
  wrong example (`LoanAmount: "Elephant"` — Pydantic's `float` type already
  blocks this at the API boundary, so it never reaches the pipeline). The
  real vector is unseen values on `str`-typed categorical fields hitting
  `CustomLabelEncoder`'s `.map()` lookup ([preprocessing.py:64-79](prediction_model/processing/preprocessing.py:64)),
  which Pydantic can't catch since any string is a valid string. Also didn't
  articulate *why* this is worse for fraud specifically vs. loan-approval:
  no human-in-the-loop + immediate money movement + adversarial incentive to
  probe encoder blind spots (loan approval has underwriting delay as a
  safety net; nobody is adversarially crafting `Property_Area` values).
  Revisit before Phase 4 interview drills — distinguish "type validation"
  (Pydantic, already present) from "domain/category validation" (missing)
  when explaining input hardening.
  - Update (2026-07-08): the `CustomLabelEncoder` `.map()` vector referenced above no
    longer exists in code — it was removed in Phase 2 (fraud data is all-numeric, no
    categorical encoding). The *conceptual* distinction (type vs domain/category
    validation) is still the thing to be able to articulate in the Phase 4 drill.
- Phase 2 runtime blockers (2026-07-08) — code is complete but was NOT run end-to-end
  this session: (1) `creditcard.csv` not present locally; (2) venv is Python 3.14 with an
  empty/incompatible package set vs the pinned legacy stack — needs a Python 3.10 venv or
  modernized pins; (3) baseline MLflow EC2 server is down (now defaulting to `file:./mlruns`).
  None are code defects, but training/serving can't be demonstrated until they're resolved.
  - **Resolved (2026-07-10)**: all three fixed — CSV placed locally, `env/` rebuilt as a
    Python 3.10.11 venv with `requirements.txt` installed, EC2 server confirmed unreachable
    (local `file:./mlruns` in use). A fourth, previously-undiscovered blocker also surfaced
    and was fixed: `pytest`/`python <script>.py` path resolution (see Decisions Log,
    2026-07-10, `pytest.ini`). Training verified end-to-end with `pr_auc = 0.8693` on the
    holdout — see Current Status for full metrics.
- Deferred consequence to remember (2026-07-08): Docker build + CI (`.github/workflows/main.yml`)
  still `dvc pull` + train loan data and will not build cleanly against the local-CSV fraud flow.
  Accepted cost of the infra de-prioritization; revisit if the deployment path is revived (Phase 3+).