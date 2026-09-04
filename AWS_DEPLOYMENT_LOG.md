# AWS Deployment Log — credit-fraud-mlops

Tracking the build-out from *"public endpoint, no security, undeployed"* to a live, HTTPS,
authenticated deployment on AWS.

**Region:** `us-east-1` (all regional resources must share it)
**Repo:** https://github.com/ShamikSeng/credit-fraud-mlops
**Started:** 2026-09-04

---

## Progress

- [ ] **Phase 1 — Containerize & push to ECR, run on EC2**
- [ ] **Phase 2 — RDS Postgres + prediction audit table (Alembic)**
- [ ] **Phase 3 — Route 53 + ACM certificate + Application Load Balancer**
- [ ] **Phase 3.5 — Test the full teardown / rebuild cycle**
- [ ] **Phase 4 — API security (X-API-Key, rate limiting, /docs off in prod)**
- [ ] **Phase 5 — AWS Secrets Manager**
- [ ] **Phase 6 — CI/CD via GitHub Actions (OIDC → ECR → EC2)**
- [ ] **Phase 7 — ARCHITECTURE.md + AWS_INTERVIEW_QNA.md**

---

## Phase 1 — Containerize & push to ECR, run on EC2

**Goal:** a self-contained image (model baked in) running on EC2, pulled from ECR via an instance
role, returning a real prediction over HTTP.

### Local code
- [ ] Multi-stage `Dockerfile` (builder → test → runtime), non-root `appuser`
- [ ] `.dockerignore`: stop excluding `mlruns/` and `reference_data.csv` — **without this the
      container starts, passes health checks, then fails on the first prediction**
- [ ] Move `pytest` from `requirements.txt` to `requirements-dev.txt`
- [ ] Local `docker build` + `docker run` returns a real prediction (verify BEFORE the multi-GB push)

### AWS
- [ ] Install AWS CLI v2; create IAM user; `aws configure` (us-east-1)
- [ ] Create ECR repository `credit-fraud-mlops` (scan-on-push enabled)
- [ ] `docker login` to ECR; build, tag `:v1`, push
- [ ] Create IAM **role for EC2** with `AmazonEC2ContainerRegistryReadOnly` (no AWS keys on the box)
- [ ] Launch EC2: Amazon Linux 2023, **t3.small**, **30 GiB gp3**, key pair, SG = SSH + 8005 from my IP only
- [ ] SSH in: install Docker, add 2 GiB swap, pull from ECR
- [ ] `docker run -d --restart unless-stopped -p 8005:8005 <ecr-uri>:v1`
- [ ] Verify `GET /`, `POST /prediction_api`, `GET /metrics`

**Exit criteria:** a real `fraud_probability` returned from the EC2 public IP, and the container comes
back after `docker restart`.

---

## Phase 2 — RDS Postgres + prediction audit table

**Goal:** every scored transaction is persisted, so the fraud engine's decisions are auditable.

- [ ] `prediction_model/db/` — SQLAlchemy engine/session + `Prediction` model
- [ ] `alembic/` scaffold + initial migration
- [ ] Persist via `BackgroundTasks` (off the request latency path, same as drift)
- [ ] `GET /predictions` (paginated)
- [ ] Create RDS `db.t4g.micro`, **Public access = No**
- [ ] RDS security group: inbound 5432 sourced from the **EC2 security group ID**, not a CIDR
- [ ] Run `alembic upgrade head` from the EC2 instance
- [ ] Verify a prediction lands as a row

---

## Phase 3 — Route 53 + ACM + ALB

- [ ] Register domain in Route 53
- [ ] Request public ACM certificate (DNS validation, auto-validated in Route 53)
- [ ] Create ALB across two public subnets + its own security group
- [ ] Target group → EC2 instance on 8005, health check `/`
- [ ] HTTPS:443 listener with the ACM cert
- [ ] HTTP:80 listener as a **301 redirect action** (not a forward)
- [ ] Route 53 alias record → ALB
- [ ] **Lock down:** EC2 SG now accepts 8005 only from the ALB SG (no longer reachable from the internet)
- [ ] Verify `https://<domain>/` and that `http://` redirects

---

## Phase 3.5 — Teardown / rebuild cycle test

Run once immediately after Phase 3, while there is still time to fix it.

- [ ] `scripts/aws-env.sh` — stable IDs + target-group health-check settings
- [ ] `scripts/teardown.sh` — stop EC2 + RDS; delete ALB, then target group
- [ ] `scripts/rebuild.sh` — start RDS + EC2; recreate ALB/TG/listeners; UPSERT Route 53 alias
- [ ] Full cycle passes: same domain, same curl, zero manual steps

---

## Phase 4 — API security

- [ ] `X-API-Key` dependency on scoring + monitoring routes (`/` and `/metrics` stay open for the ALB health check and Prometheus)
- [ ] `slowapi` rate limiting
- [ ] `/docs`, `/redoc`, `/openapi.json` disabled when `ENV=production`
- [ ] Replace the `allow_origins=["*"]` + `allow_credentials=True` CORS config in `main.py`
- [ ] Verify: valid key → 200, no key → 401, over limit → 429, `/docs` → 404

---

## Phase 5 — AWS Secrets Manager

- [ ] Create one secret holding DB credentials + the API key
- [ ] `boto3` fetch at startup, cached in-process, env-var fallback for local dev
- [ ] Extend the EC2 instance role with `secretsmanager:GetSecretValue` scoped to that **one secret ARN**
- [ ] Remove credentials from `.env` / `docker run -e`

---

## Phase 6 — CI/CD

- [ ] IAM OIDC identity provider for GitHub Actions
- [ ] IAM role, trust policy scoped to `ShamikSeng/credit-fraud-mlops` + `main` ref (no long-lived keys)
- [ ] Rewrite `.github/workflows/main.yml`: test → build → push to ECR → SSH redeploy
- [ ] Verify: a push to `main` lands on EC2 unattended

---

## Phase 7 — Documentation

- [ ] `ARCHITECTURE.md` — request path, network topology, IAM boundaries, single-worker rationale, baked-model tradeoff + S3 scaling path
- [ ] `AWS_INTERVIEW_QNA.md` — ALB+ACM vs self-signed; Secrets Manager vs env files; RDS vs containerized Postgres; ECR vs Docker Hub; instance role vs keys; OIDC vs CI secrets; SG-as-source vs CIDR

---

## Live resource inventory

Fill in as resources are created. This table is what `teardown.sh` is written against, and what stops
a resource being forgotten and quietly billed.

| Resource | Name / ID / ARN | Region | ~$/mo running | On teardown |
|---|---|---|---|---|
| ECR repository | | us-east-1 | ~0.30 | keep |
| EC2 instance | | us-east-1 | ~15 | **stop** |
| EC2 EBS root (30 GiB gp3) | | us-east-1 | ~2.40 | keep (bills while stopped) |
| IAM role (EC2 instance profile) | | global | 0 | keep |
| RDS instance | | us-east-1 | ~12 | **stop** |
| RDS storage + backups | | us-east-1 | ~2.30 | keep (bills while stopped) |
| Application Load Balancer | | us-east-1 | ~16 | **delete** |
| Target group | | us-east-1 | 0 | **delete** |
| ACM certificate | | us-east-1 | 0 | leave alone |
| Route 53 hosted zone | | global | 0.50 | leave alone |
| Domain registration | | global | ~1 (~$12/yr) | leave alone |
| Secrets Manager secret | | us-east-1 | 0.40 | keep |

**Running total:** ~$50/mo · **Stopped floor:** ~$6-9/mo (storage and DNS still bill — "stopped" is not "free")

### The 7-day RDS trap

**AWS force-starts a stopped RDS instance after 7 days** and billing silently resumes. For any gap
longer than a week, take a final snapshot and delete the instance instead of stopping it.

- Last teardown: _(not yet run)_
- RDS auto-restart date: _(teardown.sh will print this)_

---

## Decisions log

| Date | Decision | Why |
|---|---|---|
| 2026-09-04 | Bake `mlruns/` into the image rather than mounting or fetching it | Only 1.2 MB, and it makes the image a self-contained immutable artifact — model version pinned to the image tag, so rollback is a tag change. S3-backed artifact store documented as the scaling path. |
| 2026-09-04 | t3.small, not free-tier t3.micro | `drift.compute_drift` forks a second interpreter that imports Evidently, roughly doubling peak RSS. 1 GiB would OOM when the drift window fills. |
| 2026-09-04 | Single uvicorn worker | `predict._CACHE` and `drift._BUFFER`/`_COUNT` are per-process globals; multiple workers would silently break the `DRIFT_WINDOW` trigger and reload the model per worker. |
| 2026-09-04 | Added an audit table rather than dropping Phase 2 | The app had no database at all. A fraud engine that doesn't record its decisions isn't auditable — this gives RDS an honest reason to exist. |
| 2026-09-04 | API key, not JWT | Machine-to-machine scoring API: no users, no sessions, nothing for a JWT to carry. |
| 2026-09-04 | AWS CLI scripts, not Terraform, for the ALB lifecycle | Matches how the ALB is built by hand in Phase 3, so every line is explainable; avoids hand-built-then-imported state drift. |
