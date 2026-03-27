# InvoiceAgent — Constitution (CLAUDE.md)

## Project Identity
InvoiceAgent is an AI-powered invoice processing service for SMB manufacturing
and distribution companies. It takes unstructured PDF invoices received via email,
extracts all line-item and header data using Claude's vision capabilities,
structures that data into a canonical JSON schema, scores confidence, and delivers
clean CSV/JSON output to clients.

## [A] Architecture Rules

A1: All agents are plain Python modules calling the Anthropic SDK directly.
    No frameworks. No CrewAI. No LangChain. No LangGraph. No AutoGen.
    Each agent is a function with typed input and typed output.

A2: The pipeline is strictly sequential.
    ingest → extract → structure → qa → deliver.
    No parallel agent execution. No dynamic routing in MVP.

A3: Every module has a single responsibility.
    One file = one job. Agents do not call other agents directly.
    The orchestrator is the only module that calls agents in sequence.

A4: All function signatures use Python type hints.
    Input and output types are Pydantic models defined in src/models/.

A5: All file paths use pathlib.Path. Never string concatenation for paths.

A6: Configuration lives in environment variables loaded via a single config module.
    No config values hardcoded in agent files.

A7: FastAPI is Phase 2 infrastructure. MVP uses CLI and email ingestion only.

## [B] Data Rules

B1: STORE EVERYTHING. Never delete raw uploaded documents. Never delete
    intermediate extraction results. Every processing run is fully reproducible.

B2: Every processed invoice gets a UUID. This ID is used across all downstream
    references — extraction results, QA scores, delivery outputs, logs.

B3: Client data is strictly isolated. No cross-client data access. Directory
    structure and database queries enforce tenant isolation.

B4: All extracted data includes source attribution — page number and bounding
    region where the data was found. This is non-negotiable for QA tracing.

B5: Raw invoices are stored as-is in client-specific directories.
    Processed outputs go to separate output directories per client.

## [C] Quality Rules

C1: Every field extraction includes a confidence score from 0.0 to 1.0.
    The confidence is assigned by the QA agent, not the extraction agent.

C2: HARD THRESHOLD — Invoice-level confidence below 0.85 routes the entire
    invoice to the human review queue. This threshold is configurable per client
    but defaults to 0.85.

C3: No output CSV/JSON is generated for an invoice that has not passed QA.
    The pipeline halts at QA and writes to the review queue instead.

C4: All agent executions are logged with: timestamp, invoice_id, client_id,
    agent_name, input_hash, output_hash, duration_ms, confidence_score.

C5: Accuracy target is 95%+ on field extraction before any client demo.
    Do not demo to prospects until this threshold is validated on test data.

## [D] Security Rules

D1: No client financial data (dollar amounts, vendor names, account numbers)
    appears in application logs. Logs contain only IDs, timestamps, and scores.

D2: All client documents are encrypted at rest using the filesystem or S3
    server-side encryption. No unencrypted document storage in production.

D3: API keys and secrets live in environment variables ONLY.
    Never in code. Never in config files committed to git.

D4: All client relationships operate under NDA. The system treats every
    document as confidential by default.

D5: No client data is used to improve extraction for other clients unless
    explicitly anonymized and aggregated.

## [E] Anti-Patterns — NEVER DO THESE

E1: NEVER auto-send processing results to a client without human review in MVP.
    Results go to a review queue. Operator approves. Then delivery happens.

E2: NEVER modify or overwrite raw uploaded invoice files. They are immutable.

E3: NEVER hardcode client-specific extraction rules or vendor-specific templates.
    The extraction agent must handle arbitrary invoice formats using vision.

E4: NEVER skip the QA agent in the pipeline. Even for "simple" invoices.

E5: NEVER store Anthropic API keys or any secrets in source code or git.

E6: NEVER use print() for logging. Use the structured logging module.

E7: NEVER catch broad exceptions silently. All errors must be logged and
    surfaced to the operator.

## [F] Naming Conventions

F1: Agent modules:        src/agents/{agent_name}_agent.py
F2: Pipeline modules:     src/pipeline/{module_name}.py
F3: Data models:          src/models/{model_name}.py
F4: Utility modules:      src/utils/{util_name}.py
F5: Configuration:        src/config.py
F6: Tests:                tests/test_{module_name}.py
F7: Client data:          data/clients/{client_id}/inbox/
F8: Processing output:    data/clients/{client_id}/processed/
F9: Review queue:         data/clients/{client_id}/review/

## [G] Dependencies — Approved Libraries

G1: anthropic — Anthropic Python SDK (required)
G2: pydantic — Data validation and schemas (required)
G3: pdf2image + poppler — PDF to image conversion (required)
G4: Pillow — Image handling (required)
G5: python-dotenv — Environment variable loading (required)
G6: sqlite3 — Database (stdlib, no install needed)
G7: imaplib — Email ingestion (stdlib, no install needed)
G8: csv, json — Output generation (stdlib, no install needed)
G9: fastapi + uvicorn — API layer (Phase 2 only, not MVP)
G10: pytest — Testing (required)
G11: structlog — Structured logging (required)

No other libraries without explicit constitution amendment.
