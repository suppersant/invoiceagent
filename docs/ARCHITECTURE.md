# Architecture Notes

## Pipeline Flow

```
ingest → extract → structure → qa → deliver
```

## Agent Responsibilities

| Stage     | Agent              | Input                        | Output                          |
|-----------|--------------------|------------------------------|---------------------------------|
| Ingest    | Ingestion Agent    | Raw PDF from email/upload    | Page images + metadata          |
| Extract   | Extraction Agent   | Page images                  | Raw field extractions           |
| Structure | Structuring Agent  | Raw extractions              | Canonical JSON (invoice schema) |
| QA        | QA Agent           | Structured JSON              | Confidence-scored JSON          |
| Deliver   | Delivery Agent     | QA-passed JSON               | CSV/JSON output files           |

## Key Design Decisions

- **No frameworks** (Constitution A1): All agents are plain Python functions calling
  the Anthropic SDK directly. This keeps the codebase auditable and debuggable.

- **Sequential pipeline** (Constitution A2): No parallel execution or dynamic routing
  in MVP. The orchestrator calls agents in fixed order.

- **Single responsibility** (Constitution A3): One file = one job. Agents never call
  each other. Only the orchestrator composes them.

- **Store everything** (Constitution B1): Raw documents, intermediate results, and
  final outputs are all retained for full reproducibility.

- **Confidence gating** (Constitution C2): Invoices scoring below 0.85 confidence
  are routed to human review. No auto-delivery in MVP.

## Directory Layout

```
invoiceagent/
├── CLAUDE.md                  # Project constitution
├── pyproject.toml             # Dependencies and build config
├── .env.example               # Environment variable template
├── src/
│   ├── __init__.py
│   ├── config.py              # Central configuration (env vars)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── ingest_agent.py    # PDF → page images
│   │   ├── extract_agent.py   # Images → raw field data
│   │   ├── structure_agent.py # Raw data → canonical schema
│   │   ├── qa_agent.py        # Schema → confidence-scored output
│   │   └── deliver_agent.py   # Scored output → CSV/JSON
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── orchestrator.py    # Runs agents in sequence
│   ├── models/
│   │   ├── __init__.py
│   │   └── invoice.py         # Pydantic models for invoice data
│   └── utils/
│       ├── __init__.py
│       └── logging.py         # Structured logging setup
├── tests/
│   └── __init__.py
├── data/
│   └── clients/
│       └── {client_id}/
│           ├── inbox/         # Raw uploaded invoices
│           ├── processed/     # QA-passed output
│           └── review/        # Below-threshold invoices
└── docs/
    ├── ARCHITECTURE.md        # This file
    └── SCHEMA.md              # Invoice data schema docs
```
