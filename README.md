# Tickr — Multi-Agent Institutional Equity Research Desk

A proof-of-concept multi-agent research platform demonstrating collaborative AI agents that ingest SEC 10-K filings, calculate quantitative risk metrics, evaluate macro trends, and synthesize personalized investment memos.

## Table of Contents

- [1. Overview of Code & File Purpose](#1-overview-of-code--file-purpose)
- [2. Environment Setup](#2-environment-setup)
- [3. Language & Stack](#3-language--stack)
- [4. Execution Instructions](#4-execution-instructions)

---

## 1. Overview of Code & File Purpose

### a) Instructions to Run the Code

See [Section 4](#4-execution-instructions) below for complete execution steps.

### b) Script & File Overview

| File | Purpose |
Frontend (Client Presentation)
| `static/index.html` | Minimalist web interface with animated progress tracking and interactive formula modals. |

Backend Orchestration & Ingestion
| `server.py` | FastAPI backend exposing the `/api/analyze` orchestration endpoint and mounting the static UI. |
| `coordinator.py` | Intake agent that captures investor mandate criteria and queries the latest annual 10-K report from SEC EDGAR. |
| `business_overview.py` | Plain-English business model summarizer parsing Item 1 disclosures and revenue segment mix. |
| `financials_trends.py` | Historical trend engine retrieving multi-year SEC XBRL facts (revenue, profit, net margin, FCF).|

Analysis Desks (Sub-Agents)
| `risk_research_agent.py` | Quantitative desk calculating solvency, liquidity, and profitability ratios, plus Wall Street consensus comparisons via Yahoo Finance. |
| `tenk_agent.py` | Audited filing desk that uses Claude on AWS Bedrock to extract MD&A financial health and management guidance. |
| `market_sense.py` | Macroeconomic desk tracking FRED indicators and news momentum, with automated fallback handling. |

Decision Layer
| `advisor.py` | Portfolio Manager (CIO) layer that scores 4 fundamental pillars (Quality, Valuation, Narrative, Backdrop) using gated risk controls. |

---

## 2. Environment Setup

### a) Virtual Environment & Dependencies

Create and activate an isolated Python environment, then install the pinned libraries from `requirements.txt`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### b) Path Variables

The project uses a flat root-level structure. Ensure your terminal is rooted in the project directory so all local modules resolve directly, without needing to modify `PYTHONPATH`.

### c) Secrets & Keys (`.env`)

Create a `.env` file in the root folder containing the required API keys:

```env
AWS_ACCESS_KEY_ID="your_aws_access_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
AWS_SESSION_TOKEN="your_aws_session_token"
AWS_DEFAULT_REGION="us-east-1"
BEDROCK_MODEL_ID="us.anthropic.claude-haiku-4-5-20251001-v1:0"

EDGAR_IDENTITY="YourName yourname@example.com"
FRED_API_KEY="your_fred_api_key"
ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```

> **Note:** Never commit your `.env` file to version control. Add it to `.gitignore`.

---

## 3. Language & Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Backend Framework** | FastAPI, Starlette StreamingResponse, Uvicorn |
| **AI & Orchestration** | AWS Bedrock (Claude 3 Haiku / Sonnet) |
| **Financial Data Ingestion** | `edgartools` (SEC EDGAR), SEC XBRL API, `yfinance`, St. Louis Fed FRED API, Alpha Vantage |
| **Frontend** | Vanilla HTML5, Tailwind CSS, Chart.js, NDJSON Streaming Client |

---

## 4. Execution Instructions

### a) Running the Web Application

Start the unified local server:

```bash
python server.py
```

Then open [http://localhost:8000](http://localhost:8000) in your browser, enter a US ticker (e.g., `NVDA`, `AAPL`), select your mandate settings, and run the pipeline.

### b) Methodological Implementation

The decision engine follows an institutional 4-pillar gated model defined in `advisor.py`:

1. **Quality Gate** — Evaluates fundamental solvency and ROE; low quality prevents Buy calls.
2. **Valuation** — Calculates margin of safety relative to consensus target prices.
3. **Narrative Integrity** — Audited SEC 10-K disclosures cross-check management guidance.
4. **Market Backdrop** — Macro interest rates and industry news define tactical timing.

### c) Standalone CLI & Zero-API Demonstration

**Live terminal pipeline:**

```bash
python advisor.py
```

**Offline demo** (sample NVDA run without live AWS tokens):

```bash
python advisor.py --demo
```

### Troubleshooting
* **Port Conflict (`Errno 48: Address already in use`):** If port 8000 is occupied by a previous session, release it via terminal (`lsof -ti :8000 | xargs kill -9` on macOS/Linux) or specify a different port (`uvicorn server:app --port 8001`).