import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# address currently in use, run: lsof -ti :8000 | xargs kill -9
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env", override=True)

# Existing agents
from coordinator import fetch_10k, build_user_profile, DEFAULT_IDENTITY
from risk_research_agent import run_risk_research
from tenk_agent import analyze_10k_report
from market_sense import run_market_sense_agent
from advisor import advise

# New helpers (both degrade gracefully to {} if unavailable)
from business_overview import summarize_business
from financials_trends import fetch_financial_trends

app = FastAPI(title="SimplifyNext Equity Research Desk Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    ticker: str
    profile: Dict[str, Any]


# =============================================================================
# SMALL HELPERS  --  shared by the plain and streaming endpoints
# =============================================================================

def _identity() -> str:
    return os.environ.get("EDGAR_IDENTITY", DEFAULT_IDENTITY)


def _money_to_float(text: Any) -> Optional[float]:
    """'$228.45' -> 228.45 ; '$215.94B' -> 215940000000.0 ; 'N/A' -> None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().upper().replace("$", "").replace(",", "")
    if s in ("", "N/A", "NA", "NONE", "-"):
        return None
    mult = 1.0
    for suffix, factor in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if s.endswith(suffix):
            mult, s = factor, s[:-1]
            break
    try:
        return float(s) * mult
    except ValueError:
        return None


# Plain-English one-liners per pillar + signal (frontend item 3 & 5).
_VERDICTS = {
    "quality": {
        "strong":  "A genuinely strong business \u2014 it passes the core health checks.",
        "neutral": "A decent business, with a few soft spots to keep an eye on.",
        "weak":    "The business has real weak spots \u2014 treat any low price with caution.",
        "unknown": "We couldn't fully check the business's health from the filing.",
    },
    "valuation": {
        "strong":  "Looks cheap compared with what it appears to be worth.",
        "neutral": "Fairly priced \u2014 not an obvious bargain, not obviously expensive.",
        "weak":    "Looks expensive compared with what it appears to be worth.",
        "unknown": "We couldn't estimate a fair price for this one.",
    },
    "narrative": {
        "strong":  "Management's story matches the audited numbers.",
        "neutral": "Management's story mostly holds up, with a few things to watch.",
        "weak":    "Management's story doesn't fully line up with the numbers.",
        "unknown": "Not enough disclosure to judge the story either way.",
    },
}


def _pillar_ui(pillar: dict, kind: str) -> dict:
    """Turn an advisor Pillar dict into {signal, verdict, bullets} for the cards."""
    signal = str((pillar or {}).get("signal", "unknown")).lower()
    if signal not in ("strong", "neutral", "weak", "unknown"):
        signal = "unknown"
    evidence = [e for e in (pillar or {}).get("evidence", []) if isinstance(e, str) and not e.startswith("__")]
    caveats = [c for c in (pillar or {}).get("caveats", []) if isinstance(c, str)]
    bullets = (evidence[:2] + caveats[:1])[:3]
    return {
        "signal": signal,
        "verdict": _VERDICTS.get(kind, {}).get(signal, ""),
        "bullets": bullets,
    }


def _valuation_block(risk_output: dict) -> dict:
    cons = (risk_output.get("metrics", {}) or {}).get("market_consensus", {}) or {}
    cur = _money_to_float(cons.get("current_price"))
    tgt = _money_to_float(cons.get("target_price_mean"))
    upside = ((tgt - cur) / cur * 100.0) if (cur and tgt and cur > 0) else None
    return {
        "current_price": round(cur, 2) if cur is not None else None,
        "target_price": round(tgt, 2) if tgt is not None else None,
        "implied_upside_pct": round(upside, 1) if upside is not None else None,
    }


def _pretty_upside(raw: Any) -> str:
    s = str(raw or "").strip()
    return "\u2014" if s.lower() in ("", "n/a", "na", "none") else s


def build_payload(ticker: str,
                  risk_output: dict,
                  tenk_output: dict,
                  market_output: dict,
                  advisor_output: dict,
                  business_overview: dict,
                  trends: dict) -> dict:
    """Assemble the exact JSON the frontend (index.html) consumes."""
    pillars = advisor_output.get("pillars", {}) or {}
    quality_ui = _pillar_ui(pillars.get("quality", {}), "quality")
    valuation_ui = _pillar_ui(pillars.get("valuation", {}), "valuation")
    narrative_ui = _pillar_ui(pillars.get("narrative", {}), "narrative")

    verdict = str(advisor_output.get("recommendation", "HOLD")).upper()
    conviction_info = advisor_output.get("conviction", {})
    conviction_label = (conviction_info.get("label", "MEDIUM").upper()
                        if isinstance(conviction_info, dict) else "MEDIUM")

    tenk_out = dict(tenk_output or {})
    if business_overview:
        tenk_out["business_overview"] = business_overview

    return {
        "status": "success",
        "ticker": ticker,
        "advisor": {
            "verdict": verdict,
            "conviction": conviction_label,
            "implied_upside": _pretty_upside(advisor_output.get("implied_upside")),
            "quality_gate": {
                "passed": quality_ui["signal"] == "strong",
                "signal": quality_ui["signal"],
                "verdict": quality_ui["verdict"],
                "bullets": quality_ui["bullets"],
                "assessment": quality_ui["verdict"] or "Quality metrics evaluated.",
            },
            "valuation_assessment": {
                "status": valuation_ui["signal"].upper(),
                "signal": valuation_ui["signal"],
                "verdict": valuation_ui["verdict"],
                "bullets": valuation_ui["bullets"],
                "assessment": valuation_ui["verdict"] or "Valuation evaluated.",
            },
            "narrative_integrity": {
                "credibility": "HIGH" if narrative_ui["signal"] == "strong"
                               else ("MODERATE" if narrative_ui["signal"] == "neutral" else "LOW"),
                "signal": narrative_ui["signal"],
                "verdict": narrative_ui["verdict"],
                "bullets": narrative_ui["bullets"],
                "assessment": narrative_ui["verdict"] or "Management disclosures vetted.",
            },
            "actionable_execution": advisor_output.get("rationale")
                or f"Suggested position size: {advisor_output.get('position_size', 'modest')}.",
            "profile_alignment": (advisor_output.get("personalization", {}) or {}).get("summary")
                or " ".join((advisor_output.get("personalization", {}) or {}).get("notes", []))
                or "Aligned with the answers you gave.",
        },
        "risk": {
            "signal": risk_output.get("signal", "neutral"),
            "scorecard": risk_output.get("scorecard", []),
            "metrics": risk_output.get("metrics", {}),
            "key_points": risk_output.get("key_points", []),
        },
        "financials": {
            "trends": trends or {},
            "valuation": _valuation_block(risk_output),
        },
        "tenk": {
            "financial_health": tenk_out.get("financial_health", {}),
            "management_claims": tenk_out.get("management_claims", {}),
            "business_overview": tenk_out.get("business_overview", {}),
        },
        "market": {
            "signal": market_output.get("signal", "neutral"),
            "summary": market_output.get("summary", "Market backdrop synthesized."),
            "tailwinds": market_output.get("tailwinds", []),
            "headwinds": market_output.get("headwinds", []),
            "key_points": market_output.get("key_points", []),
        },
    }


# =============================================================================
# PLAIN ENDPOINT  --  unchanged contract, now with the enriched payload
# =============================================================================

@app.post("/api/analyze")
async def analyze_equity(request: AnalysisRequest):
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")
    try:
        identity = _identity()
        profile_obj = build_user_profile(request.profile)
        filing_data = fetch_10k(ticker, identity)
        context = {"profile": profile_obj.to_dict(), "ticker": ticker, **filing_data}
        context.setdefault("company", {})["ticker"] = ticker

        tenk_output = analyze_10k_report(context)
        biz = summarize_business((context.get("sections", {}) or {}).get("business", ""),
                                 (context.get("company", {}) or {}).get("name", ticker))
        risk_output = run_risk_research(context)
        trends = fetch_financial_trends((context.get("company", {}) or {}).get("cik"), identity)
        market_output = run_market_sense_agent(context)
        advisor_output = advise(context=context, risk_research_out=risk_output,
                                tenk_out=tenk_output, market_sense_out=market_output)

        return build_payload(ticker, risk_output, tenk_output, market_output,
                             advisor_output, biz, trends)
    except LookupError as le:
        raise HTTPException(status_code=404, detail=str(le))
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")


# =============================================================================
# STREAMING ENDPOINT  --  NDJSON, one line per event (frontend item 8)
# Emits {"type":"log", stage, message, level, pct} events as each agent
# finishes, then a final {"type":"result", data}. The sync generator is run in
# a threadpool by Starlette, so the blocking agent calls are fine here.
# =============================================================================

def _nd(obj: dict) -> str:
    return json.dumps(obj) + "\n"


def _log(stage: str, message: str, level: str = "info", pct: Optional[float] = None) -> str:
    return _nd({"type": "log", "stage": stage, "message": message, "level": level, "pct": pct})


def _analysis_events(ticker: str, profile: Dict[str, Any]):
    try:
        identity = _identity()
        yield _log("fetch_10k", f"resolving {ticker} on SEC EDGAR\u2026", "info", 5)

        profile_obj = build_user_profile(profile)
        filing_data = fetch_10k(ticker, identity)
        context = {"profile": profile_obj.to_dict(), "ticker": ticker, **filing_data}
        context.setdefault("company", {})["ticker"] = ticker
        period = (context.get("filing", {}) or {}).get("period", "")
        fdate = (context.get("filing", {}) or {}).get("filing_date", "")
        yield _log("fetch_10k", f"pulled latest 10-K ({period or 'annual'}, filed {fdate or 'n/a'})", "ok", 25)

        # --- 10-K narrative + plain-English business summary (stage 1) ---
        yield _log("fetch_10k", "reading the 10-K narrative (MD&A + business)\u2026", "info", 32)
        tenk_output = analyze_10k_report(context)
        yield _log("fetch_10k", "10-K narrative analysed", "ok", 40)

        yield _log("fetch_10k", "writing a plain-English business summary\u2026", "info", 44)
        biz = summarize_business((context.get("sections", {}) or {}).get("business", ""),
                                 (context.get("company", {}) or {}).get("name", ticker))
        yield _log("fetch_10k", "business summary ready" if biz else "business summary skipped (no data)",
                   "ok" if biz else "warn", 48)

        # --- numbers (stage 2) ---
        yield _log("risk", "reading balance sheet & income statement\u2026", "info", 55)
        risk_output = run_risk_research(context)
        gate = "PASSED" if str(risk_output.get("signal", "")).lower() in ("positive", "strong") else "reviewed"
        yield _log("risk", f"ratios computed \u00b7 health check: {gate}", "ok", 64)

        yield _log("risk", "pulling multi-year history from SEC\u2026", "info", 68)
        trends = fetch_financial_trends((context.get("company", {}) or {}).get("cik"), identity)
        n = len(trends.get("years", [])) if trends else 0
        yield _log("risk", f"{n}-year history loaded" if n else "history unavailable \u2014 using latest year only",
                   "ok" if n else "warn", 72)

        # --- market backdrop (stage 3) ---
        yield _log("market", "scanning macro (FRED) & recent headlines\u2026", "info", 78)
        market_output = run_market_sense_agent(context)
        msig = str(market_output.get("signal", "neutral")).lower()
        yield _log("market", f"market backdrop: {msig}", "ok", 84)

        # --- portfolio-manager synthesis (stage 4) ---
        yield _log("advisor", "portfolio-manager layer weighing the 4 pillars\u2026", "info", 88)
        advisor_output = advise(context=context, risk_research_out=risk_output,
                                tenk_out=tenk_output, market_sense_out=market_output)
        conv = (advisor_output.get("conviction", {}) or {}).get("label", "MEDIUM")
        yield _log("advisor", f"gated call formed \u00b7 confidence: {conv}", "ok", 96)

        payload = build_payload(ticker, risk_output, tenk_output, market_output,
                               advisor_output, biz, trends)
        yield _nd({"type": "result", "data": payload})
        yield _log("done", "done \u2014 building your report", "ok", 100)

    except LookupError as le:
        yield _nd({"type": "error", "message": str(le)})
    except Exception as exc:
        traceback.print_exc()
        yield _nd({"type": "error", "message": f"Pipeline error: {str(exc)}"})


@app.post("/api/analyze/stream")
async def analyze_equity_stream(request: AnalysisRequest):
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")
    return StreamingResponse(
        _analysis_events(ticker, request.profile),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve index.html from ./static at the root URL.
static_dir = ROOT_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\nStarting SimplifyNext Multi-Agent Backend on http://127.0.0.1:8000 ...\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
