"""
Advisor agent  --  "The Overwhelmed Everyday Investor"

This is the agent that sits ABOVE the three analysis sub-agents and plays the
role a Portfolio Manager plays on a real research desk: it reads three
specialist reports and turns them into ONE accountable call.

It consumes the outputs of:
  - Risk & Research agent   (the numbers  -> Quality + Valuation)
  - 10-K Analysis agent     (Financial Health + Management Claims
                             -> Quality + Narrative integrity)
  - Market Sense agent       (macro / sentiment -> Backdrop / timing)
...plus the user profile the coordinator already built.

------------------------------------------------------------------------------
THE METHODOLOGY (how a professional actually forms a view)
------------------------------------------------------------------------------
One idea drives everything: a good COMPANY is not the same as a good STOCK.
Quality tells you if you'd want to own the business at all; valuation tells you
whether owning it *right now* is a good trade. We keep them separate, score
four pillars, then combine them with GATED logic (not a blended average):

  1. Quality     is a GATE.   Weak quality caps the ceiling (avoids value traps).
  2. Valuation   sets DIRECTION + SIZE.  Discount -> lean buy; premium -> lean sell.
  3. Narrative   can VETO.    Management spin caught by the numbers, or a real
                              thesis-breaker, overrides a cheap price.
  4. Backdrop    is TIMING.   And "neutral because we have no data" is NOT a
                              neutral reading -- it's a missing pillar. We mark
                              it UNKNOWN, weight it zero, and lower conviction.

Conviction then SCALES the call: low conviction pulls everything toward HOLD
and shrinks position size.  HOLD is the honest default when we don't know.

Finally we PERSONALISE: the same evidence can yield a BUY for a long-horizon
investor who buys the dip and a HOLD for someone who'd panic-sell -- because
that is what an advisor is for.

------------------------------------------------------------------------------
ARCHITECTURE  (why it's a hybrid, not pure-LLM and not pure-rules)
------------------------------------------------------------------------------
Layer A  --  deterministic evidence engine (this file, plain Python)
             Normalise -> reconcile/cross-check -> score 4 pillars ->
             detect conflicts -> compute conviction -> derive a gated base call.
             Reproducible. This is the "quant" half.

Layer B  --  Claude (on Bedrock) as the PM
             Takes the SCORED evidence + profile and writes the human call and
             rationale, but is CONSTRAINED by Layer A's gates: it literally
             cannot output BUY if the quality gate is shut, and cannot claim a
             market view when the backdrop pillar is missing. If Bedrock is
             unavailable, a deterministic rationale is generated instead, so the
             pipeline never hard-fails.

------------------------------------------------------------------------------
TRY IT NOW (no AWS needed -- runs on baked-in NVDA sample data):
    python advisor.py
------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

# Load the team .env (AWS creds + BEDROCK_MODEL_ID) the same way the sub-agents
# do, so the PM layer switches on automatically when it's configured.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# =============================================================================
# CONFIG  --  every tunable lives here so the methodology is auditable
# =============================================================================

# Conviction weights (must sum to ~1.0; a penalty for conflicts is subtracted).
W_DATA_COMPLETENESS = 0.40
W_PILLAR_AGREEMENT  = 0.35
W_AGENT_CONFIDENCE  = 0.25

CONVICTION_HIGH = 0.66      # >= this  -> HIGH
CONVICTION_MED  = 0.42      # >= this  -> MEDIUM, else LOW

# Valuation thresholds on implied upside to consensus fair value.
VAL_CHEAP_UPSIDE     =  0.20    # > +20% upside  -> attractive
VAL_EXPENSIVE_UPSIDE = -0.10    # < -10% upside  -> expensive

# Which model to use for the PM layer. Reuse whatever you already have enabled
# in Bedrock for the narrative agents. Leave blank to force the deterministic
# rationale (handy for offline testing / the video demo).
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "")
BEDROCK_REGION   = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
                    or os.environ.get("BEDROCK_REGION") or "us-east-1")

# Words in a risk line that hint at a THESIS-BREAKER rather than a watch-item.
# (Layer A uses this as a first pass; Layer B refines the judgment.)
SEVERE_RISK_MARKERS = [
    "foreclosed", "material adverse", "material and adverse", "going concern",
    "default", "delisting", "restated", "impairment of the business",
    "cannot compete", "effectively locked out", "loss of a major customer",
]

DISCLAIMER = (
    "This is automated research synthesis for education, not personalised "
    "financial advice. It is not a recommendation to buy or sell any security. "
    "Do your own research or speak to a licensed adviser before investing."
)


# =============================================================================
# SMALL PARSING HELPERS  --  the sub-agents speak slightly different dialects,
# so we translate defensively (same spirit as the coordinator's _get_section).
# =============================================================================

def _first_present(d: Any, keys: list[str], default: Any = None) -> Any:
    """Return the first key that exists and is non-empty, on a dict OR object."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, "", [], {}):
            return d[k]
        if not isinstance(d, dict) and getattr(d, k, None) not in (None, "", [], {}):
            return getattr(d, k)
    return default


def _money_to_float(text: Any) -> Optional[float]:
    """'$325.99' -> 325.99 ; '$215.94B' -> 215_940_000_000 ; 'N/A' -> None."""
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


def _pct_to_float(text: Any) -> Optional[float]:
    """'55.6%' -> 0.556 ; 0.556 -> 0.556 ; 'N/A' -> None."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    s = str(text).strip().replace("%", "")
    if s.upper() in ("", "N/A", "NA", "NONE", "-"):
        return None
    try:
        v = float(s)
        return v / 100.0 if abs(v) > 1.5 else v   # tolerate "55.6" or "0.556"
    except ValueError:
        return None


def _as_list(x: Any) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Pillar:
    """One of the four legs of the thesis."""
    name: str
    signal: str                       # 'strong' | 'neutral' | 'weak' | 'unknown'
    direction: Optional[int] = None   # +1 good, 0 neutral, -1 bad, None if unknown
    evidence: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataFlag:
    """A reconciliation note: a conflict we resolved, or a gap we can't fill."""
    kind: str                         # 'missing' | 'reconcilable' | 'contradiction' | 'info'
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# LAYER A -- STEP 1: NORMALISE each sub-agent into a common shape
# =============================================================================
# NOTE FOR THE TEAM: if your agents use different key names, this is the ONE
# place to adjust. Everything downstream reads from these normalised dicts.

def normalise_risk_research(out: Any) -> dict:
    """Numbers agent -> the fields Quality & Valuation need."""
    metrics = _first_present(out, ["metrics"], {}) or {}
    prof = metrics.get("profitability", {}) if isinstance(metrics, dict) else {}
    liq  = metrics.get("liquidity", {}) if isinstance(metrics, dict) else {}
    solv = metrics.get("solvency_and_leverage", {}) if isinstance(metrics, dict) else {}
    cf   = metrics.get("cash_flow_and_quality", {}) if isinstance(metrics, dict) else {}
    cons = _first_present(out, ["market_consensus"], {}) or {}

    return {
        "signal": str(_first_present(out, ["signal"], "")).lower(),
        "scorecard": _as_list(_first_present(out, ["scorecard"], [])),
        "net_margin":     _pct_to_float(prof.get("net_margin")),
        "gross_margin":   _pct_to_float(prof.get("gross_margin")),
        "operating_margin": _pct_to_float(prof.get("operating_margin")),
        "roe":            _pct_to_float(prof.get("roe")),
        "current_ratio":  liq.get("current_ratio"),
        "debt_to_equity": solv.get("debt_to_equity"),
        "operating_cash_flow": _money_to_float(cf.get("operating_cash_flow_formatted")),
        "free_cash_flow":      _money_to_float(cf.get("free_cash_flow_formatted")),
        "capex":               _money_to_float(cf.get("capital_expenditures_formatted")),
        "target_price":  _money_to_float(cons.get("target_price_mean")),
        "current_price": _money_to_float(cons.get("current_price")),
        "key_points": _as_list(_first_present(out, ["key_points"], [])),
        "_raw": out,
    }


def normalise_tenk(out: Any) -> dict:
    """10-K agent -> Financial-Health block + Management-Claims block.
    Accepts either {'financial_health':..., 'management_claims':...} or an
    object with those attributes."""
    fh = _first_present(out, ["financial_health", "financialHealth"], {}) or {}
    mc = _first_present(out, ["management_claims", "managementClaims"], {}) or {}

    def block(b: Any) -> dict:
        return {
            "signal": str(_first_present(b, ["signal"], "")).upper(),
            "summary": _first_present(b, ["summary"], "") or "",
            "key_points": _as_list(_first_present(b, ["key_points", "keyPoints"], [])),
            # The 10-K agent emits 'watch_items'; keep the older aliases as fallbacks.
            "watch_for": _as_list(_first_present(b, ["watch_items", "watch_for", "watchFor", "watch"], [])),
        }

    return {"financial_health": block(fh), "management_claims": block(mc), "_raw": out}


def normalise_market_sense(out: Any) -> dict:
    """Backdrop agent. Crucially, we detect 'neutral because we got no data'."""
    signal = str(_first_present(out, ["signal"], "")).upper()
    conf = _first_present(out, ["confidence"], None)
    conf = float(conf) if isinstance(conf, (int, float)) else None
    summary = str(_first_present(out, ["summary"], "") or "")

    # Is this a REAL neutral, or an empty one? Look at confidence + tell-tale prose.
    empty_markers = ["no macro", "failed", "missing api", "no company news",
                     "no current market data", "cannot assess", "not successfully retrieved"]
    looks_empty = (conf is not None and conf <= 0.10) or \
                  any(m in summary.lower() for m in empty_markers)

    return {
        "signal": signal,
        "confidence": conf,
        "summary": summary,
        "key_points": _as_list(_first_present(out, ["key_points", "keyPoints"], [])),
        "is_empty": bool(looks_empty),
        "_raw": out,
    }


# =============================================================================
# LAYER A -- STEP 2: RECONCILE / CROSS-CHECK (catch conflicts & gaps)
# =============================================================================

def reconcile(rr: dict, tk: dict, ms: dict) -> tuple[dict, list[DataFlag]]:
    """Fill gaps from the other agents where safe, and record every conflict.
    Returns (facts, flags). Text parsing is wrapped so a miss never crashes us."""
    flags: list[DataFlag] = []
    facts = dict(rr)   # start from the numbers agent, then patch

    fh_text = " ".join([tk["financial_health"]["summary"]] +
                       [str(x) for x in tk["financial_health"]["key_points"]])

    # (a) Gross margin was None in the numbers agent -> try to recover it from
    #     the 10-K narrative ("gross margin declined to 71.1% ...").
    if facts.get("gross_margin") is None:
        try:
            m = re.search(r"gross margin[^0-9]{0,40}(\d{1,2}\.\d)\s*%", fh_text, re.I)
            if m:
                facts["gross_margin"] = float(m.group(1)) / 100.0
                flags.append(DataFlag("reconcilable",
                    f"Gross margin was missing from the numbers agent; recovered "
                    f"{facts['gross_margin']*100:.1f}% from the 10-K narrative."))
            else:
                flags.append(DataFlag("missing", "Gross margin unavailable from any agent."))
        except Exception:
            flags.append(DataFlag("missing", "Gross margin unavailable from any agent."))

    # (b) Operating cash flow: numbers agent had N/A but the 10-K narrates it.
    if facts.get("operating_cash_flow") is None:
        try:
            m = re.search(r"operating cash flow[^$]{0,30}\$([\d.]+)\s*billion", fh_text, re.I)
            if m:
                facts["operating_cash_flow"] = float(m.group(1)) * 1e9
                flags.append(DataFlag("reconcilable",
                    f"Operating cash flow recovered from 10-K narrative "
                    f"(~${facts['operating_cash_flow']/1e9:.1f}B)."))
        except Exception:
            pass

    # (c) Implausible zeros: capex = $0 and FCF = N/A for a company like this is
    #     almost certainly a data gap, not reality. Flag it -- don't report a fake 0.
    if facts.get("capex") == 0 and facts.get("free_cash_flow") is None:
        flags.append(DataFlag("contradiction",
            "Capex reported as $0 and free cash flow as N/A -- implausible for an "
            "operating company. Treating both as MISSING, not real values."))
        facts["capex"] = None

    # (d) Valuation rests on a single anchor (a sell-side consensus target) with
    #     no independent multiple. Directionally usable, but cap the confidence.
    facts["valuation_single_anchor"] = facts.get("target_price") is not None
    if facts["valuation_single_anchor"]:
        flags.append(DataFlag("info",
            "Valuation uses only the analyst consensus target (no independent P/E "
            "or growth-adjusted multiple). Sell-side targets herd and lag, so we "
            "treat the implied upside as directional, not precise."))

    # (e) The backdrop pillar is dead -- state it plainly.
    if ms.get("is_empty"):
        flags.append(DataFlag("missing",
            "Market Sense returned NEUTRAL with no usable data (macro feeds "
            "unavailable). The backdrop pillar is UNASSESSED, not benign."))

    return facts, flags


# =============================================================================
# LAYER A -- STEP 3: SCORE THE FOUR PILLARS
# =============================================================================

def score_quality(facts: dict, tk: dict) -> Pillar:
    """Is this a durable, well-run business? Reuse the numbers agent's own
    PASS/FAIL benchmarks, then overlay the TREND from the 10-K."""
    scorecard = facts.get("scorecard", [])
    passes = sum(1 for r in scorecard if str(r.get("status", "")).upper() == "PASS")
    total = len(scorecard)
    evidence, caveats = [], []

    for r in scorecard:
        evidence.append(f"{r.get('metric')}: {r.get('actual')} vs {r.get('benchmark')} "
                        f"[{r.get('status')}]")

    # Balance-sheet safety gate: a broken balance sheet caps quality regardless.
    dte, cur = facts.get("debt_to_equity"), facts.get("current_ratio")
    balance_broken = (isinstance(dte, (int, float)) and dte > 2.0) or \
                     (isinstance(cur, (int, float)) and cur < 1.0)

    if total and passes == total and not balance_broken:
        signal, direction = "strong", +1
    elif total and passes / total >= 0.5 and not balance_broken:
        signal, direction = "neutral", 0
    elif total:
        signal, direction = "weak", -1
    else:
        signal, direction = "unknown", None

    # TREND overlay from the 10-K financial-health block -- level can be elite
    # while direction is deteriorating; a professional records both.
    fh = tk["financial_health"]
    fh_text = (fh["summary"] + " " + " ".join(str(x) for x in fh["watch_for"])).lower()
    for phrase, note in [
        ("margin", "Margin compression flagged in 10-K -- level is high but the trend has pressure."),
        ("provision", "Inventory/credit provisions rising -- watch earnings quality."),
        ("charge", "One-off charge dented reported profitability this period."),
    ]:
        if phrase in fh_text and note not in caveats:
            caveats.append(note)

    if fh.get("signal") == "NEGATIVE" and signal == "strong":
        signal, direction = "neutral", 0
        caveats.append("Downgraded from strong: 10-K financial-health signal is negative.")

    return Pillar("quality", signal, direction, evidence, caveats)


def score_valuation(facts: dict) -> Pillar:
    """What am I paying vs what I'm getting? Implied upside to fair value."""
    tgt, px = facts.get("target_price"), facts.get("current_price")
    if not tgt or not px or px <= 0:
        return Pillar("valuation", "unknown", None,
                      ["No usable price or fair-value anchor."], [])

    upside = (tgt - px) / px
    ev = [f"Price ${px:,.2f} vs consensus fair value ${tgt:,.2f} "
          f"-> implied upside {upside*100:+.1f}%."]
    caveats = []
    if facts.get("valuation_single_anchor"):
        caveats.append("Single anchor (consensus target only) -> conviction capped at moderate.")

    if upside > VAL_CHEAP_UPSIDE:
        signal, direction = "strong", +1
    elif upside < VAL_EXPENSIVE_UPSIDE:
        signal, direction = "weak", -1
    else:
        signal, direction = "neutral", 0

    p = Pillar("valuation", signal, direction, ev, caveats)
    p_dict_upside = upside
    p.evidence.append(f"__upside__={p_dict_upside:.4f}")   # machine-readable tag
    return p


def _classify_risks(lines: list[str]) -> tuple[list[str], list[str]]:
    """Split risk lines into (thesis_breakers, watch_items). Deterministic first
    pass on severity keywords; Layer B refines. Concentration is treated as
    thesis-relevant when a single customer is a large share of revenue."""
    breakers, watch = [], []
    for ln in lines:
        low = str(ln).lower()
        severe = any(mk in low for mk in SEVERE_RISK_MARKERS)
        # customer concentration with a big % is thesis-relevant
        conc = "concentration" in low or re.search(r"\b(2[0-9]|[3-9][0-9])%\b.*customer", low) \
               or re.search(r"customer.*\b(2[0-9]|[3-9][0-9])%\b", low)
        (breakers if (severe or conc) else watch).append(str(ln))
    return breakers, watch


def score_narrative(tk: dict, facts: dict) -> Pillar:
    """Can I trust the story, and what could break it? Runs the claims-vs-reality
    cross-check: is management's tone consistent with the hard numbers?"""
    mc, fh = tk["management_claims"], tk["financial_health"]
    mc_sig, fh_sig = mc.get("signal", ""), fh.get("signal", "")

    rank = {"POSITIVE": 1, "MIXED": 0, "NEUTRAL": 0, "NEGATIVE": -1}
    mc_r, fh_r = rank.get(mc_sig, 0), rank.get(fh_sig, 0)

    evidence, caveats = [], []
    # CLAIMS-VS-REALITY: the credibility check that a plain summariser can't do.
    if mc_r - fh_r >= 2:
        caveats.append("Optimism gap: management's tone is markedly more upbeat than "
                       "the financials support -- treat guidance with caution.")
        credibility = -1
    elif mc_r > fh_r:
        caveats.append("Management slightly ahead of the numbers -- minor optimism.")
        credibility = 0
    else:
        evidence.append("Management narrative is consistent with the financials "
                        "(risks acknowledged, not spun) -- a credibility positive.")
        credibility = +1

    breakers, watch = _classify_risks(mc.get("watch_for", []) + fh.get("watch_for", []))
    for b in breakers:
        evidence.append(f"THESIS RISK: {b}")

    # Map management signal to the pillar, then let credibility & breakers adjust.
    base_dir = mc_r
    if breakers and base_dir >= 0:
        base_dir = 0           # real thesis risks pull an upbeat story to neutral
        caveats.append("Thesis-level risks present -> narrative held at neutral despite upbeat tone.")
    if credibility < 0:
        base_dir = min(base_dir, -1 if base_dir <= 0 else 0)

    signal = {1: "strong", 0: "neutral", -1: "weak"}.get(
        max(-1, min(1, base_dir)), "neutral")
    p = Pillar("narrative", signal, base_dir, evidence, caveats)
    p.evidence.append(f"__breakers__={json.dumps(breakers)}")
    p.evidence.append(f"__watch__={json.dumps(watch)}")
    return p


def score_backdrop(ms: dict) -> Pillar:
    """Macro/sentiment timing. The discipline: 'no data' -> UNKNOWN, not neutral."""
    if ms.get("is_empty"):
        return Pillar("backdrop", "unknown", None,
                      ["Market Sense returned no usable data; backdrop unassessed."],
                      ["Weighted at zero. Conviction lowered accordingly."])
    sig = ms.get("signal", "")
    direction = {"POSITIVE": 1, "SUPPORTIVE": 1, "NEUTRAL": 0,
                 "NEGATIVE": -1, "HEADWIND": -1}.get(sig, 0)
    signal = {1: "strong", 0: "neutral", -1: "weak"}[max(-1, min(1, direction))]
    return Pillar("backdrop", signal, direction,
                  [ms.get("summary", "")[:200]], [])


# =============================================================================
# LAYER A -- STEP 4: CONVICTION  (a real number, decomposed & inspectable)
# =============================================================================

def compute_conviction(pillars: dict[str, Pillar],
                       rr: dict, tk: dict, ms: dict,
                       flags: list[DataFlag], facts: dict) -> dict:
    """conviction = w1*data_completeness + w2*pillar_agreement
                   + w3*mean_agent_confidence  -  penalty(conflicts)."""

    # -- data_completeness: how much of what we WANTED did we actually get? --
    assessable = [p for p in pillars.values() if p.signal != "unknown"]
    pillar_coverage = len(assessable) / 4.0
    key_fields = ["net_margin", "roe", "current_ratio", "debt_to_equity",
                  "gross_margin", "operating_cash_flow", "target_price", "current_price"]
    present = sum(1 for k in key_fields if facts.get(k) is not None)
    field_coverage = present / len(key_fields)
    data_completeness = 0.5 * pillar_coverage + 0.5 * field_coverage

    # -- pillar_agreement: do the assessable pillars point the same way? --
    dirs = [p.direction for p in assessable if p.direction is not None]
    if len(dirs) <= 1:
        pillar_agreement = 1.0
    else:
        # std over [-1,1] has a theoretical max of 1.0; agreement = 1 - std.
        pillar_agreement = max(0.0, 1.0 - statistics.pstdev(dirs))

    # -- mean_agent_confidence: only over agents that ACTUALLY spoke --
    confs = []
    # Numbers agent has no self-reported confidence -> infer from decisiveness.
    if rr.get("scorecard"):
        pr = sum(1 for r in rr["scorecard"] if str(r.get("status")).upper() == "PASS") \
             / len(rr["scorecard"])
        confs.append(0.5 + 0.4 * pr)                         # 0.5..0.9
    # 10-K agent -> infer from its two signals.
    sig_conf = {"POSITIVE": 0.75, "NEGATIVE": 0.75, "MIXED": 0.5, "NEUTRAL": 0.5}
    tk_confs = [sig_conf.get(tk[b]["signal"], 0.5) for b in ("financial_health", "management_claims")
                if tk[b]["signal"]]
    if tk_confs:
        confs.append(sum(tk_confs) / len(tk_confs))
    # Market Sense -> include ONLY if it produced usable data (else completeness
    # already carries the penalty; counting a 0.0 would double-punish).
    if not ms.get("is_empty") and ms.get("confidence") is not None:
        confs.append(ms["confidence"])
    mean_agent_confidence = sum(confs) / len(confs) if confs else 0.4

    # -- penalty: contradictions hurt more than reconcilable gaps --
    penalty = 0.0
    for f in flags:
        penalty += {"contradiction": 0.06, "missing": 0.03,
                    "reconcilable": 0.01, "info": 0.0}.get(f.kind, 0.0)
    penalty = min(penalty, 0.20)

    raw = (W_DATA_COMPLETENESS * data_completeness
           + W_PILLAR_AGREEMENT * pillar_agreement
           + W_AGENT_CONFIDENCE * mean_agent_confidence
           - penalty)
    score = max(0.0, min(1.0, raw))
    label = "HIGH" if score >= CONVICTION_HIGH else "MEDIUM" if score >= CONVICTION_MED else "LOW"

    return {
        "score": round(score, 3),
        "label": label,
        "components": {
            "data_completeness": round(data_completeness, 3),
            "pillar_agreement": round(pillar_agreement, 3),
            "mean_agent_confidence": round(mean_agent_confidence, 3),
            "conflict_penalty": round(penalty, 3),
        },
    }


# =============================================================================
# LAYER A -- STEP 5: GATED DECISION + PERSONALISATION
# =============================================================================

def _pillar_upside(valuation: Pillar) -> Optional[float]:
    for e in valuation.evidence:
        if e.startswith("__upside__="):
            return float(e.split("=", 1)[1])
    return None


def base_recommendation(pillars: dict[str, Pillar], conviction: dict) -> dict:
    """The profile-NEUTRAL 'house view', built with gated logic."""
    q, v, n = pillars["quality"], pillars["valuation"], pillars["narrative"]
    reasons = []

    # QUALITY GATE ------------------------------------------------------------
    if q.signal == "weak":
        reasons.append("Quality gate is shut (weak business fundamentals) -> no BUY.")
        call = "SELL" if v.signal == "weak" else "HOLD"
        return _finalise_base(call, conviction, reasons)

    # NARRATIVE VETO ----------------------------------------------------------
    if n.signal == "weak":
        reasons.append("Narrative veto: management credibility or a thesis-breaker "
                       "overrides the price signal.")
        return _finalise_base("HOLD", conviction, reasons)

    # VALUATION SETS DIRECTION -------------------------------------------------
    if v.signal == "strong":                     # cheap
        reasons.append("Business clears the quality gate and trades at a discount to "
                       "fair value with no thesis-breaker -> lean BUY.")
        call = "BUY"
    elif v.signal == "weak":                     # expensive
        reasons.append("Good business but priced above fair value -> lean SELL/trim.")
        call = "SELL"
    else:
        reasons.append("Fairly valued, quality intact -> HOLD.")
        call = "HOLD"

    return _finalise_base(call, conviction, reasons)


def _finalise_base(call: str, conviction: dict, reasons: list[str]) -> dict:
    # CONVICTION SCALES: low conviction pulls an active call toward HOLD.
    if conviction["label"] == "LOW" and call in ("BUY", "SELL"):
        reasons.append(f"Conviction is LOW -> softening {call} toward HOLD "
                       f"(a starter position at most).")
        call = "HOLD"
    return {"recommendation": call, "reasons": reasons}


def personalise(base: dict, pillars: dict[str, Pillar], conviction: dict,
                profile: dict) -> dict:
    """Turn the house view into a call for THIS investor. Same evidence can yield
    different calls -- that's the point of an adviser."""
    call = base["recommendation"]
    notes: list[str] = []
    n = pillars["narrative"]

    breakers = []
    for e in n.evidence:
        if e.startswith("__breakers__="):
            breakers = json.loads(e.split("=", 1)[1])
    elevated_risk = bool(breakers) or n.signal in ("neutral", "weak")

    horizon   = profile.get("horizon", "long")
    reaction  = profile.get("drawdown_reaction", "hold")
    size_pref = profile.get("position_size", "small")
    goal      = profile.get("goal", "growth")

    # Recommended position size (before honouring the user's own preference).
    if conviction["label"] == "HIGH" and n.signal == "strong":
        rec_size = "meaningful"
    elif elevated_risk or conviction["label"] == "MEDIUM":
        rec_size = "modest / starter"
    else:
        rec_size = "small"

    # HORIZON: a thesis that needs years, held by someone with a <1y horizon.
    if horizon == "short" and call == "BUY" and elevated_risk:
        call = "HOLD"
        notes.append("Your horizon is under a year, but this thesis needs years to play "
                     "out and can swing hard in between -> HOLD rather than buy now.")

    # DRAWDOWN BEHAVIOUR: the single most useful profile input.
    if reaction == "sell" and elevated_risk:
        rec_size = "small / starter"
        if call == "BUY":
            call = "HOLD"
            notes.append("You'd likely sell into a 25% drop, and this name can deliver "
                         "exactly that -> holding off protects you from buying into a "
                         "decline you'd bail on.")
    elif reaction == "buy_more" and horizon in ("medium", "long") and goal == "growth":
        notes.append("You add to quality on weakness and hold for years -> a drawdown is "
                     "an opportunity for you, so the BUY stands (sized with discipline).")

    # GOAL FIT.
    if goal == "income":
        notes.append("Heads-up: this is a growth name with negligible dividend -- a weak "
                     "fit for an income mandate regardless of the call.")
    elif goal == "understand":
        call = "HOLD / INFORM"
        notes.append("You asked to understand the company, not to trade it -- here is the "
                     "read, with no action implied.")

    # SIZE PREFERENCE vs prudent size.
    if size_pref == "large" and (elevated_risk or conviction["label"] != "HIGH"):
        notes.append(f"You'd size this large; given single-name concentration risk and "
                     f"{conviction['label']} conviction, consider capping it to a "
                     f"{rec_size} position instead.")

    return {"recommendation": call, "position_size": rec_size, "notes": notes}


# =============================================================================
# LAYER B -- Claude (Bedrock) as the PM, with a deterministic fallback
# =============================================================================

def _build_pm_prompt(evidence: dict, profile: dict) -> tuple[str, str]:
    system = (
        "You are the senior portfolio manager on an equity research desk. Three "
        "analysts have handed you scored evidence about one US-listed stock, and a "
        "client profile. Write the final call and a clear, honest rationale.\n\n"
        "HARD RULES (the desk's risk controls -- you may not break them):\n"
        "1. If quality.signal == 'weak', you may NOT recommend BUY.\n"
        "2. If narrative.signal == 'weak', a cheap price does not justify BUY; hold or avoid.\n"
        "3. If backdrop.signal == 'unknown', do NOT claim any view on the macro/market "
        "backdrop; explicitly say it is unassessed and that this lowers conviction.\n"
        "4. Never invent numbers. Use only the evidence given. Where a field is missing, "
        "say so.\n"
        "5. This is research synthesis, not personalised financial advice. No hype.\n\n"
        "Return STRICT JSON only (no prose, no markdown) with keys: thesis (one "
        "sentence), rationale (2-4 sentences), recommendation (BUY|HOLD|SELL), "
        "conviction (HIGH|MEDIUM|LOW), thesis_breakers (list), watch_items (list), "
        "personalization (one sentence tying the call to THIS client), "
        "confidence_rationale (what we know, what we don't, how to raise conviction)."
    )
    user = ("EVIDENCE:\n" + json.dumps(evidence, indent=2, default=str) +
            "\n\nCLIENT PROFILE:\n" + json.dumps(profile, indent=2))
    return system, user


def _invoke_bedrock(system: str, user: str) -> Optional[str]:
    """Call Claude on Bedrock. Returns text, or None if unavailable."""
    if not BEDROCK_MODEL_ID:
        return None
    try:
        import boto3   # guarded: absent in offline/test envs
        client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1200,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        resp = client.invoke_model(modelId=BEDROCK_MODEL_ID, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        return payload["content"][0]["text"]
    except Exception as e:
        print(f"  (Bedrock unavailable -- {type(e).__name__}: using deterministic rationale.)")
        return None


def _parse_pm_json(text: str) -> Optional[dict]:
    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        return json.loads(cleaned[start:end + 1])
    except Exception:
        return None


def _deterministic_rationale(evidence: dict, base: dict, personal: dict,
                             conviction: dict) -> dict:
    """Templated PM write-up when the LLM isn't available -- says the same things."""
    p = evidence["pillars"]
    up = evidence.get("implied_upside", "n/a")
    thesis = (f"{evidence['ticker']}: "
              f"{'high-quality' if p['quality']['signal']=='strong' else p['quality']['signal']} "
              f"business, valuation {p['valuation']['signal']} ({up} to consensus), "
              f"narrative {p['narrative']['signal']}, backdrop {p['backdrop']['signal']}.")
    rationale = " ".join(base["reasons"] + personal["notes"]) or "See pillar detail."
    conf = (f"Conviction {conviction['label']} ({conviction['score']}). Driven by "
            f"data completeness {conviction['components']['data_completeness']}, "
            f"pillar agreement {conviction['components']['pillar_agreement']}, "
            f"agent confidence {conviction['components']['mean_agent_confidence']}. "
            f"Biggest lever to raise it: restore the Market Sense (backdrop) pillar.")
    return {
        "thesis": thesis,
        "rationale": rationale,
        "recommendation": personal["recommendation"],
        "conviction": conviction["label"],
        "thesis_breakers": evidence["risks"]["thesis_breakers"],
        "watch_items": evidence["risks"]["watch_items"],
        "personalization": " ".join(personal["notes"]) or "No profile-specific adjustment.",
        "confidence_rationale": conf,
    }


def _enforce_gates(pm: dict, pillars: dict[str, Pillar]) -> dict:
    """The LLM proposes; the gates dispose. Override any answer that violates
    Layer A's risk controls, and record that we did."""
    reco = str(pm.get("recommendation", "HOLD")).upper()
    if reco not in ("BUY", "HOLD", "SELL"):
        reco = "HOLD"
    if pillars["quality"].signal == "weak" and reco == "BUY":
        pm["gate_override"] = "LLM said BUY but quality gate is shut -> forced HOLD."
        reco = "HOLD"
    if pillars["narrative"].signal == "weak" and reco == "BUY":
        pm["gate_override"] = "LLM said BUY but narrative veto is active -> forced HOLD."
        reco = "HOLD"
    pm["recommendation"] = reco
    return pm


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_advisor(risk_research_out: Any,
                tenk_out: Any,
                market_sense_out: Any,
                profile: dict,
                ticker: str = "",
                llm: Optional[Callable[[str, str], str]] = None) -> dict:
    """The advisor's whole job in one call.

    Pass the three sub-agent outputs, the user profile (from the coordinator's
    context['profile']), and optionally your own `llm(system, user)->str` wrapper
    (e.g. reuse claude_helper.py). With no llm and no BEDROCK_MODEL_ID, it runs
    fully on the deterministic engine."""

    # ---- Layer A ----
    rr = normalise_risk_research(risk_research_out)
    tk = normalise_tenk(tenk_out)
    ms = normalise_market_sense(market_sense_out)
    ticker = ticker or _first_present(risk_research_out, ["ticker"], "") or ""

    facts, flags = reconcile(rr, tk, ms)

    pillars = {
        "quality":   score_quality(facts, tk),
        "valuation": score_valuation(facts),
        "narrative": score_narrative(tk, facts),
        "backdrop":  score_backdrop(ms),
    }
    conviction = compute_conviction(pillars, rr, tk, ms, flags, facts)
    base = base_recommendation(pillars, conviction)
    personal = personalise(base, pillars, conviction, profile)

    # Pull machine tags back out of the valuation/narrative pillars for the schema.
    upside = _pillar_upside(pillars["valuation"])
    breakers, watch = [], []
    for e in pillars["narrative"].evidence:
        if e.startswith("__breakers__="):
            breakers = json.loads(e.split("=", 1)[1])
        if e.startswith("__watch__="):
            watch = json.loads(e.split("=", 1)[1])

    # Clean the internal __tags__ out of the evidence we expose.
    def _clean(p: Pillar) -> dict:
        d = p.to_dict()
        d["evidence"] = [e for e in d["evidence"] if not e.startswith("__")]
        return d

    supports = []
    for name in ("quality", "valuation", "narrative", "backdrop"):
        if pillars[name].direction == 1:
            supports.extend(pillars[name].evidence)
    supports = [s for s in supports if not s.startswith("__")]

    evidence = {
        "agent": "Advisor Agent",
        "ticker": ticker,
        "implied_upside": f"{upside*100:+.1f}%" if upside is not None else "n/a",
        "pillars": {k: _clean(v) for k, v in pillars.items()},
        "conviction": conviction,
        "supports": supports,
        "risks": {"thesis_breakers": breakers, "watch_items": watch},
        "data_flags": [f.to_dict() for f in flags],
    }

    # ---- Layer B ----
    system, user = _build_pm_prompt(evidence, profile)
    text = llm(system, user) if llm else _invoke_bedrock(system, user)
    pm = _parse_pm_json(text) if text else None
    source = "bedrock/llm" if pm else "deterministic_fallback"
    if not pm:
        pm = _deterministic_rationale(evidence, base, personal, conviction)
    pm = _enforce_gates(pm, pillars)

    # ---- Assemble the final output object the dashboard reads ----
    return {
        "agent": "Advisor Agent",
        "ticker": ticker,
        "recommendation": personal["recommendation"],   # personalised, gated
        "base_recommendation": base["recommendation"],   # profile-neutral house view
        "llm_recommendation": pm.get("recommendation"),  # what the PM layer said
        "conviction": conviction,
        "implied_upside": evidence["implied_upside"],
        "thesis": pm.get("thesis", ""),
        "rationale": pm.get("rationale", ""),
        "pillars": evidence["pillars"],
        "supports": supports,
        "risks": {
            "thesis_breakers": pm.get("thesis_breakers", breakers),
            "watch_items": pm.get("watch_items", watch),
        },
        "data_flags": evidence["data_flags"],
        "position_size": personal["position_size"],
        "personalization": {
            "notes": personal["notes"],
            "summary": pm.get("personalization", ""),
        },
        "confidence_rationale": pm.get("confidence_rationale", ""),
        "gate_override": pm.get("gate_override"),
        "rationale_source": source,
        "disclaimer": DISCLAIMER,
    }


# =============================================================================
# CONVENIENCE ENTRY POINTS  --  wire straight onto the coordinator
# =============================================================================

def advise(context: dict,
           risk_research_out: Any,
           tenk_out: Any,
           market_sense_out: Any,
           llm: Optional[Callable[[str, str], str]] = None) -> dict:
    """Same as run_advisor(), but accepts the coordinator's `context` object
    directly -- it pulls the profile and ticker out of it for you.

        context = run_coordinator(ticker)
        advice  = advise(context, rr_out, tk_out, ms_out)
    """
    profile = context.get("profile", {}) or {}
    ticker = (context.get("company", {}) or {}).get("ticker", "") \
        or _first_present(risk_research_out, ["ticker"], "")
    return run_advisor(risk_research_out, tenk_out, market_sense_out,
                       profile=profile, ticker=ticker, llm=llm)


def run_pipeline(ticker: str,
                 profile_answers: Optional[dict] = None,
                 llm: Optional[Callable[[str, str], str]] = None,
                 risk_research_fn: Optional[Callable[[dict], Any]] = None,
                 tenk_fn: Optional[Callable[[dict], Any]] = None,
                 market_sense_fn: Optional[Callable[[dict], Any]] = None) -> dict:
    """The whole flow in one call: coordinator -> 3 sub-agents -> advisor.

    By default it calls the team's real sub-agents. `profile_answers` is
    optional: pass the web-form answers to skip the terminal questions, or leave
    it None to be asked. The *_fn params exist only so tests can inject stubs.

        out = run_pipeline("NVDA")                       # live
        out = run_pipeline("NVDA", profile_answers={...}) # web form, no prompts
    """
    from coordinator import run_coordinator          # the team's coordinator

    # Bind the real sub-agents (their actual module + function names).
    if risk_research_fn is None:
        from risk_research_agent import run_risk_research as risk_research_fn
    if tenk_fn is None:
        from tenk_agent import analyze_10k_report as tenk_fn
    if market_sense_fn is None:
        from market_sense import run_market_sense_agent as market_sense_fn

    context = run_coordinator(ticker, profile_answers=profile_answers)

    # --- Ticker bridge --------------------------------------------------------
    # The Risk & Research agent reads context["ticker"] (top level) for its
    # yfinance price/target lookup, but the coordinator stores the ticker at
    # context["company"]["ticker"]. Without this line it silently defaults to
    # "NVDA" and fetches the wrong company's valuation. Bridge them once here.
    context.setdefault("ticker", (context.get("company") or {}).get("ticker", ticker))

    # The three specialists read from the SAME shared context object.
    rr_out = risk_research_fn(context)
    tk_out = tenk_fn(context)
    ms_out = market_sense_fn(context)

    return advise(context, rr_out, tk_out, ms_out, llm=llm)


# =============================================================================
# PRETTY PRINT  (terminal view for testing & the demo video)
# =============================================================================

def preview_advice(out: dict) -> None:
    line = "=" * 68
    print("\n" + line)
    print(f"  ADVISOR VERDICT  --  {out['ticker']}")
    print(line)
    print(f"  Recommendation (for this investor): {out['recommendation']}")
    print(f"  House view (profile-neutral):       {out['base_recommendation']}")
    print(f"  Conviction: {out['conviction']['label']} ({out['conviction']['score']})"
          f"   |   Implied upside: {out['implied_upside']}")
    print(f"  Suggested position size: {out['position_size']}")
    if out.get("gate_override"):
        print(f"  ! Gate override: {out['gate_override']}")
    print(f"\n  Thesis: {out['thesis']}")
    print(f"\n  Rationale: {out['rationale']}")

    print("\n  Four pillars:")
    for name, p in out["pillars"].items():
        print(f"    - {name:9s} {p['signal'].upper():8s}")
        for c in p.get("caveats", []):
            print(f"        caveat: {c}")

    print("\n  Conviction breakdown:")
    for k, v in out["conviction"]["components"].items():
        print(f"    - {k:24s} {v}")

    if out["risks"]["thesis_breakers"]:
        print("\n  Thesis-level risks:")
        for r in out["risks"]["thesis_breakers"]:
            print(f"    ! {r}")
    if out["risks"]["watch_items"]:
        print("\n  Watch-items:")
        for r in out["risks"]["watch_items"]:
            print(f"    - {r}")

    print("\n  Data flags (reconciliations & gaps):")
    for f in out["data_flags"]:
        print(f"    [{f['kind']}] {f['message']}")

    if out["personalization"]["notes"]:
        print("\n  Why this call for you:")
        for nnote in out["personalization"]["notes"]:
            print(f"    - {nnote}")

    print(f"\n  Confidence: {out['confidence_rationale']}")
    print(f"\n  Rationale source: {out['rationale_source']}")
    print(f"\n  {out['disclaimer']}")
    print(line + "\n")


# =============================================================================
# DEMO  --  runs on the real NVDA sub-agent outputs, no AWS required
# =============================================================================

def _demo_inputs():
    """Reconstructed from the three sub-agents' actual NVDA runs."""
    risk_research = {
        "agent": "Risk & Research Agent", "ticker": "NVDA", "signal": "positive",
        "scorecard": [
            {"category": "Profitability", "metric": "Net Profit Margin",
             "benchmark": ">= 15.0%", "actual": "55.6%", "status": "PASS"},
            {"category": "Profitability", "metric": "Return on Equity (ROE)",
             "benchmark": ">= 15.0%", "actual": "76.33%", "status": "PASS"},
            {"category": "Liquidity", "metric": "Current Ratio",
             "benchmark": ">= 1.50x", "actual": "3.91x", "status": "PASS"},
            {"category": "Solvency", "metric": "Debt-to-Equity",
             "benchmark": "<= 1.50x", "actual": "0.31x", "status": "PASS"},
        ],
        "key_points": [
            "Profitability: 55.6% net margin and 76.33% ROE.",
            "Balance sheet: current ratio 3.91x with $93.44B net working capital.",
        ],
        "metrics": {
            "profitability": {"gross_margin": None, "operating_margin": 0.6038,
                              "net_margin": 0.556, "roe": 0.7633, "roa": 0.5806},
            "liquidity": {"current_ratio": 3.9053, "quick_ratio": 3.2398},
            "solvency_and_leverage": {"debt_to_equity": 0.3148, "debt_to_assets": 0.2394},
            "cash_flow_and_quality": {"operating_cash_flow_formatted": "N/A",
                                      "capital_expenditures_formatted": "$0.00",
                                      "free_cash_flow_formatted": "N/A"},
            "raw_dollars": {"revenue": "$215.94B", "net_income": "$120.07B",
                            "cash_and_equivalents": "$10.61B"},
        },
        "market_consensus": {"target_price_mean": "$325.99", "current_price": "$228.45",
                             "expected_eps_consensus": "N/A"},
    }

    tenk = {
        "financial_health": {
            "signal": "POSITIVE",
            "summary": ("NVIDIA delivered exceptional fiscal 2026 performance with revenue "
                        "up 65% to $215.9 billion and net income up 65% to $120.1 billion, "
                        "driven by AI and data center demand. However, gross margin "
                        "compressed to 71.1% from 75.0% due to a $4.5 billion H20 inventory "
                        "charge and a shift to lower-margin full-scale data center solutions."),
            "key_points": [
                "Revenue surged 65% to $215.9 billion; Data Center up 68%.",
                "Net income grew 65% to $120.1 billion; diluted EPS up 67% to $4.90.",
                "Operating cash flow reached $102.7 billion (up 60%).",
            ],
            "watch_for": [
                "Revenue concentration: one customer is 22% of revenue and another 14%.",
                "Geopolitical/regulatory: $4.5B H20 charge from China export restrictions.",
                "Inventory provisions elevated at $7.2B (vs $3.7B prior year).",
            ],
        },
        "management_claims": {
            "signal": "MIXED",
            "summary": ("Exceptional results (65% revenue growth) but significant headwinds "
                        "from U.S. export controls to China that cost $4.5 billion in H20 "
                        "charges and effectively locked the company out of China's data "
                        "center market."),
            "key_points": [
                "Revenue up 65% to $215.9 billion; Data Center up 68%.",
                "Gross margin declined to 71.1% from 75.0%.",
            ],
            "watch_for": [
                "Management states they are 'effectively foreclosed from competing in "
                "China's data center computing market' with 'material and adverse impact'.",
                "Gross margin pressure: 2.6% unfavorable impact from inventory provisions.",
                "Supply chain concentration: manufacturing 'mainly concentrated in Asia'.",
            ],
        },
    }

    market_sense = {
        "company": "NVIDIA CORP", "ticker": "NVDA", "industry": "",
        "signal": "NEUTRAL", "confidence": 0.0,
        "summary": ("No macroeconomic, industry, or company-specific data was successfully "
                    "retrieved. The analysis cannot assess current market conditions for NVIDIA."),
        "key_points": [
            "All macroeconomic data sources failed due to missing API credentials.",
            "No company or industry news was provided.",
        ],
    }
    return risk_research, tenk, market_sense


def _run_demo() -> None:
    """Offline sanity check on the baked-in NVDA data -- no coordinator, no AWS.
    Shows the SAME evidence producing two different calls for two investors."""
    rr_out, tk_out, ms_out = _demo_inputs()

    # Profile A: long-horizon growth investor who buys the dip.
    profile_a = {"experience": "some", "drawdown_reaction": "buy_more",
                 "horizon": "long", "ownership": "considering",
                 "position_size": "medium", "goal": "growth"}
    # Profile B: short-horizon investor who panic-sells.
    profile_b = {"experience": "new", "drawdown_reaction": "sell",
                 "horizon": "short", "ownership": "none",
                 "position_size": "large", "goal": "growth"}

    print("\n########## [DEMO] SAME EVIDENCE, INVESTOR A (long / buys the dip) ##########")
    preview_advice(run_advisor(rr_out, tk_out, ms_out, profile_a, ticker="NVDA"))
    print("\n########## [DEMO] SAME EVIDENCE, INVESTOR B (short / panic-sells) ##########")
    preview_advice(run_advisor(rr_out, tk_out, ms_out, profile_b, ticker="NVDA"))


if __name__ == "__main__":
    import sys
    print("Advisor agent  --  The Overwhelmed Everyday Investor")

    # Offline sample:  python advisor.py --demo
    if "--demo" in sys.argv:
        _run_demo()
        sys.exit(0)

    # -------------------------------------------------------------------------
    # REAL PIPELINE  --  live ticker straight from the coordinator, then the
    # three real sub-agents, then the advisor. run_pipeline() wires them all.
    # -------------------------------------------------------------------------
    try:
        from coordinator import ask_ticker_cli
        ticker = ask_ticker_cli()                 # asks the user (web form: pass answers)
        advice = run_pipeline(ticker)
        preview_advice(advice)
    except ImportError as e:
        raise SystemExit(
            f"\nCouldn't import a pipeline module ({e}).\n"
            "Make sure coordinator.py, risk_research_agent.py, tenk_agent.py and "
            "market_sense.py sit in the same folder as advisor.py.\n\n"
            "To run the offline NVDA sample instead:  python advisor.py --demo\n"
        )
