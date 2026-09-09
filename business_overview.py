"""
business_overview.py  --  plain-English "what this company actually does"

Feeds the "Business & 10-K" tab (frontend item 7). It reads ONLY the 10-K's
Item 1 "Business" text and asks Claude to summarise it for a layperson, with a
hard rule: state nothing that isn't in the filing. If Bedrock is unavailable or
the section is empty, it degrades gracefully to {} and the UI shows a fallback
line -- consistent with the rest of the pipeline's graceful-degradation policy.

Output shape (all keys optional; UI tolerates missing ones):
    {
      "summary":  "2-3 plain sentences on what the company sells and how it earns",
      "bullets":  ["short fact from the filing", ...],   # <= 4
      "segments": [{"name": "...", "pct": 88}, ...]       # ONLY if stated in text
    }
"""

from __future__ import annotations

import os
import re
import json
from typing import Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "")
BEDROCK_REGION = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
                  or os.environ.get("BEDROCK_REGION") or "us-east-1")

# Cap how much filing text we send (Item 1 can be huge; the first chunk is
# where the "what we do / segments" description lives).
MAX_CHARS = 12000

_SYSTEM = (
    "You explain public companies to a complete beginner in plain English. "
    "You are given the 'Business' section of a company's own 10-K annual report. "
    "STRICT RULE: use ONLY facts stated in that text. Do not add outside knowledge, "
    "figures, competitors, or segments that are not in the text. If the text does "
    "not state a revenue split by segment, return an empty 'segments' list. "
    "No hype, no adjectives like 'leading' unless the filing uses them. "
    "Return ONLY JSON, no markdown, no preamble."
)

_USER_TMPL = """Company: {company}

Here is the Business section from the 10-K:
\"\"\"
{text}
\"\"\"

Return JSON exactly like this:
{{
  "summary": "2-3 short sentences: what it sells and how it makes money, in plain words a beginner understands",
  "bullets": ["<=4 short factual points from the text, plain language"],
  "segments": [{{"name": "segment name as written", "pct": <integer 0-100 ONLY if the text gives a revenue share; otherwise omit this list entirely>}}]
}}"""


def _invoke_bedrock(system: str, user: str) -> Optional[str]:
    if not BEDROCK_MODEL_ID:
        return None
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 900,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        resp = client.invoke_model(modelId=BEDROCK_MODEL_ID, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        return payload["content"][0]["text"]
    except Exception as e:
        print(f"  (business_overview: Bedrock unavailable -- {type(e).__name__})")
        return None


def _parse_json(text: str) -> dict:
    if not text:
        return {}
    # tolerate ```json fences or stray prose around the object
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _sanitize(d: dict) -> dict:
    out: dict[str, Any] = {}
    summ = d.get("summary")
    if isinstance(summ, str) and summ.strip():
        out["summary"] = summ.strip()

    bullets = d.get("bullets")
    if isinstance(bullets, list):
        clean = [str(b).strip() for b in bullets if str(b).strip()]
        if clean:
            out["bullets"] = clean[:4]

    segs = d.get("segments")
    if isinstance(segs, list):
        clean_segs = []
        for s in segs:
            if not isinstance(s, dict):
                continue
            name = str(s.get("name", "")).strip()
            pct = s.get("pct")
            if name and isinstance(pct, (int, float)) and 0 <= pct <= 100:
                clean_segs.append({"name": name, "pct": int(round(pct))})
        if clean_segs:
            out["segments"] = clean_segs[:6]
    return out


def summarize_business(business_text: str, company_name: str = "",
                       llm=None) -> dict:
    """Return a plain-English, filing-grounded business overview, or {} on failure.

    `llm(system, user) -> str` may be passed to reuse an existing wrapper
    (e.g. claude_helper.py); otherwise Bedrock is called directly.
    """
    text = (business_text or "").strip()
    if len(text) < 200:            # nothing meaningful to summarise
        return {}
    user = _USER_TMPL.format(company=company_name or "the company", text=text[:MAX_CHARS])
    raw = llm(_SYSTEM, user) if llm else _invoke_bedrock(_SYSTEM, user)
    return _sanitize(_parse_json(raw)) if raw else {}
