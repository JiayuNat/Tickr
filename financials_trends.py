"""
financials_trends.py  --  real multi-year financials for the Overview charts

Frontend item 4 (path B): instead of a single-year snapshot, we pull a few
years of Revenue, Net income, Net margin and Free cash flow straight from SEC's
XBRL "company concept" API. No extra API key -- SEC only requires that we send a
real identity in the User-Agent (the same EDGAR identity the coordinator uses).

Everything here is best-effort. Any network/parse problem returns {} and the
charts fall back to whatever single-period data is available. That keeps the
pipeline's graceful-degradation contract: a missing trend never breaks a run.

    from financials_trends import fetch_financial_trends
    trends = fetch_financial_trends(cik="1045810",
                                    identity="Jiayu Zhu zhujiayu878@gmail.com")
"""

from __future__ import annotations

import json
import urllib.request
from typing import Optional

SEC_CONCEPT = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"

# Preferred XBRL tags, in fallback order (companies tag revenue differently).
TAGS = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireMachineryAndEquipment",
    ],
}


def _cik_int(cik) -> Optional[int]:
    if cik is None:
        return None
    try:
        return int(str(cik).lstrip("CIK").lstrip("0") or "0")
    except Exception:
        return None


def _get_json(url: str, identity: str) -> Optional[dict]:
    req = urllib.request.Request(url, headers={
        "User-Agent": identity or "research contact@example.com",
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            # SEC may gzip; urllib doesn't auto-decompress, so handle it.
            if resp.headers.get("Content-Encoding") == "gzip":
                import gzip
                data = gzip.decompress(data)
            return json.loads(data)
    except Exception:
        return None


def _annual_by_fy(concept_json: Optional[dict]) -> dict:
    """Collapse a companyconcept payload into {fiscal_year: value} using annual
    (10-K, full-year) facts, preferring the most recently reported value."""
    out: dict[int, tuple[str, float]] = {}   # fy -> (end_date, value)
    if not concept_json:
        return {}
    usd = (concept_json.get("units") or {}).get("USD") or []
    for f in usd:
        fy = f.get("fy")
        fp = f.get("fp")
        form = str(f.get("form", ""))
        val = f.get("val")
        end = str(f.get("end", ""))
        if fy is None or val is None:
            continue
        # Full-year facts only: fp == 'FY' and a 10-K style form.
        if fp != "FY" or not form.startswith("10-K"):
            continue
        prev = out.get(int(fy))
        if prev is None or end > prev[0]:
            out[int(fy)] = (end, float(val))
    return {fy: v for fy, (_, v) in out.items()}


def _pull(cik_int: int, key: str, identity: str) -> dict:
    for tag in TAGS[key]:
        url = SEC_CONCEPT.format(cik=cik_int, tag=tag)
        data = _annual_by_fy(_get_json(url, identity))
        if data:
            return data
    return {}


def fetch_financial_trends(cik, identity: str, years: int = 5) -> dict:
    """Return up to `years` of annual trend data, or {} if unavailable.

    Shape:
      {
        "years":      ["FY22", ..., "FY26"],
        "revenue":    [..],
        "net_income": [..],
        "net_margin": [.. as decimals ..],
        "fcf":        [..]    # operating cash flow - capex, when both exist
      }
    """
    cik_int = _cik_int(cik)
    if cik_int is None:
        return {}

    revenue = _pull(cik_int, "revenue", identity)
    net_income = _pull(cik_int, "net_income", identity)
    if not revenue or not net_income:
        return {}

    ocf = _pull(cik_int, "operating_cash_flow", identity)
    capex = _pull(cik_int, "capex", identity)

    # Use fiscal years present in BOTH revenue and net income, most recent `years`.
    common = sorted(set(revenue) & set(net_income))[-years:]
    if len(common) < 2:
        return {}

    def series(src):
        return [src.get(fy) for fy in common]

    rev = series(revenue)
    ni = series(net_income)
    margin = [(ni[i] / rev[i]) if rev[i] else None for i in range(len(common))]
    fcf = None
    if ocf and capex:
        fcf = [(ocf.get(fy) - capex.get(fy)) if (ocf.get(fy) is not None and capex.get(fy) is not None) else None
               for fy in common]

    return {
        "years": [f"FY{fy % 100:02d}" for fy in common],
        "revenue": rev,
        "net_income": ni,
        "net_margin": margin,
        "fcf": fcf if fcf else [None] * len(common),
    }


if __name__ == "__main__":
    import sys
    # Quick live check:  python financials_trends.py NVDA_CIK "Name email@x.com"
    cik = sys.argv[1] if len(sys.argv) > 1 else "1045810"          # NVIDIA
    ident = sys.argv[2] if len(sys.argv) > 2 else "Research research@example.com"
    from pprint import pprint
    pprint(fetch_financial_trends(cik, ident))
