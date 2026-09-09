#pip3 install pandas edgartools matplotlib python-dotenv boto3
#pip3 install yfinance
import os
import re
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import yfinance as yf #to compare estimates, if any


# =====================================================================
# 1. MATH & SANITIZATION UTILITIES
# =====================================================================

def safe_div(n: Optional[float], d: Optional[float], round_digits: int = 4) -> Optional[float]:
    """Safely handles division avoiding ZeroDivisionError or None propagation."""
    if n is None or d is None or d == 0:
        return None
    return round(float(n) / float(d), round_digits)


def _clean_numeric(val: Any) -> Optional[float]:
    """Converts mixed accounting types (strings with $, commas, parentheses) into clean floats."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip().replace("$", "").replace(",", "").replace(" ", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]

    try:
        return float(s)
    except ValueError:
        return None


def format_currency(val: Optional[float]) -> str:
    """Formats raw floats into readable currency values ($B, $M, or thousands)."""
    if val is None:
        return "N/A"
    if abs(val) >= 1e9:
        return f"${val / 1e9:.2f}B"
    elif abs(val) >= 1e6:
        return f"${val / 1e6:.2f}M"
    return f"${val:,.2f}"


def format_pct(val: Optional[float]) -> str:
    """Formats decimal proportions to percentages."""
    if val is None:
        return "N/A"
    return f"{round(val * 100, 2)}%"


# =====================================================================
# 2. SEC EDGAR DATA EXTRACTION ENGINE
# =====================================================================

def _extract_from_statement_df(stmt_df: Any, search_terms: List[str]) -> Optional[float]:
    """Extracts a numeric line item from an edgartools statement DataFrame using fallback heuristics."""
    if stmt_df is None:
        return None

    if hasattr(stmt_df, "to_dataframe"):
        stmt_df = stmt_df.to_dataframe()
    elif callable(stmt_df):
        try:
            res = stmt_df()
            stmt_df = res.to_dataframe() if hasattr(res, "to_dataframe") else pd.DataFrame(res)
        except Exception:
            return None

    if not isinstance(stmt_df, pd.DataFrame) or stmt_df.empty:
        return None

    label_series = None
    for col in ["standard_concept", "concept", "label", "fact", "item", "line_item"]:
        if col in stmt_df.columns:
            label_series = stmt_df[col].astype(str)
            break

    if label_series is None:
        label_series = stmt_df.index.astype(str)

    val_cols = [c for c in stmt_df.columns if str(c).startswith("20") or pd.api.types.is_numeric_dtype(stmt_df[c])]
    if not val_cols:
        val_cols = [c for c in stmt_df.columns if c != label_series.name]

    for term in search_terms:
        matches = stmt_df[label_series.str.lower().str.contains(term.lower(), na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            for col in val_cols:
                val = _clean_numeric(row[col])
                if val is not None:
                    return val
    return None


def extract_financial_data(financials: Any) -> Dict[str, Optional[float]]:
    """Extracts all required raw dollar figures from edgartools statements and native getters."""
    if isinstance(financials, dict):
        return financials

    data: Dict[str, Optional[float]] = {}

    # 1. Native edgartools convenience methods
    getter_map = {
        "revenue": ["get_revenue", "revenue"],
        "net_income": ["get_net_income", "net_income"],
        "operating_income": ["get_operating_income", "operating_income"],
        "total_assets": ["get_total_assets", "total_assets"],
        "total_liabilities": ["get_total_liabilities", "total_liabilities"],
        "stockholders_equity": ["get_stockholders_equity", "equity"],
        "current_assets": ["get_current_assets", "current_assets"],
        "current_liabilities": ["get_current_liabilities", "current_liabilities"],
    }

    for key, methods in getter_map.items():
        for m in methods:
            if hasattr(financials, m):
                attr = getattr(financials, m)
                try:
                    val = attr() if callable(attr) else attr
                    num = _clean_numeric(val)
                    if num is not None:
                        data[key] = num
                        break
                except Exception:
                    pass

    # 2. Extract statement DataFrames
    income_stmt = None
    balance_sheet = None
    cash_flow = None

    for m in ["income_statement", "income", "get_income_statement"]:
        if hasattr(financials, m):
            obj = getattr(financials, m)
            income_stmt = obj() if callable(obj) else obj
            break

    for m in ["balance_sheet", "get_balance_sheet"]:
        if hasattr(financials, m):
            obj = getattr(financials, m)
            balance_sheet = obj() if callable(obj) else obj
            break

    for m in ["cash_flow_statement", "cash_flow", "get_cash_flow_statement", "cashflow_statement"]:
        if hasattr(financials, m):
            obj = getattr(financials, m)
            cash_flow = obj() if callable(obj) else obj
            break

    # 3. Fallback extraction for missing fields
    if not data.get("revenue"):
        data["revenue"] = _extract_from_statement_df(income_stmt, ["total revenue", "revenue", "sales"])
    if not data.get("net_income"):
        data["net_income"] = _extract_from_statement_df(income_stmt, ["net income", "net loss", "net earnings"])
    if not data.get("operating_income"):
        data["operating_income"] = _extract_from_statement_df(income_stmt, ["operating income", "operating profit", "ebit"])

    data["cogs"] = _extract_from_statement_df(income_stmt, ["cost of revenue", "cost of goods", "cost of sales", "cogs"])
    data["gross_profit"] = _extract_from_statement_df(income_stmt, ["gross profit"])
    data["interest_expense"] = _extract_from_statement_df(income_stmt, ["interest expense", "interest and other"])

    if not data.get("current_assets"):
        data["current_assets"] = _extract_from_statement_df(balance_sheet, ["current assets", "assets, current"])
    if not data.get("current_liabilities"):
        data["current_liabilities"] = _extract_from_statement_df(balance_sheet, ["current liabilities", "liabilities, current"])
    if not data.get("total_assets"):
        data["total_assets"] = _extract_from_statement_df(balance_sheet, ["total assets", "assets"])
    if not data.get("total_liabilities"):
        data["total_liabilities"] = _extract_from_statement_df(balance_sheet, ["total liabilities", "liabilities"])
    if not data.get("stockholders_equity"):
        data["stockholders_equity"] = _extract_from_statement_df(balance_sheet, ["stockholders' equity", "shareholders' equity", "total equity"])

    data["inventory"] = _extract_from_statement_df(balance_sheet, ["inventory", "inventories"]) or 0.0
    data["cash_and_equivalents"] = _extract_from_statement_df(balance_sheet, ["cash and cash equivalents", "cash & cash", "cash"])
    data["total_debt"] = _extract_from_statement_df(balance_sheet, ["total debt", "long-term debt", "commercial paper", "total borrowings"])

    data["operating_cash_flow"] = _extract_from_statement_df(cash_flow, ["operating activities", "operating cash flow", "cash from operations"])
    data["capital_expenditures"] = _extract_from_statement_df(cash_flow, ["capital expenditures", "property and equipment", "additions to property", "capex"]) or 0.0

    return data


# =====================================================================
# 3. YAHOO FINANCE CONSENSUS ESTIMATES
# =====================================================================

def fetch_analyst_consensus(ticker: str) -> Dict[str, Any]:
    """Fetches consensus analyst expectations and target metrics using yfinance."""
    estimates = {
        "expected_revenue_avg": None,
        "expected_revenue_low": None,
        "expected_revenue_high": None,
        "expected_eps_avg": None,
        "target_price_mean": None,
        "current_price": None,
    }
    try:
        stock = yf.Ticker(ticker)
        rev_est = stock.revenue_estimate
        if rev_est is not None and not rev_est.empty:
            col = "0y" if "0y" in rev_est.columns else rev_est.columns[0]
            if "avg" in rev_est.index:
                estimates["expected_revenue_avg"] = float(rev_est.loc["avg", col])
            if "low" in rev_est.index:
                estimates["expected_revenue_low"] = float(rev_est.loc["low", col])
            if "high" in rev_est.index:
                estimates["expected_revenue_high"] = float(rev_est.loc["high", col])

        eps_est = stock.earnings_estimate
        if eps_est is not None and not eps_est.empty:
            col = "0y" if "0y" in eps_est.columns else eps_est.columns[0]
            if "avg" in eps_est.index:
                estimates["expected_eps_avg"] = float(eps_est.loc["avg", col])

        info = stock.info or {}
        estimates["target_price_mean"] = info.get("targetMeanPrice")
        estimates["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
    except Exception as e:
        print(f"[Warning] Could not fetch consensus estimates: {e}")

    return estimates


# =====================================================================
# 4. DETERMINISTIC METRIC ENGINE
# =====================================================================

def calculate_metrics(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates all key financial health ratios with cleanly formatted text representations."""
    rev = raw_data.get("revenue")
    cogs = raw_data.get("cogs")
    gp = raw_data.get("gross_profit") or (rev - cogs if rev is not None and cogs is not None else None)
    op_inc = raw_data.get("operating_income")
    net_inc = raw_data.get("net_income")

    ca = raw_data.get("current_assets")
    cl = raw_data.get("current_liabilities")
    inv = raw_data.get("inventory", 0.0)
    cash = raw_data.get("cash_and_equivalents", 0.0)

    total_assets = raw_data.get("total_assets")
    total_liab = raw_data.get("total_liabilities")
    equity = raw_data.get("stockholders_equity") or (total_assets - total_liab if total_assets is not None and total_liab is not None else None)
    
    extracted_debt = raw_data.get("total_debt")
    total_debt = extracted_debt if extracted_debt is not None else total_liab
    int_exp = raw_data.get("interest_expense")

    cfo = raw_data.get("operating_cash_flow")
    capex = raw_data.get("capital_expenditures", 0.0)
    fcf = (cfo - abs(capex)) if (cfo is not None and capex is not None) else None

    # Working Capital
    nwc = (ca - cl) if (ca is not None and cl is not None) else None

    # Ratio Calculations
    gross_margin = safe_div(gp, rev)
    op_margin = safe_div(op_inc, rev)
    net_margin = safe_div(net_inc, rev)
    roe = safe_div(net_inc, equity)
    roa = safe_div(net_inc, total_assets)

    current_ratio = safe_div(ca, cl)
    quick_ratio = safe_div((ca - inv) if ca is not None and inv is not None else None, cl)
    cash_ratio = safe_div(cash, cl)

    debt_to_equity = safe_div(total_debt, equity)
    debt_to_assets = safe_div(total_debt, total_assets)
    interest_coverage = safe_div(op_inc, int_exp)
    cash_conversion = safe_div(cfo, net_inc)

    return {
        "profitability": {
            "gross_margin": gross_margin,
            "gross_margin_formatted": format_pct(gross_margin),
            "operating_margin": op_margin,
            "operating_margin_formatted": format_pct(op_margin),
            "net_margin": net_margin,
            "net_margin_formatted": format_pct(net_margin),
            "roe": roe,
            "roe_formatted": format_pct(roe),
            "roa": roa,
            "roa_formatted": format_pct(roa),
        },
        "liquidity": {
            "current_ratio": current_ratio,
            "quick_ratio": quick_ratio,
            "cash_ratio": cash_ratio,
            "net_working_capital_raw": nwc,
            "net_working_capital_formatted": format_currency(nwc),
        },
        "solvency_and_leverage": {
            "debt_to_equity": debt_to_equity,
            "debt_to_assets": debt_to_assets,
            "interest_coverage": interest_coverage,
            "total_debt_formatted": format_currency(total_debt),
            "stockholders_equity_formatted": format_currency(equity),
        },
        "cash_flow_and_quality": {
            "operating_cash_flow_formatted": format_currency(cfo),
            "capital_expenditures_formatted": format_currency(capex),
            "free_cash_flow_raw": fcf,
            "free_cash_flow_formatted": format_currency(fcf),
            "cash_conversion_ratio": cash_conversion,
            "cash_conversion_formatted": format_pct(cash_conversion),
        },
        "raw_dollars": {
            "revenue": format_currency(rev),
            "operating_income": format_currency(op_inc),
            "net_income": format_currency(net_inc),
            "total_assets": format_currency(total_assets),
            "total_liabilities": format_currency(total_liab),
            "cash_and_equivalents": format_currency(cash),
        }
    }


# =====================================================================
# 5. BENCHMARK SCORECARD & HEURISTICS
# =====================================================================

def evaluate_scorecard(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluates computed financial ratios against objective fundamental benchmarks."""
    scorecard = []
    
    prof = metrics.get("profitability", {})
    liq = metrics.get("liquidity", {})
    solv = metrics.get("solvency_and_leverage", {})
    cf = metrics.get("cash_flow_and_quality", {})

    # 1. Net Margin (Benchmark: >= 15%)
    nm = prof.get("net_margin")
    if nm is not None:
        status = "PASS" if nm >= 0.15 else ("WATCH" if nm >= 0.05 else "FAIL")
        scorecard.append({
            "category": "Profitability",
            "metric": "Net Profit Margin",
            "benchmark": ">= 15.0%",
            "actual": prof.get("net_margin_formatted"),
            "status": status,
            "insight": "High profit retention after all costs" if status == "PASS" else "Low margin cushion"
        })

    # 2. Return on Equity (Benchmark: >= 15%)
    roe = prof.get("roe")
    if roe is not None:
        status = "PASS" if roe >= 0.15 else ("WATCH" if roe >= 0.08 else "FAIL")
        scorecard.append({
            "category": "Profitability",
            "metric": "Return on Equity (ROE)",
            "benchmark": ">= 15.0%",
            "actual": prof.get("roe_formatted"),
            "status": status,
            "insight": "Outstanding capital compounding efficiency" if status == "PASS" else "Sub-par equity return"
        })

    # 3. Current Ratio (Benchmark: >= 1.5)
    cr = liq.get("current_ratio")
    if cr is not None:
        status = "PASS" if cr >= 1.5 else ("WATCH" if cr >= 1.0 else "FAIL")
        scorecard.append({
            "category": "Liquidity",
            "metric": "Current Ratio",
            "benchmark": ">= 1.50x",
            "actual": f"{cr:.2f}x",
            "status": status,
            "insight": "Comfortable buffer for short-term obligations" if status == "PASS" else "Potential working capital strain"
        })

    # 4. Debt-to-Equity (Benchmark: <= 1.5)
    de = solv.get("debt_to_equity")
    if de is not None:
        status = "PASS" if de <= 1.5 else ("WATCH" if de <= 2.5 else "FAIL")
        scorecard.append({
            "category": "Solvency",
            "metric": "Debt-to-Equity",
            "benchmark": "<= 1.50x",
            "actual": f"{de:.2f}x",
            "status": status,
            "insight": "Conservatively capitalized balance sheet" if status == "PASS" else "High debt leverage relative to equity"
        })

    # 5. Cash Conversion (Benchmark: >= 100%)
    cc = cf.get("cash_conversion_ratio")
    if cc is not None:
        status = "PASS" if cc >= 1.0 else ("WATCH" if cc >= 0.70 else "FAIL")
        scorecard.append({
            "category": "Earnings Quality",
            "metric": "Cash Conversion (CFO / Net Income)",
            "benchmark": ">= 100.0%",
            "actual": cf.get("cash_conversion_formatted"),
            "status": status,
            "insight": "Net income backed by real operating cash inflows" if status == "PASS" else "Accrual-heavy earnings"
        })

    return scorecard


# =====================================================================
# 6. MAIN AGENT ENTRYPOINT
# =====================================================================

def run_risk_research(context: Dict[str, Any]) -> Dict[str, Any]:
    ticker = context.get("ticker", "NVDA")
    financials = context.get("financials")

    # 1. Ingest and Calculate Deterministic Metrics
    raw_data = extract_financial_data(financials)
    metrics = calculate_metrics(raw_data)

    # 2. Wall Street Estimates vs Actuals
    analyst_estimates = fetch_analyst_consensus(ticker)
    actual_rev = raw_data.get("revenue")
    expected_rev = analyst_estimates.get("expected_revenue_avg")

    expected_vs_actual = []
    if actual_rev is not None and expected_rev is not None:
        diff_pct = (actual_rev - expected_rev) / expected_rev
        status = "BEAT" if diff_pct >= 0 else "MISS"
        expected_vs_actual.append({
            "metric": "Total Annual Revenue",
            "analyst_consensus": format_currency(expected_rev),
            "actual_reported": format_currency(actual_rev),
            "variance": f"{'+' if diff_pct >= 0 else ''}{round(diff_pct * 100, 2)}%",
            "status": status,
            "verdict": f"Reported revenue exceeded Wall Street consensus by {round(diff_pct * 100, 2)}%." if status == "BEAT" else f"Reported revenue trailed consensus by {round(abs(diff_pct) * 100, 2)}%."
        })
    elif actual_rev is not None:
        expected_vs_actual.append({
            "metric": "Total Annual Revenue",
            "analyst_consensus": "N/A (Historical/Unlisted)",
            "actual_reported": format_currency(actual_rev),
            "variance": "N/A",
            "status": "REPORTED",
            "verdict": "Reported directly from audited SEC 10-K filing."
        })

    # 3. Fundamental Scorecard
    scorecard = evaluate_scorecard(metrics)

    # 4. Generate Key Takeaway Bullets
    key_points = []
    for item in expected_vs_actual:
        if item.get("status") in ["BEAT", "MISS"]:
            key_points.append(f"Consensus Comparison: {item['metric']} came in at {item['actual_reported']} vs expected {item['analyst_consensus']} ({item['variance']} {item['status']}).")

    prof = metrics["profitability"]
    liq = metrics["liquidity"]
    cf = metrics["cash_flow_and_quality"]

    if prof.get("net_margin") is not None:
        key_points.append(f"Profitability: Generates {prof['net_margin_formatted']} net profit margin and {prof['roe_formatted']} Return on Equity.")
    if liq.get("current_ratio") is not None:
        key_points.append(f"Balance Sheet Safety: Current ratio of {liq['current_ratio']:.2f}x with {liq['net_working_capital_formatted']} in net working capital.")
    if cf.get("free_cash_flow_formatted") != "N/A":
        key_points.append(f"Cash Generation: Generated {cf['free_cash_flow_formatted']} in Free Cash Flow with a cash conversion of {cf['cash_conversion_formatted']}.")

    # 5. Deterministic Signal & Health Rating
    passes = sum(1 for item in scorecard if item["status"] == "PASS")
    fails = sum(1 for item in scorecard if item["status"] == "FAIL")

    net_margin = prof.get("net_margin") or 0.0

    if fails == 0 and passes >= 3 and net_margin > 0.10:
        signal = "positive"
        summary_verdict = "EXCELLENT: Robust balance sheet, superior profitability, and strong cash flow generation."
    elif fails >= 2 or net_margin < 0:
        signal = "cautious"
        summary_verdict = "CAUTION: Financial warning signs detected in leverage, liquidity, or negative margins."
    else:
        signal = "neutral"
        summary_verdict = "STABLE: Balanced risk profile meeting baseline operational benchmarks."

    return {
        "agent": "Risk & Research Agent",
        "ticker": ticker,
        "signal": signal,
        "executive_summary": summary_verdict,
        "scorecard": scorecard,
        "expected_vs_actual": expected_vs_actual,
        "key_points": key_points,
        "metrics": metrics,
        "market_consensus": {
            "target_price_mean": f"${analyst_estimates['target_price_mean']:.2f}" if analyst_estimates.get("target_price_mean") else "N/A",
            "current_price": f"${analyst_estimates['current_price']:.2f}" if analyst_estimates.get("current_price") else "N/A",
            "expected_eps_consensus": f"${analyst_estimates['expected_eps_avg']:.2f}" if analyst_estimates.get("expected_eps_avg") else "N/A",
        }
    }


# =====================================================================
# 7. EXECUTION / TEST BLOCK, not necessary if running with coordinator.py
# =====================================================================

if __name__ == "__main__":
    from coordinator import run_coordinator
    import pprint

    print("Executing Risk & Research Agent analysis pipeline...\n")
    context = run_coordinator(
        "NVDA",
        profile_answers={
            "experience": "some",
            "drawdown_reaction": "hold",
            "horizon": "long",
            "ownership": "none",
            "position_size": "small",
            "goal": "growth",
        },
    )
    result = run_risk_research(context)
    pprint.pprint(result, sort_dicts=False, width=120)