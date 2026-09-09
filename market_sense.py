"""
market_sense.py

MARKET SENSE AGENT
------------------
Purpose:
    Given the shared context produced by coordinator.py, research the
    CURRENT market environment relevant to the selected company.

Flow:
    coordinator.py
        ↓
    context
        ↓
    identify company + SEC industry
        ↓
    select relevant macro indicators
        ↓
    FRED macroeconomic data
        +
    Alpha Vantage company news
        +
    Alpha Vantage industry news
        +
    Alpha Vantage macro news
        ↓
    Claude via Amazon Bedrock
        ↓
    structured Market Sense result

Important:
    This agent does NOT:
        - calculate beta
        - calculate covariance
        - value the company
        - analyse financial statements
        - make a buy/sell recommendation

    Its role is simply:

        "What is happening in the market now,
         and why does it matter to THIS company?"
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv(override=True)

FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()

ALPHA_VANTAGE_API_KEY = os.environ.get(
    "ALPHA_VANTAGE_API_KEY",
    ""
).strip()

AWS_REGION = os.environ.get(
    "AWS_DEFAULT_REGION",
    "us-east-1"
).strip()

BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"
).strip()


FRED_URL = (
    "https://api.stlouisfed.org/"
    "fred/series/observations"
)

ALPHA_VANTAGE_URL = (
    "https://www.alphavantage.co/query"
)


# Number of articles passed to Claude.
# Keep this relatively small to reduce token usage.
MAX_COMPANY_NEWS = 8
MAX_INDUSTRY_NEWS = 8
MAX_MACRO_NEWS = 5

# Search approximately one month of news.
NEWS_LOOKBACK_DAYS = 30


# ============================================================
# TRUSTED NEWS SOURCES
# ============================================================

# We prefer established news organisations.
#
# If Alpha Vantage returns no articles from these sources,
# the program can optionally fall back to other sources instead
# of crashing.

TRUSTED_NEWS_SOURCES = {
    "reuters",
    "bloomberg",
    "financial times",
    "wall street journal",
    "the wall street journal",
    "associated press",
    "ap news",
    "cnbc",
    "bbc",
    "nikkei asia",
    "barron's",
    "barrons",
    "marketwatch",
}


# ============================================================
# 1. CLASSIFY SEC INDUSTRY
# ============================================================

def classify_industry(industry: str) -> str:
    """
    Convert the detailed SEC / EdgarTools industry description
    into a broad research category.

    These categories broadly align with the topic filters
    supported by Alpha Vantage NEWS_SENTIMENT.
    """

    text = (industry or "").upper()

    # --------------------------------------------------------
    # TECHNOLOGY
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "SEMICONDUCTOR",
            "SOFTWARE",
            "COMPUTER",
            "ELECTRONIC",
            "INTERNET",
            "TECHNOLOGY",
            "DATA PROCESSING",
            "COMMUNICATION EQUIPMENT",
        ]
    ):
        return "technology"

    # --------------------------------------------------------
    # FINANCE
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "BANK",
            "FINANCE",
            "FINANCIAL",
            "INSURANCE",
            "INVESTMENT",
            "CREDIT",
            "BROKER",
            "SECURITIES",
        ]
    ):
        return "finance"

    # --------------------------------------------------------
    # ENERGY / TRANSPORTATION
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "OIL",
            "GAS",
            "ENERGY",
            "PETROLEUM",
            "PIPELINE",
            "AIRLINE",
            "TRANSPORTATION",
            "RAILROAD",
        ]
    ):
        return "energy_transportation"

    # --------------------------------------------------------
    # LIFE SCIENCES
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "PHARMACEUTICAL",
            "BIOTECH",
            "MEDICAL",
            "HEALTH",
            "DRUG",
            "BIOLOGICAL",
        ]
    ):
        return "life_sciences"

    # --------------------------------------------------------
    # MANUFACTURING / INDUSTRIALS
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "MANUFACTUR",
            "MACHINERY",
            "CHEMICAL",
            "INDUSTRIAL",
            "STEEL",
            "METAL",
            "EQUIPMENT",
        ]
    ):
        return "manufacturing"

    # --------------------------------------------------------
    # REAL ESTATE
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "REAL ESTATE",
            "REIT",
            "PROPERTY",
            "CONSTRUCTION",
            "HOME BUILDER",
        ]
    ):
        return "real_estate"

    # --------------------------------------------------------
    # RETAIL / CONSUMER
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "RETAIL",
            "WHOLESALE",
            "STORE",
            "RESTAURANT",
            "FOOD SERVICE",
            "APPAREL",
        ]
    ):
        return "retail_wholesale"

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    return "financial_markets"


# ============================================================
# 2. EXTRACT COMPANY INFORMATION FROM CONTEXT
# ============================================================

def get_company_context(context: dict) -> dict:
    """
    Read company information already collected by coordinator.py.

    Market Sense should NOT fetch SEC company identity again.
    """

    if not isinstance(context, dict):
        raise TypeError(
            "context must be a dictionary returned by coordinator.py"
        )

    company = context.get("company", {})

    ticker = str(
        company.get("ticker", "")
    ).strip().upper()

    if not ticker:
        raise ValueError(
            "context['company']['ticker'] is missing."
        )

    name = str(
        company.get("name", ticker)
    ).strip()

    industry = str(
        company.get("industry", "")
    ).strip()

    sic = company.get("sic")

    exchange = str(
        company.get("exchange", "")
    ).strip()

    research_topic = classify_industry(industry)

    return {
        "ticker": ticker,
        "name": name,
        "industry": industry,
        "sic": sic,
        "exchange": exchange,
        "research_topic": research_topic,
    }


# ============================================================
# 3. MACRO DATA CONFIGURATION
# ============================================================

# These indicators are useful for almost every stock.

BASE_MACRO_SERIES = {

    "fed_funds": {
        "series_id": "DFF",
        "label": "Effective Federal Funds Rate",
        "units": None,
    },

    "treasury_10y": {
        "series_id": "DGS10",
        "label": "10-Year US Treasury Yield",
        "units": None,
    },

    "inflation_cpi_yoy": {
        "series_id": "CPIAUCSL",
        "label": "US CPI Inflation YoY",
        "units": "pc1",
    },

    "unemployment": {
        "series_id": "UNRATE",
        "label": "US Unemployment Rate",
        "units": None,
    },

    "real_gdp_growth": {
        "series_id": "A191RL1Q225SBEA",
        "label": "US Real GDP Growth",
        "units": None,
    },

    "vix": {
        "series_id": "VIXCLS",
        "label": "CBOE VIX",
        "units": None,
    },
}


# Additional variables only retrieved when relevant
# to that industry.

INDUSTRY_MACRO_SERIES = {

    # --------------------------------------------------------
    # FINANCIALS
    # --------------------------------------------------------

    "finance": {

        "yield_curve": {
            "series_id": "T10Y2Y",
            "label": "10Y minus 2Y Treasury Spread",
            "units": None,
        },

    },

    # --------------------------------------------------------
    # ENERGY / TRANSPORTATION
    # --------------------------------------------------------

    "energy_transportation": {

        "wti_oil": {
            "series_id": "DCOILWTICO",
            "label": "WTI Crude Oil Price",
            "units": None,
        },

    },

    # --------------------------------------------------------
    # MANUFACTURING
    # --------------------------------------------------------

    "manufacturing": {

        "industrial_production": {
            "series_id": "INDPRO",
            "label": "US Industrial Production YoY",
            "units": "pc1",
        },

    },

    # --------------------------------------------------------
    # REAL ESTATE
    # --------------------------------------------------------

    "real_estate": {

        "mortgage_rate": {
            "series_id": "MORTGAGE30US",
            "label": "30-Year US Mortgage Rate",
            "units": None,
        },

    },

    # --------------------------------------------------------
    # CONSUMER / RETAIL
    # --------------------------------------------------------

    "retail_wholesale": {

        "retail_sales": {
            "series_id": "RSAFS",
            "label": "US Retail Sales YoY",
            "units": "pc1",
        },

    },

    # Technology and life sciences currently use
    # the common macro variables.
    #
    # Their sector-specific information is primarily obtained
    # through the news / industry research layer.

    "technology": {},

    "life_sciences": {},

    "financial_markets": {},
}


def get_macro_series_for_company(
    research_topic: str
) -> dict:
    """
    Combine general macro indicators with indicators
    specific to the company's industry.
    """

    selected = {
        key: value.copy()
        for key, value
        in BASE_MACRO_SERIES.items()
    }

    extra = INDUSTRY_MACRO_SERIES.get(
        research_topic,
        {}
    )

    for key, value in extra.items():
        selected[key] = value.copy()

    return selected


# ============================================================
# 4. FETCH FRED DATA
# ============================================================

def fetch_fred_series(
    series_id: str,
    units: str | None = None,
) -> dict:
    """
    Retrieve recent observations for one FRED series.

    Returns the latest and previous valid observations.
    """

    if not FRED_API_KEY:
        raise RuntimeError(
            "FRED_API_KEY is missing from .env"
        )

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 12,
    }

    if units:
        params["units"] = units

    response = requests.get(
        FRED_URL,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    if "error_message" in data:
        raise RuntimeError(
            data["error_message"]
        )

    valid_observations = []

    for observation in data.get(
        "observations",
        []
    ):

        raw_value = observation.get(
            "value"
        )

        if raw_value in {
            None,
            "",
            ".",
        }:
            continue

        try:
            value = float(raw_value)
        except ValueError:
            continue

        valid_observations.append({
            "date": observation.get("date"),
            "value": value,
        })

    if not valid_observations:
        return {
            "series_id": series_id,
            "latest": None,
            "previous": None,
        }

    latest = valid_observations[0]

    previous = (
        valid_observations[1]
        if len(valid_observations) > 1
        else None
    )

    change = None

    if previous is not None:
        change = (
            latest["value"]
            - previous["value"]
        )

    return {
        "series_id": series_id,
        "latest": latest,
        "previous": previous,
        "change_from_previous": change,
    }


def fetch_macro_snapshot(
    research_topic: str
) -> dict:
    """
    Retrieve all macro indicators relevant to the company.

    One failed FRED series does NOT crash the entire agent.
    """

    selected_series = (
        get_macro_series_for_company(
            research_topic
        )
    )

    results = {}

    for key, config in selected_series.items():

        try:

            fetched = fetch_fred_series(
                series_id=config["series_id"],
                units=config.get("units"),
            )

            results[key] = {
                "label": config["label"],
                **fetched,
            }

        except Exception as exc:

            results[key] = {
                "label": config["label"],
                "series_id": config["series_id"],
                "error": str(exc),
            }

    return results


# ============================================================
# 5. NEWS HELPERS
# ============================================================

def normalise_source_name(
    source: str
) -> str:

    return (
        source
        .lower()
        .strip()
    )


def is_trusted_source(
    source: str
) -> bool:
    """
    Determine whether a returned publication is one
    of our preferred credible sources.
    """

    normalised = normalise_source_name(
        source
    )

    if not normalised:
        return False

    for trusted in TRUSTED_NEWS_SOURCES:

        if trusted in normalised:
            return True

    return False


def parse_alpha_vantage_date(
    value: str
) -> str:
    """
    Alpha Vantage commonly returns timestamps in:
        YYYYMMDDTHHMMSS

    Convert to ISO-like text for easier downstream use.
    """

    if not value:
        return ""

    try:

        parsed = datetime.strptime(
            value,
            "%Y%m%dT%H%M%S",
        )

        return parsed.isoformat()

    except ValueError:

        return value


def fetch_news(
    ticker: str | None = None,
    topic: str | None = None,
    limit: int = 10,
    trusted_only: bool = True,
) -> list[dict]:
    """
    Fetch news from Alpha Vantage NEWS_SENTIMENT.

    Can search by:
        ticker
        OR
        topic

    We intentionally do NOT use Alpha Vantage's
    sentiment score as our conclusion.

    Claude will interpret materiality itself.
    """

    if not ALPHA_VANTAGE_API_KEY:

        raise RuntimeError(
            "ALPHA_VANTAGE_API_KEY is missing from .env"
        )

    if not ticker and not topic:

        raise ValueError(
            "fetch_news requires ticker or topic."
        )

    time_from = (
        datetime.now(timezone.utc)
        - timedelta(days=NEWS_LOOKBACK_DAYS)
    )

    time_from_string = time_from.strftime(
        "%Y%m%dT%H%M"
    )

    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": ALPHA_VANTAGE_API_KEY,
        "sort": "LATEST",
        "limit": 50,
        "time_from": time_from_string,
    }

    if ticker:
        params["tickers"] = ticker

    if topic:
        params["topics"] = topic

    response = requests.get(
        ALPHA_VANTAGE_URL,
        params=params,
        timeout=25,
    )

    response.raise_for_status()

    data = response.json()

    # Alpha Vantage sometimes returns messages about
    # rate limits / invalid API keys instead of "feed".

    if "Information" in data:

        raise RuntimeError(
            data["Information"]
        )

    if "Note" in data:

        raise RuntimeError(
            data["Note"]
        )

    if "Error Message" in data:

        raise RuntimeError(
            data["Error Message"]
        )

    feed = data.get(
        "feed",
        []
    )

    all_articles = []

    for article in feed:

        source = str(
            article.get(
                "source",
                ""
            )
        )

        all_articles.append({

            "title": article.get(
                "title",
                ""
            ),

            "summary": article.get(
                "summary",
                ""
            ),

            "source": source,

            "published": (
                parse_alpha_vantage_date(
                    article.get(
                        "time_published",
                        ""
                    )
                )
            ),

            "url": article.get(
                "url",
                ""
            ),

            "trusted_source": (
                is_trusted_source(
                    source
                )
            ),
        })

    # Prefer trusted publications.

    trusted_articles = [
        article
        for article in all_articles
        if article["trusted_source"]
    ]

    if trusted_only:

        selected = trusted_articles

    else:

        selected = all_articles

    return selected[:limit]


# ============================================================
# 6. FETCH COMPANY / INDUSTRY / MACRO NEWS
# ============================================================

def safe_fetch_news(
    ticker: str | None = None,
    topic: str | None = None,
    limit: int = 10,
) -> dict:
    """
    Wrapper so one failed API query does not crash
    the entire Market Sense agent.
    """

    try:

        trusted = fetch_news(
            ticker=ticker,
            topic=topic,
            limit=limit,
            trusted_only=True,
        )

        # If nothing from our preferred source list appears,
        # return a small fallback set but mark this clearly.

        if not trusted:

            fallback = fetch_news(
                ticker=ticker,
                topic=topic,
                limit=min(limit, 5),
                trusted_only=False,
            )

            return {
                "articles": fallback,
                "trusted_filter_used": False,
                "warning": (
                    "No articles passed the preferred "
                    "news-source filter. Fallback results "
                    "were included."
                ),
            }

        return {
            "articles": trusted,
            "trusted_filter_used": True,
            "warning": None,
        }

    except Exception as exc:

        return {
            "articles": [],
            "trusted_filter_used": False,
            "error": str(exc),
        }


def fetch_market_news(
    company_info: dict
) -> dict:
    """
    Run three separate news searches:

        1. Company
        2. Industry
        3. Macro economy
    """

    ticker = company_info["ticker"]

    research_topic = company_info[
        "research_topic"
    ]

    # COMPANY-SPECIFIC NEWS

    company_news = safe_fetch_news(
        ticker=ticker,
        limit=MAX_COMPANY_NEWS,
    )

    # INDUSTRY NEWS

    industry_news = safe_fetch_news(
        topic=research_topic,
        limit=MAX_INDUSTRY_NEWS,
    )

    # MACRO NEWS

    macro_news = safe_fetch_news(
        topic="economy_macro",
        limit=MAX_MACRO_NEWS,
    )

    return {
        "company_news": company_news,
        "industry_news": industry_news,
        "macro_news": macro_news,
    }


# ============================================================
# 7. BUILD SOURCE / EVIDENCE PACKET
# ============================================================

def build_market_evidence(
    context: dict
) -> dict:
    """
    Build the factual evidence packet that Claude receives.

    IMPORTANT:
    Claude does not retrieve current facts itself.

    Python/APIs retrieve the facts.
    Claude interprets them.
    """

    company_info = (
        get_company_context(
            context
        )
    )

    print(
        "\nMarket Sense Agent"
    )

    print(
        f"Company: "
        f"{company_info['name']} "
        f"({company_info['ticker']})"
    )

    print(
        f"SEC industry: "
        f"{company_info['industry'] or 'Not provided'}"
    )

    print(
        f"Research category: "
        f"{company_info['research_topic']}"
    )

    print(
        "\nFetching macroeconomic data..."
    )

    macro = (
        fetch_macro_snapshot(
            company_info[
                "research_topic"
            ]
        )
    )

    print(
        "Fetching latest company, "
        "industry and macro news..."
    )

    news = fetch_market_news(
        company_info
    )

    evidence = {

        "as_of": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "company": company_info,

        "macro_data": macro,

        "company_news": (
            news[
                "company_news"
            ]
        ),

        "industry_news": (
            news[
                "industry_news"
            ]
        ),

        "macro_news": (
            news[
                "macro_news"
            ]
        ),

    }

    return evidence


# ============================================================
# 8. BUILD SOURCE IDS
# ============================================================

def add_source_ids(
    evidence: dict
) -> tuple[dict, dict]:
    """
    Give evidence explicit IDs.

    Claude must refer to these IDs instead of inventing sources.

    Returns:
        evidence_for_model
        source_registry
    """

    source_registry = {}

    model_evidence = {
        "as_of": evidence["as_of"],
        "company": evidence["company"],
        "macro_data": {},
        "company_news": [],
        "industry_news": [],
        "macro_news": [],
    }

    # --------------------------------------------------------
    # FRED
    # --------------------------------------------------------

    for key, value in (
        evidence[
            "macro_data"
        ].items()
    ):

        source_id = (
            f"fred_{value.get('series_id', key)}"
        )

        source_registry[source_id] = {

            "type": "macroeconomic_data",

            "publisher": (
                "Federal Reserve Bank "
                "of St. Louis (FRED)"
            ),

            "series_id": value.get(
                "series_id"
            ),

            "label": value.get(
                "label"
            ),

            "latest": value.get(
                "latest"
            ),

        }

        model_evidence[
            "macro_data"
        ][source_id] = value

    # --------------------------------------------------------
    # NEWS
    # --------------------------------------------------------

    news_groups = [
        (
            "company_news",
            "company"
        ),

        (
            "industry_news",
            "industry"
        ),

        (
            "macro_news",
            "macro"
        ),
    ]

    for group_name, prefix in news_groups:

        group = evidence.get(
            group_name,
            {}
        )

        articles = group.get(
            "articles",
            []
        )

        for index, article in enumerate(
            articles,
            start=1,
        ):

            source_id = (
                f"news_{prefix}_{index}"
            )

            source_registry[
                source_id
            ] = {

                "type": "news",

                "publisher": (
                    article.get(
                        "source"
                    )
                ),

                "title": (
                    article.get(
                        "title"
                    )
                ),

                "date": (
                    article.get(
                        "published"
                    )
                ),

                "url": (
                    article.get(
                        "url"
                    )
                ),

            }

            model_article = {
                "source_id": source_id,
                **article,
            }

            model_evidence[
                group_name
            ].append(
                model_article
            )

    return (
        model_evidence,
        source_registry,
    )


# ============================================================
# 9. AMAZON BEDROCK / CLAUDE
# ============================================================

def call_claude(
    prompt: str
) -> str:
    """
    Send prompt to Claude through Amazon Bedrock.

    BEDROCK_MODEL_ID must be placed in .env.
    """

    if not BEDROCK_MODEL_ID:

        raise RuntimeError(
            "BEDROCK_MODEL_ID is missing from .env"
        )

    client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
    )

    response = client.converse(
        modelId=BEDROCK_MODEL_ID,

        system=[
            {
                "text": (
                    "You are a disciplined market and "
                    "macroeconomic research analyst. "
                    "Use ONLY evidence supplied by the "
                    "user. Never invent market data, "
                    "news, company facts, forecasts, "
                    "sources, or investment returns. "
                    "Clearly distinguish evidence from "
                    "interpretation. Do not give a "
                    "buy/sell recommendation."
                )
            }
        ],

        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ],
            }
        ],

        inferenceConfig={
            "maxTokens": 5000,
            "temperature": 0.1,
        },
    )

    print("Claude stop reason:", response.get("stopReason"))
    print(
        "Claude output tokens:",
        response.get("usage", {}).get("outputTokens")
    )
    content = (
        response
        .get("output", {})
        .get("message", {})
        .get("content", [])
    )

    texts = []

    for item in content:

        if "text" in item:
            texts.append(
                item["text"]
            )

    if not texts:

        raise RuntimeError(
            "Bedrock returned no text."
        )

    return "\n".join(texts)


# ============================================================
# 10. JSON EXTRACTION
# ============================================================

def extract_json(
    text: str
) -> dict:
    """
    Claude may occasionally surround JSON with markdown.

    This extracts the JSON object safely.
    """

    text = text.strip()

    # Remove Markdown fences.

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    # First try direct parsing.

    try:

        parsed = json.loads(text)

        if isinstance(
            parsed,
            dict
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    # Otherwise find the first full-looking object.

    start = text.find("{")
    end = text.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):

        raise ValueError(
            "Claude did not return a JSON object."
        )

    candidate = text[
        start:end + 1
    ]

    parsed = json.loads(
        candidate
    )

    if not isinstance(
        parsed,
        dict
    ):

        raise ValueError(
            "Claude JSON output was not a dictionary."
        )

    return parsed


# ============================================================
# 11. BUILD CLAUDE PROMPT
# ============================================================

def build_analysis_prompt(
    evidence: dict
) -> str:
    """
    Ask Claude to convert raw evidence into a structured
    Market Sense assessment.
    """

    evidence_json = json.dumps(
        evidence,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    return f"""
You are the MARKET SENSE sub-agent in a larger portfolio
decision-support system.

Your job is NOT to decide whether the investor should buy
or sell this stock.

Other agents handle:
- 10-K fundamental analysis
- valuation
- financial ratios
- covariance
- beta
- portfolio risk
- portfolio return
- final investment recommendation

YOUR JOB:

Determine what is happening CURRENTLY in:

1. the macroeconomic environment,
2. the company's industry,
3. the company itself,

and explain why those developments could matter to the
company.

============================================================
EVIDENCE
============================================================

{evidence_json}

============================================================
ANALYSIS RULES
============================================================

1. Use ONLY the supplied evidence.

2. Do NOT use your internal memory to state current:
   - interest rates
   - inflation
   - GDP
   - market prices
   - news
   - regulations
   - company announcements

3. If evidence is missing, say it is missing.

4. Do NOT make a BUY, SELL or HOLD recommendation.

5. Do NOT provide a target price.

6. Do NOT calculate valuation.

7. Do NOT calculate beta, covariance or portfolio risk.

8. Distinguish:

   FACT:
   directly shown in supplied data/news

   INTERPRETATION:
   your explanation of why the fact may matter

9. Focus on transmission mechanisms.

Example:

    Treasury yields rise
        ->
    discount rates may rise
        ->
    valuations of long-duration growth equities
    may face pressure

rather than simply saying:

    "Higher rates are bad."

10. Material news should be prioritised over minor stories.

11. Every material event and important macro conclusion
must include one or more source_ids from the supplied
evidence.

12. Treat management/news claims cautiously.

============================================================
SIGNAL DEFINITION
============================================================

The signal measures ONLY the current market backdrop for
this company.

positive:
    current market / macro / industry developments appear
    more supportive than adverse.

neutral:
    effects appear broadly balanced or there is
    insufficient evidence.

negative:
    current developments appear more adverse than
    supportive.

This is NOT an investment recommendation.

============================================================
RETURN FORMAT
============================================================

Return ONLY valid JSON.

No Markdown.

Use exactly this general structure:

{{
    "signal": "positive | neutral | negative",

    "summary": "1-3 concise sentences explaining the current market environment for this company",

    "key_points": [
        "important point 1",
        "important point 2",
        "important point 3"
    ],

    "industry_outlook": {{
        "signal": "positive | neutral | negative",
        "reason": "explanation",
        "source_ids": ["source_id"]
    }},

    "macro_assessment": {{

        "rates": {{
            "effect": "positive | neutral | negative | unclear",
            "reason": "explanation",
            "source_ids": ["source_id"]
        }},

        "inflation": {{
            "effect": "positive | neutral | negative | unclear",
            "reason": "explanation",
            "source_ids": ["source_id"]
        }},

        "growth": {{
            "effect": "positive | neutral | negative | unclear",
            "reason": "explanation",
            "source_ids": ["source_id"]
        }},

        "risk_appetite": {{
            "effect": "positive | neutral | negative | unclear",
            "reason": "explanation",
            "source_ids": ["source_id"]
        }}
    }},

    "company_events": [
        {{
            "event": "description",
            "effect": "positive | neutral | negative | mixed",
            "materiality": "high | medium | low",
            "horizon": "short | medium | long",
            "reason": "why it matters",
            "source_ids": ["source_id"]
        }}
    ],

    "industry_events": [
        {{
            "event": "description",
            "effect": "positive | neutral | negative | mixed",
            "materiality": "high | medium | low",
            "reason": "why it matters",
            "source_ids": ["source_id"]
        }}
    ],

    "tailwinds": [
        {{
            "factor": "description",
            "reason": "why this supports the company",
            "source_ids": ["source_id"]
        }}
    ],

    "headwinds": [
        {{
            "factor": "description",
            "reason": "why this may hurt the company",
            "source_ids": ["source_id"]
        }}
    ],

    "what_to_watch": [
        "future development 1",
        "future development 2",
        "future development 3"
    ],

    "confidence": 0.0,

    "limitations": [
        "limitation if applicable"
    ]
}}

Confidence must be between 0 and 1.
"""


# ============================================================
# 12. VALIDATE RESULT
# ============================================================

VALID_SIGNALS = {
    "positive",
    "neutral",
    "negative",
}


def validate_market_result(
    result: dict
) -> dict:
    """
    Basic validation so the final advisor receives
    a predictable dictionary.
    """

    signal = str(
        result.get(
            "signal",
            "neutral"
        )
    ).lower()

    if signal not in VALID_SIGNALS:
        signal = "neutral"

    result["signal"] = signal

    if not isinstance(
        result.get("summary"),
        str
    ):
        result["summary"] = (
            "Market assessment unavailable."
        )

    if not isinstance(
        result.get("key_points"),
        list
    ):
        result["key_points"] = []

    try:

        confidence = float(
            result.get(
                "confidence",
                0.5
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        confidence = 0.5

    confidence = max(
        0.0,
        min(
            1.0,
            confidence
        )
    )

    result[
        "confidence"
    ] = confidence

    return result


# ============================================================
# 13. MAIN CLAUDE ANALYSIS
# ============================================================

def analyse_market_evidence(
    raw_evidence: dict
) -> dict:
    """
    Add source IDs, send evidence to Claude,
    and parse the structured result.
    """

    (
        model_evidence,
        source_registry,
    ) = add_source_ids(
        raw_evidence
    )

    prompt = build_analysis_prompt(
        model_evidence
    )

    print(
        "Analysing current market environment "
        "with Claude..."
    )

    response_text = call_claude(
        prompt
    )

    result = extract_json(
        response_text
    )

    result = validate_market_result(
        result
    )

    # Attach the source registry AFTER Claude responds.
    # This makes the final output auditable.

    result["sources"] = (
        source_registry
    )

    result["as_of"] = (
        raw_evidence["as_of"]
    )

    result["company"] = (
        raw_evidence["company"]
    )

    return result


# ============================================================
# 14. PUBLIC AGENT FUNCTION
# ============================================================

def run_market_sense_agent(context: dict) -> dict:
    try:
        evidence = build_market_evidence(context)
        result = analyse_market_evidence(evidence)
        return result
    except Exception as exc:
        print(f"[Warning] Market Sense encountered an error: {exc}")
        return {
            "signal": "neutral",
            "confidence": 0.0,
            "summary": f"Market backdrop analysis unavailable ({type(exc).__name__}: check AWS token validity).",
            "key_points": ["Could not reach AWS Bedrock Claude model."],
            "is_empty": True,
            "company": context.get("company", {}),
            "macro_assessment": {},
            "tailwinds": [],
            "headwinds": [],
        }


# ============================================================
# 15. PRETTY PRINT FOR TESTING
# ============================================================

def preview_market_result(
    result: dict
) -> None:
    """
    Terminal preview.
    """

    print(
        "\n"
        + "=" * 65
    )

    print(
        "MARKET SENSE RESULT"
    )

    print(
        "=" * 65
    )

    company = result.get(
        "company",
        {}
    )

    print(
        f"\nCompany: "
        f"{company.get('name')} "
        f"({company.get('ticker')})"
    )

    print(
        f"Industry: "
        f"{company.get('industry')}"
    )

    print(
        f"Signal: "
        f"{result.get('signal', '').upper()}"
    )

    print(
        f"Confidence: "
        f"{result.get('confidence')}"
    )

    print(
        "\nSummary:"
    )

    print(
        result.get(
            "summary",
            ""
        )
    )

    print(
        "\nKey points:"
    )

    for point in result.get(
        "key_points",
        []
    ):

        print(
            f"  - {point}"
        )

    print(
        "\nWhat to watch:"
    )

    for item in result.get(
        "what_to_watch",
        []
    ):

        print(
            f"  - {item}"
        )

    print(
        "\n"
        + "=" * 65
    )


# ============================================================
# 16. TEST THIS AGENT BY ITSELF
# ============================================================

if __name__ == "__main__":

    # Import only during local testing.
    #
    # This avoids circular imports when the website eventually
    # imports run_market_sense_agent().

    from coordinator import run_coordinator, ask_ticker_cli

    ticker = ask_ticker_cli()

    print(
        f"\nTesting Market Sense Agent with {ticker}...\n"
    )

    context = run_coordinator(

        ticker,

        profile_answers={

            "experience": "some",

            "drawdown_reaction": "hold",

            "horizon": "long",

            "ownership": "none",

            "position_size": "small",

            "goal": "growth",

        },
    )

    result = (
        run_market_sense_agent(
            context
        )
    )

    preview_market_result(
        result
    )

    # Uncomment this if you want to see
    # the entire final dictionary / JSON.

    # print(
    #     json.dumps(
    #         result,
    #         indent=2,
    #         default=str
    #     )
    # )