"""
Coordinator agent  --  "The Overwhelmed Everyday Investor"

This file does exactly the two jobs from the plan:
  1. Build a short USER PROFILE by asking a few simple questions.
  2. FETCH the latest 10-K annual report for a stock ticker from SEC EDGAR.

It bundles both into ONE tidy `context` dictionary that the three analysis
agents (10-K, Market Sense, Risk & Research) will read from.

-------------------------------------------------------------------------
TRY IT NOW (in a terminal):
    pip install edgartools
    python coordinator.py
-------------------------------------------------------------------------
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import os

try:
    from edgar import Company, set_identity
except ImportError:
    raise SystemExit("edgartools is not installed.  Run:  pip install edgartools")


# =========================================================================
# PART 1  --  ASK WHICH STOCK
# =========================================================================

def ask_ticker_cli() -> str:
    """Ask the person which stock to look at.
    In the website this simply becomes a text box."""
    while True:
        raw = input("Which US stock? Enter its ticker (e.g. NVDA, AAPL): ").strip().upper()
        cleaned = raw.replace(".", "").replace("-", "")
        if cleaned.isalnum() and cleaned != "" and len(raw) <= 8:
            return raw
        print("  Please enter a ticker made of letters, like NVDA or BRK.B.")


# =========================================================================
# PART 2  --  USER PROFILE
# =========================================================================
# The questions live here as plain data, so the SAME list can drive both the
# terminal version below AND the website form later. To change a question,
# just edit this list -- nothing else needs to change.
#
# Note: the questions avoid asking for actual dollar amounts (private, and
# not needed). Where "how big" matters, we ask it as a rough % band instead.

PROFILE_QUESTIONS = [
    {
        "key": "experience",
        "prompt": "How long have you been investing?",
        "options": [
            ("new",         "Less than 1 year"),
            ("some",        "1 to 5 years"),
            ("experienced", "More than 5 years"),
        ],
    },
    {
        # A behaviour question, not a self-label. People predict how they'd
        # ACT far more reliably than they rate their own "risk tolerance".
        "key": "drawdown_reaction",
        "prompt": "If this stock dropped 25% in a month, what would you most likely do?",
        "options": [
            ("buy_more",    "Buy more -- it's cheaper now"),
            ("hold",        "Hold and wait it out"),
            ("hold_uneasy", "Hold, but lose some sleep over it"),
            ("sell",        "Sell to stop further losses"),
        ],
    },
    {
        "key": "horizon",
        "prompt": "How long do you plan to hold?",
        "options": [
            ("short",  "Under a year"),
            ("medium", "One to five years"),
            ("long",   "Five years or more"),
        ],
    },
    {
        "key": "ownership",
        "prompt": "What's your relationship to this stock right now?",
        "options": [
            ("none",        "I don't own it"),
            ("considering", "I'm thinking of buying"),
            ("own",         "I already own it"),
        ],
    },
    {
        # Quantified as a % of the person's total investing money, so it's
        # concrete instead of vague -- but still no private dollar figures.
        "key": "position_size",
        "prompt": "If you invest, how big would this be within your investments?",
        "options": [
            ("small",  "A small slice -- under about 5%"),
            ("medium", "A meaningful chunk -- about 5% to 20%"),
            ("large",  "A large part -- over about 20%"),
        ],
    },
    {
        "key": "goal",
        "prompt": "What matters most to you here?",
        "options": [
            ("growth",     "Growing my money over time"),
            ("income",     "Steady income / dividends"),
            ("understand", "Just understanding the company"),
        ],
    },
]


@dataclass
class UserProfile:
    """The six things we learn about the person. Defaults are the safe,
    middle-of-the-road choice, used if an answer is ever missing."""
    experience: str = "new"
    drawdown_reaction: str = "hold"
    horizon: str = "long"
    ownership: str = "none"
    position_size: str = "small"
    goal: str = "growth"

    def to_dict(self) -> dict:
        return asdict(self)


def build_user_profile(answers: dict) -> UserProfile:
    """Turn a dict of answers into a validated UserProfile.
    This is what the WEBSITE will call later (form values arrive as a dict).
    Any missing or unrecognised answer quietly falls back to the default."""
    profile = UserProfile()
    for question in PROFILE_QUESTIONS:
        key = question["key"]
        allowed = [value for value, _label in question["options"]]
        if answers.get(key) in allowed:
            setattr(profile, key, answers[key])
    return profile


def ask_user_profile_cli() -> UserProfile:
    """Interactive terminal version -- asks the questions one at a time.
    Great for testing today, before the website exists."""
    answers = {}
    print("\nA few quick questions so we can tailor the advice (type a number).\n")
    for question in PROFILE_QUESTIONS:
        print(question["prompt"])
        options = question["options"]
        for i, (_value, label) in enumerate(options, start=1):
            print(f"  {i}. {label}")
        choice = _read_choice(len(options))
        answers[question["key"]] = options[choice - 1][0]
        print()
    return build_user_profile(answers)


def _read_choice(n_options: int) -> int:
    """Read a number 1..n from the user, re-asking politely on bad input."""
    while True:
        raw = input(f"Your choice (1-{n_options}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= n_options:
            return int(raw)
        print("  Please type one of the numbers shown.")


# =========================================================================
# PART 3  --  FETCH THE 10-K FROM SEC EDGAR
# =========================================================================

def _get_section(tenk, possible_names) -> str:
    """Try several possible attribute names and return the first real text.
    (edgartools attribute names can vary slightly by version, so we're
    defensive rather than assuming one exact name.)"""
    for name in possible_names:
        text = getattr(tenk, name, None)
        if isinstance(text, str) and text.strip():
            return text
    return ""


def fetch_10k(ticker: str, identity: str) -> dict:
    """Fetch the latest 10-K for `ticker` from SEC EDGAR.
    Returns a dict with company info, filing info, the key TEXT sections
    (for the narrative agents) and a live FINANCIALS object (for the
    numbers agent).

    The full filing is still available inside the returned `tenk` object --
    we just pull out the three most useful sections for convenience."""
    set_identity(identity)          # SEC requires a name + email on every request

    ticker = ticker.strip().upper()
    company = Company(ticker)

    filings = company.get_filings(form="10-K")
    if filings is None or len(filings) == 0:
        raise LookupError(
            f"No 10-K found for '{ticker}'.  Note: SEC EDGAR only covers "
            f"US-listed companies, so foreign tickers won't work here."
        )

    filing = filings.latest()
    tenk = filing.obj()             # a typed TenK object with sections + financials

    # --- the NARRATIVE half (text) -> goes to 10-K & Market Sense agents ---
    sections = {
        "business":     _get_section(tenk, ["business_description", "business"]),
        "risk_factors": _get_section(tenk, ["risk_factors"]),
        "mda":          _get_section(tenk, ["mda", "management_discussion"]),
    }

    # --- the NUMBERS half -> goes to the Risk & Research agent ---
    try:
        financials = company.get_financials()      # multi-year statements
    except Exception:
        financials = getattr(tenk, "financials", None)

    return {
        "company": {
            "ticker": ticker,
            "name":   getattr(company, "name", ticker),
            "cik":    getattr(company, "cik", None),
        },
        "filing": {
            "form":        "10-K",
            "filing_date": str(getattr(filing, "filing_date", "")),
            "period":      str(getattr(filing, "period_of_report", "")),
            "accession":   str(getattr(filing, "accession_no", "")),
        },
        "sections":   sections,     # text, for the narrative agents
        "financials": financials,   # live object, for the numbers agent
        "tenk":       tenk,         # full object, in case an agent wants more
    }


# =========================================================================
# PART 4  --  THE COORDINATOR  (ties it all together)
# =========================================================================

# The SEC asks every program to identify itself with a real name and email.
# You can also override this without editing the file:
#     export EDGAR_IDENTITY="Jiayu Zhu zhujiayu878@gmail.com"
DEFAULT_IDENTITY = os.environ.get("EDGAR_IDENTITY", "Jiayu Zhu zhujiayu878@gmail.com")


def run_coordinator(ticker: str,
                    profile_answers: dict | None = None,
                    identity: str = DEFAULT_IDENTITY) -> dict:
    """The coordinator's whole job in one call.
      - If `profile_answers` is given (from a web form), use it.
      - Otherwise, ask the questions in the terminal.
    Returns ONE `context` dict that every downstream agent reads from."""
    if profile_answers is None:
        profile = ask_user_profile_cli()
    else:
        profile = build_user_profile(profile_answers)

    print(f"\nFetching the latest 10-K for {ticker.upper()} from SEC EDGAR ...")
    filing_data = fetch_10k(ticker, identity)

    # The single shared hand-off object for the three analysis agents.
    context = {"profile": profile.to_dict(), **filing_data}
    return context


def preview_context(context: dict) -> None:
    """Print a short, readable summary so you can SEE that it worked."""
    company, filing, sections = context["company"], context["filing"], context["sections"]
    print("\n" + "=" * 60)
    print(f"  {company['name']}  ({company['ticker']})")
    print(f"  10-K filed {filing['filing_date']}  |  fiscal period {filing['period']}")
    print("=" * 60)
    print("  Profile:", context["profile"])
    print("\n  Text sections pulled (size in characters):")
    for name, text in sections.items():
        print(f"    - {name:13s} {len(text):>9,} chars")
    ready = "ready" if context["financials"] is not None else "NOT found"
    print(f"\n  Financials object: {ready}")
    print("  (These pieces now get handed to the three analysis agents.)\n")


if __name__ == "__main__":
    print("Coordinator agent  --  The Overwhelmed Everyday Investor\n")
    ticker = ask_ticker_cli()
    context = run_coordinator(ticker)
    preview_context(context)
