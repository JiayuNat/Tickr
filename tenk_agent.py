import os
import boto3
import json
from dotenv import load_dotenv

load_dotenv()  # pulls AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION from .env


def analyze_10k_report(context: dict) -> dict:
    """Analyzes the 10-K report to summarize financial health and management claims.

    Args:
        context: The context dictionary containing the 10-K sections.

    Returns:
        A dictionary with structured summaries of financial health and
        management claims, each shaped like:
            {"signal": "positive|neutral|negative|mixed",
            "summary": "1-2 plain sentences",
            "key_points": ["...", "..."],
            "watch_items": ["...", "..."]}
    """
    try:
        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )
        claude_model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    except Exception as e:
        error_block = {
            "signal": "error",
            "summary": f"Error initializing AWS Bedrock client: {e}. Please ensure AWS credentials are correctly configured.",
            "key_points": [],
            "watch_items": [],
        }
        return {"financial_health": error_block, "management_claims": error_block}

    sections = context.get('sections', {})
    business_section = sections.get('business', '')
    mda_section = sections.get('mda', '')

    # Shared system prompt: sets the tone/format rules once instead of repeating
    # them in every user prompt, and forces clean, parseable JSON back.
    SYSTEM_PROMPT = """You are a financial analyst explaining a 10-K filing to a busy,
non-expert retail investor. Be accurate and specific (cite real numbers, trends, and
terms from the text) but write in plain English -- avoid jargon, and explain any
technical term you must use.

Respond with ONLY a single JSON object, no markdown fences, no preamble, matching
exactly this schema:
{
  "signal": "positive" | "neutral" | "negative" | "mixed",
  "summary": "1-2 sentence plain-English takeaway",
  "key_points": ["3-5 short bullet strings, each one concrete fact or trend"],
  "watch_items": ["0-3 short bullet strings on risks or things to monitor -- empty list if none"]
}"""

    def get_bedrock_completion(user_prompt: str) -> dict:
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 700,
                "temperature": 0.3,  # low temperature: keep summaries consistent/factual, not creative
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            })
            response = bedrock_client.invoke_model(
                body=body,
                modelId=claude_model_id,
                accept='application/json',
                contentType='application/json'
            )
            response_body = json.loads(response.get('body').read())
            raw_text = response_body['content'][0]['text'].strip()

            # Defensive cleanup in case the model wraps JSON in ```json fences anyway
            if raw_text.startswith("```"):
                raw_text = raw_text.strip("`")
                if raw_text.lower().startswith("json"):
                    raw_text = raw_text[4:].strip()

            return json.loads(raw_text)

        except json.JSONDecodeError:
            # Model didn't return valid JSON -- fall back gracefully instead of crashing
            return {
                "signal": "unknown",
                "summary": raw_text if 'raw_text' in locals() else "Could not parse model response.",
                "key_points": [],
                "watch_items": [],
            }
        except Exception as e:
            return {
                "signal": "error",
                "summary": f"Error calling Bedrock API: {e}",
                "key_points": [],
                "watch_items": [],
            }

    financial_health_prompt = f"""Based on the Management's Discussion and Analysis (MDA)
section below from a 10-K report, summarize the company's financial health: revenue trends,
profitability, costs, and overall trajectory.

MDA section:
{mda_section}"""

    management_claims_prompt = f"""Based on the Business and MDA sections below from a 10-K
report, summarize what management says the company does, their stated strategy, and their
outlook for the future. Distinguish between what already happened (results) and what
management is projecting (forward-looking claims).

Business section:
{business_section}

MDA section:
{mda_section}"""

    empty_block = {
        "signal": "unknown",
        "summary": "No relevant section was provided in the 10-K data.",
        "key_points": [],
        "watch_items": [],
    }

    financial_health = get_bedrock_completion(financial_health_prompt) if mda_section else empty_block
    management_claims = get_bedrock_completion(management_claims_prompt) if (business_section or mda_section) else empty_block

    return {
        "financial_health": financial_health,
        "management_claims": management_claims,
    }


def save_analysis(results: dict, ticker: str, output_dir: str = "outputs") -> dict:
    """Saves the analysis to disk as JSON so other Python files can read it
    later without re-calling Bedrock. Saves financial_health and
    management_claims as separate files (for easy standalone reuse) plus one
    combined file (for convenience).

    Returns a dict of the file paths written.
    """
    os.makedirs(output_dir, exist_ok=True)
    ticker = ticker.strip().upper()

    paths = {
        "financial_health": os.path.join(output_dir, f"{ticker}_financial_health.json"),
        "management_claims": os.path.join(output_dir, f"{ticker}_management_claims.json"),
        "combined": os.path.join(output_dir, f"{ticker}_tenk_analysis.json"),
    }

    with open(paths["financial_health"], "w") as f:
        json.dump(results["financial_health"], f, indent=2)

    with open(paths["management_claims"], "w") as f:
        json.dump(results["management_claims"], f, indent=2)

    with open(paths["combined"], "w") as f:
        json.dump(results, f, indent=2)

    return paths


def load_analysis(ticker: str, output_dir: str = "outputs") -> dict:
    """Reads a previously saved combined analysis back from disk.
    This is what OTHER files (e.g. the advisor agent) should call if they
    just want the results without re-running the 10-K agent."""
    ticker = ticker.strip().upper()
    path = os.path.join(output_dir, f"{ticker}_tenk_analysis.json")
    with open(path, "r") as f:
        return json.load(f)


def print_summary_block(title: str, block: dict) -> None:
    """Pretty-prints one structured block to the terminal for human reading."""
    print(f"\n{title}")
    print(f"  Signal: {block.get('signal', 'unknown').upper()}")
    print(f"  {block.get('summary', '')}")
    if block.get('key_points'):
        print("  Key points:")
        for point in block['key_points']:
            print(f"    - {point}")
    if block.get('watch_items'):
        print("  Watch for:")
        for item in block['watch_items']:
            print(f"    - {item}")


if __name__ == "__main__":
    from coordinator import run_coordinator

    ticker = input("Which ticker do you want to analyze? (e.g. AAPL, NVDA): ").strip().upper()

    print(f"\nFetching the 10-K and building context for {ticker}...")
    # Using fixed profile answers here so this test run doesn't stop to ask the
    # 6 profile questions -- swap in ask_user_profile_cli() flow if you want that.
    context = run_coordinator(ticker, profile_answers=None)

    print(f"\nRunning tenk_agent analysis for {ticker}...")
    analysis_results = analyze_10k_report(context)

    # Save to disk so other scripts can read this later without re-running Bedrock
    saved_paths = save_analysis(analysis_results, ticker=ticker)
    print(f"\nSaved to: {saved_paths['combined']}")

    # Print now so you can eyeball the output immediately
    print_summary_block("Financial Health", analysis_results["financial_health"])
    print_summary_block("Management Claims", analysis_results["management_claims"])