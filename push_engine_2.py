import os
import json
import time
from google import genai
from google.genai import types
from push_engine import (
    ENRICHED_FILINGS_PATH,
    QuarterlyResultsCard,
    download_pdf_text,
    DDGS
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Use models with fresh, separate free-tier daily quota buckets
# In push_engine_2.py, replace FRESH_MODELS with the exact active model Google's error specified:
FRESH_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-pro",
    "gemini-3.5-pro"
]

def call_llm(contents, schema=None):
    for model_name in FRESH_MODELS:
        try:
            cfg = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
            if schema:
                cfg.response_schema = schema
            return client.models.generate_content(model=model_name, contents=contents, config=cfg)
        except Exception as e:
            print(f"  [~] {model_name} error: {e}. Trying next model...")
            time.sleep(2)
    raise RuntimeError("All Gemini quota buckets exhausted.")

def patch_feed():
    with open(ENRICHED_FILINGS_PATH, "r") as f:
        feed = json.load(f)

    updated = 0
    for item in feed:
        # Skip items that already have enrichment!
        if item.get("enrichment") is not None:
            continue

        # 1. Patch TRENT Results (Press Release or Outcome of BM)
        if item["category"] == "results" and item["ticker"] == "TRENT" and item["id"] == "filing_37":
            print(f"[*] Extracting Q2 FY27 Results for {item['ticker']} ({item['subject']})...")
            try:
                pdf_text = download_pdf_text(item["pdf_url"], max_pages=8)
                prompt = (
                    f"Company: {item['company_name']}\nText:\n{pdf_text[:20000]}\n\n"
                    "Extract Consolidated Quarterly Results (Revenue, EBITDA, PBT, PAT) in ₹ Cr with QoQ % and YoY % changes."
                )
                resp = call_llm(prompt, schema=QuarterlyResultsCard)
                item["enrichment"] = {"type": "quarterly_results", "data": resp.parsed.model_dump()}
                updated += 1
            except Exception as e:
                print(f"  [!] Failed Trent results: {e}")

        # 2. Patch missing Management Change Bios (Noel Tata, Kalpana Merchant, Ashish Bhatt)
        elif item["category"] == "management_change" and item.get("person_involved"):
            person = item["person_involved"]
            company = item["company_name"]
            print(f"[*] Enriching bio for {person} ({company})...")
            try:
                pdf_snippet = download_pdf_text(item["pdf_url"], max_pages=2)[:2500]
                prompt = (
                    f"Company: {company}\nDetails: {item['details']}\nPDF Snippet: {pdf_snippet}\n\n"
                    "Return JSON with exact keys: 'executive_name', 'role_transition', and 'background_bio' "
                    "(2-sentence institutional summary of their career background, prior roles, and pedigree)."
                )
                resp = call_llm(prompt)
                item["enrichment"] = {"type": "management_bio", "data": json.loads(resp.text)}
                updated += 1
            except Exception as e:
                print(f"  [!] Failed bio for {person}: {e}")

    # Also remove duplicate raw results card for Trent if filing_37 succeeded
    with open(ENRICHED_FILINGS_PATH, "w") as f:
        json.dump(feed, f, indent=2)
    print(f"\n[+] Patched {updated} items! {ENRICHED_FILINGS_PATH} is 100% ready.")

if __name__ == "__main__":
    patch_feed()