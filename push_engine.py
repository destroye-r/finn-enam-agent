import os
import io
import re
import json
import time
import requests
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from pypdf import PdfReader
from duckduckgo_search import DDGS
from google import genai
from google.genai import types

# Initialize Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Smart Model Fallback Chain: If primary model is 503/429, cascade immediately
MODEL_CASCADE = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

CACHE_DIR = "./data_cache"
RAW_FILINGS_PATH = os.path.join(CACHE_DIR, "corporate_filings.json")
ENRICHED_FILINGS_PATH = os.path.join(CACHE_DIR, "enriched_push_feed.json")


def generate_with_fallback(contents, config: types.GenerateContentConfig):
    """
    Tries models in MODEL_CASCADE sequentially so a 503 traffic spike
    or 429 rate-limit on one model never crashes the pipeline.
    """
    last_err = None
    for model_name in MODEL_CASCADE:
        for attempt in range(2):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_sec = 2 * (attempt + 1)
                    print(f"    [~] {model_name} busy (503/429). Switching/retrying in {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    raise e
    raise last_err


# =====================================================================
# 1. Pydantic Schemas (Aligned with Reference Doc Categories)
# =====================================================================
CategoryType = Literal[
    "results", "M&A", "fund_raise", "management_change", "litigation",
    "insider_trading", "earnings_call", "investor_meetings", "credit_rating",
    "dividend", "bonus_split", "capex", "press_release", "postal_ballot",
    "agm_egm", "pledging", "unclassified"
]

class ClassifiedFiling(BaseModel):
    filing_index: int
    category: CategoryType
    is_material: bool = Field(description="True if this can move the stock or matters to an institutional investor; False for routine paperwork.")
    priority_score: int = Field(description="1 (routine/ignore) to 5 (critical: results, M&A, CEO/Board change, QIP, major order/molecule).")
    headline_summary: str = Field(description="Crisp, institutional-grade 1-line summary (max 18 words). No fluff.")
    person_involved: Optional[str] = Field(None, description="If management_change, full name of incoming/outgoing person.")
    role_change: Optional[str] = Field(None, description="e.g., 'Appointed Non-Executive Director' or 'Retired as Chairman'")

class BatchClassificationResult(BaseModel):
    items: List[ClassifiedFiling]

class FinancialMetric(BaseModel):
    metric: str = Field(description="Revenue, EBITDA, PBT, or PAT")
    current_qtr: str = Field(description="Value in Cr with unit, e.g., '₹41,764 Cr'")
    qoq_pct: str = Field(description="Quarter-on-Quarter % change, e.g., '+4.2%' or '-1.5%'")
    yoy_pct: str = Field(description="Year-on-Year % change, e.g., '+12.8%'")

class QuarterlyResultsCard(BaseModel):
    period: str = Field(description="e.g., 'Q2 FY27 (Ended Sep 30, 2026)'")
    basis: str = Field(description="'Consolidated' or 'Standalone'")
    metrics: List[FinancialMetric]
    key_takeaway: str = Field(description="1-sentence analytical takeaway on margin expansion/compression or growth.")


# =====================================================================
# 2. Step 1: Deduplication & Zero-Cost Regex Pre-Filter
# =====================================================================
def deduplicate_and_prefilter(filings: list[dict]) -> tuple[list[dict], list[dict]]:
    seen_keys = set()
    unique_filings = []
    
    for f in filings:
        dedup_key = (f["ticker"], f["subject"], f["details"][:120])
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        unique_filings.append(f)

    routine_preclassified = []
    needs_llm = []

    routine_patterns = [
        r"loss of share certificate",
        r"copy of newspaper publication",
        r"esop/esos/esps",
        r"allotment.*esop",
        r"employee stock option",
        r"certificate under sebi \(depositories",
        r"investor education and protection fund",
        r"auditors on financial statements.*with\s*udin"
    ]
    combined_regex = re.compile("|".join(routine_patterns), re.IGNORECASE)

    for idx, f in enumerate(unique_filings):
        f["id"] = f"filing_{idx}"
        text_blob = f"{f['subject']} {f['details']}"
        
        if combined_regex.search(text_blob):
            f.update({
                "category": "unclassified",
                "is_material": False,
                "priority_score": 1,
                "headline_summary": f"Routine statutory filing: {f['subject']}",
                "enrichment": None
            })
            routine_preclassified.append(f)
        else:
            needs_llm.append(f)

    print(f"[*] Deduplicated {len(filings)} -> {len(unique_filings)} filings.")
    print(f"[*] Zero-cost regex caught {len(routine_preclassified)} routine items. Sending {len(needs_llm)} to LLM in chunks.")
    return routine_preclassified, needs_llm


# =====================================================================
# 3. Step 2: Fast LLM Batch Classification (Chunked into 20s)
# =====================================================================
def classify_filings_batch(filings_to_classify: list[dict], chunk_size: int = 20) -> list[dict]:
    system_instruction = (
        "You are an institutional public-markets analyst at ENAM Holdings. "
        "Classify each NSE corporate announcement into one of the exact allowed categories. "
        "Rules:\n"
        "- Board meeting outcomes submitting unaudited financial results OR press releases on quarterly financial results -> 'results' (priority 5).\n"
        "- CEO/MD/Chairman/Director appointments or retirements -> 'management_change' (priority 4-5). Extract person_involved and role_change.\n"
        "- Divestments, subsidiary strike-offs, IPO of subsidiary -> 'M&A' (priority 4-5).\n"
        "- QIP allotments, NCD issuances -> 'fund_raise' (priority 4).\n"
        "- NCLAT / insolvency orders -> 'litigation' (priority 4).\n"
        "- New product launches (like novel molecules), major order wins -> 'press_release' (priority 4).\n"
        "- Concall transcripts or audio recordings -> 'earnings_call' (priority 3).\n"
        "- Schedule of 1-on-1 investor conferences/meetings or concall intimations -> 'investor_meetings' or 'earnings_call' (priority 2, is_material=False)."
    )

    classified_map = {}

    for start in range(0, len(filings_to_classify), chunk_size):
        chunk = filings_to_classify[start:start + chunk_size]
        prompt_items = [
            {
                "filing_index": start + idx,
                "ticker": f["ticker"],
                "subject": f["subject"],
                "details": f["details"]
            }
            for idx, f in enumerate(chunk)
        ]

        print(f"  -> Classifying batch {start // chunk_size + 1} ({len(chunk)} items)...")
        response = generate_with_fallback(
            contents=f"Classify these corporate filings:\n{json.dumps(prompt_items, indent=2)}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=BatchClassificationResult,
                temperature=0.1
            )
        )

        parsed: BatchClassificationResult = response.parsed
        if parsed and parsed.items:
            for item in parsed.items:
                classified_map[item.filing_index] = item
        time.sleep(1)

    for idx, f in enumerate(filings_to_classify):
        c = classified_map.get(idx)
        if c:
            f["category"] = c.category
            f["is_material"] = c.is_material
            f["priority_score"] = c.priority_score
            f["headline_summary"] = c.headline_summary
            f["person_involved"] = c.person_involved
            f["role_change"] = c.role_change
            f["enrichment"] = None
        else:
            f["category"] = "unclassified"
            f["is_material"] = False
            f["priority_score"] = 1
            f["headline_summary"] = f["subject"]
            f["enrichment"] = None

    return filings_to_classify


# =====================================================================
# 4. Step 3: Deep PDF Extraction for Quarterly Results
# =====================================================================
def download_pdf_text(pdf_url: str, max_pages: int = 8) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Referer": "https://www.nseindia.com/"
    }
    resp = requests.get(pdf_url, headers=headers, timeout=15)
    resp.raise_for_status()
    
    reader = PdfReader(io.BytesIO(resp.content))
    extracted_pages = []
    for i, page in enumerate(reader.pages[:max_pages]):
        text = page.extract_text() or ""
        if text.strip():
            extracted_pages.append(f"--- PAGE {i+1} ---\n{text}")
    return "\n".join(extracted_pages)


def enrich_results_filing(filing: dict) -> Optional[dict]:
    details_lower = filing["details"].lower()
    if "schedule" in details_lower or "to consider and approve" in details_lower or "record date" in filing["subject"].lower():
        return None

    print(f"  [PDF Results Extractor] Parsing {filing['ticker']} -> {filing['pdf_url']} ...")
    try:
        pdf_text = download_pdf_text(filing["pdf_url"], max_pages=10)
        if len(pdf_text) < 200:
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"}
            pdf_bytes = requests.get(filing["pdf_url"], headers=headers, timeout=15).content
            content_payload = [
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                "Extract Consolidated Quarterly Financial Results: Revenue, EBITDA, PBT, and PAT for the current quarter, along with calculated QoQ (%) and YoY (%) changes."
            ]
        else:
            content_payload = (
                f"Company: {filing['company_name']} ({filing['ticker']})\n"
                f"Filing Text:\n{pdf_text[:25000]}\n\n"
                "Extract Consolidated Quarterly Results (Revenue from operations, EBITDA/Operating Profit, PBT, PAT). "
                "Compute exact QoQ % (vs previous quarter) and YoY % (vs same quarter last year). Convert figures to ₹ Cr."
            )

        resp = generate_with_fallback(
            contents=content_payload,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuarterlyResultsCard,
                temperature=0.0
            )
        )
        return resp.parsed.model_dump()
    except Exception as e:
        print(f"  [!] Failed PDF extraction for {filing['ticker']}: {e}")
        return None


# =====================================================================
# 5. Step 4: Executive Bio Enrichment for Management Changes
# =====================================================================
def enrich_management_change(filing: dict) -> Optional[dict]:
    person = filing.get("person_involved")
    company = filing.get("company_name")
    
    pdf_snippet = ""
    try:
        pdf_snippet = download_pdf_text(filing["pdf_url"], max_pages=2)[:3000]
    except Exception:
        pass

    search_query = f"{person or ''} {company} executive profile career biography LinkedIn".strip()
    print(f"  [Bio Enricher] Searching background for {company} ({person or 'Executive'})...")
    
    search_snippets = ""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=3))
            search_snippets = "\n".join([f"- {r.get('title', '')}: {r.get('body', '')}" for r in results])
    except Exception as e:
        print(f"  [!] Search fallback note: {e}")

    prompt = (
        f"Company: {company}\n"
        f"Announcement Details: {filing['details']}\n"
        f"PDF Extract: {pdf_snippet}\n"
        f"Web Search Snippets:\n{search_snippets}\n\n"
        "Return a JSON object with keys: "
        "'executive_name' (string), 'role_transition' (string), and 'background_bio' "
        "(a crisp 2-sentence institutional summary of who they are, their prior roles/companies, and track record)."
    )

    try:
        resp = generate_with_fallback(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            )
        )
        return json.loads(resp.text)
    except Exception as e:
        print(f"  [!] Bio enrichment failed: {e}")
        return None


# =====================================================================
# Main Pipeline Execution
# =====================================================================
def build_push_feed():
    with open(RAW_FILINGS_PATH, "r") as f:
        raw_filings = json.load(f)

    # 1. Deduplicate & Pre-filter routine items
    routine_items, to_classify = deduplicate_and_prefilter(raw_filings)

    # 2. Classify non-routine items via LLM in chunks of 20
    classified_items = classify_filings_batch(to_classify, chunk_size=20)

    # 3. Run specialized enrichers on high-signal categories
    for item in classified_items:
        if item["category"] == "results" and item["priority_score"] >= 4:
            res_card = enrich_results_filing(item)
            if res_card:
                item["enrichment"] = {"type": "quarterly_results", "data": res_card}
            time.sleep(1)

        elif item["category"] == "management_change":
            bio_card = enrich_management_change(item)
            if bio_card:
                item["enrichment"] = {"type": "management_bio", "data": bio_card}
            time.sleep(1)

    # 4. Merge and sort: Priority descending, then broadcast_time descending
    all_items = classified_items + routine_items
    all_items.sort(key=lambda x: (x.get("priority_score", 1), x.get("broadcast_time", "")), reverse=True)

    with open(ENRICHED_FILINGS_PATH, "w") as f:
        json.dump(all_items, f, indent=2)

    print(f"\n[+] Push Feed Built Successfully! Saved {len(all_items)} items to {ENRICHED_FILINGS_PATH}")


if __name__ == "__main__":
    build_push_feed()