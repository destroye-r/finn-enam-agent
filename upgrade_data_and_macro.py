import os
import json
import time
import xml.etree.ElementTree as ET
import requests
import yfinance as yf
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types

try:
    from src.data_pipeline import PORTFOLIO_TICKERS
except ModuleNotFoundError:
    from data_pipeline import PORTFOLIO_TICKERS

try:
    from src.push_engine import download_pdf_text
    from src.push_engine import DDGS
except ModuleNotFoundError:
    from push_engine import download_pdf_text, DDGS

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

CACHE_DIR = "./data_cache"
ENRICHED_FILINGS_PATH = os.path.join(CACHE_DIR, "enriched_push_feed.json")
EOD_METRICS_PATH = os.path.join(CACHE_DIR, "eod_metrics.json")
MACRO_PULSE_PATH = os.path.join(CACHE_DIR, "macro_pulse.json")

MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]

# =====================================================================
# 1. Strict Schema to Fix "undefined" in Management Change Bios
# =====================================================================
class ManagementBioCard(BaseModel):
    executive_name: str = Field(description="Full name of the executive/director, e.g., 'Noel N Tata'")
    role_transition: str = Field(description="Clear role transition, e.g., 'Retired as Chairman of Trent Limited'")
    background_bio: str = Field(description="2-sentence institutional biography covering leadership roles across Tata Group/Indian Inc, board memberships, and track record.")

def call_gemini_structured(prompt: str, schema):
    for m in MODELS:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1
                )
            )
            if resp.parsed:
                return resp.parsed.model_dump()
            return json.loads(resp.text)
        except Exception as e:
            print(f"  [~] Model {m} note: {str(e)[:80]}...")
            time.sleep(2)
    return None

def fix_all_management_bios():
    print("\n[*] Step 1: Auditing & Fixing Management Change Bios in enriched_push_feed.json...")
    with open(ENRICHED_FILINGS_PATH, "r", encoding="utf-8") as f:
        feed = json.load(f)

    fixed_count = 0
    for item in feed:
        if item.get("category") == "management_change":
            # Check if enrichment is missing OR has missing/undefined keys
            enr = item.get("enrichment")
            data = enr.get("data") if isinstance(enr, dict) else None
            if isinstance(data, list) and len(data) > 0:
                data = data[0]

            needs_fix = (
                not data
                or not isinstance(data, dict)
                or not data.get("executive_name")
                or not data.get("background_bio")
                or "undefined" in str(data.get("executive_name", "")).lower()
            )

            if needs_fix:
                person = item.get("person_involved") or "Executive"
                company = item.get("company_name")
                print(f"  -> Fixing bio for {person} ({company})...")

                pdf_snippet = ""
                try:
                    pdf_snippet = download_pdf_text(item["pdf_url"], max_pages=2)[:2500]
                except Exception:
                    pass

                search_snippets = ""
                try:
                    with DDGS() as ddgs:
                        q = f"{person} {company} biography corporate career board director"
                        res = list(ddgs.text(q, max_results=3))
                        search_snippets = "\n".join([f"- {r.get('title','')}: {r.get('body','')}" for r in res])
                except Exception:
                    pass

                prompt = (
                    f"Company: {company}\n"
                    f"Person: {person}\n"
                    f"Role Change: {item.get('role_change') or item.get('subject')}\n"
                    f"Filing Details: {item.get('details')}\n"
                    f"PDF Extract: {pdf_snippet}\n"
                    f"Search Info: {search_snippets}\n\n"
                    "Generate an institutional Executive Pedigree summary. Even if not on LinkedIn, use public corporate governance knowledge and filing context."
                )
                bio_data = call_gemini_structured(prompt, ManagementBioCard)
                if bio_data:
                    item["enrichment"] = {"type": "management_bio", "data": bio_data}
                    fixed_count += 1
                    time.sleep(1)

    with open(ENRICHED_FILINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(feed, f, indent=2)
    print(f"[+] Management Bios Verified & Fixed ({fixed_count} updated)!")


# =====================================================================
# 2. Expand EOD Metrics to Include 30-Day Historical Series for Charts
# =====================================================================
def upgrade_eod_30d_history(tickers: list[str]):
    print("\n[*] Step 2: Fetching 1-Month (30 Trading Days) Price & Volume Series for Charts...")
    clean_tickers = [t for t in tickers if t != "TATAMOTORS"]
    if "M&M" not in clean_tickers:
        clean_tickers.append("M&M")

    yf_symbols = [f"{t}.NS" for t in clean_tickers]
    data = yf.download(yf_symbols, period="2mo", interval="1d", group_by="ticker", progress=False)

    results = []
    for ticker in clean_tickers:
        sym = f"{ticker}.NS"
        try:
            df = data[sym].dropna()
            if len(df) < 21:
                continue
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            month_ago = df.iloc[-22] if len(df) >= 22 else df.iloc[0]

            close_curr = float(latest["Close"])
            close_prev = float(prev["Close"])
            close_1m = float(month_ago["Close"])
            vol_curr = float(latest["Volume"])
            vol_sma_20 = float(df["Volume"].iloc[-21:-1].mean())

            daily_ret = ((close_curr - close_prev) / close_prev) * 100.0
            month_ret = ((close_curr - close_1m) / close_1m) * 100.0
            vol_ratio = (vol_curr / vol_sma_20) if vol_sma_20 > 0 else 0.0

            # Extract last 25 trading days (~1.2 calendar months) for sparklines & charts
            recent_df = df.tail(25)
            history_30d = [
                {
                    "date": idx.strftime("%d %b"),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"])
                }
                for idx, row in recent_df.iterrows()
            ]

            results.append({
                "ticker": ticker,
                "date": df.index[-1].strftime("%Y-%m-%d"),
                "close": round(close_curr, 2),
                "prev_close": round(close_prev, 2),
                "daily_return_pct": round(daily_ret, 2),
                "month_return_pct": round(month_ret, 2),
                "volume": int(vol_curr),
                "vol_sma_20": int(vol_sma_20),
                "vol_ratio": round(vol_ratio, 2),
                "triggered_alert": abs(daily_ret) >= 3.0 or vol_ratio >= 2.0,
                "history_30d": history_30d
            })
        except Exception as e:
            print(f"  [!] Skipped {ticker}: {e}")

    with open(EOD_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved 30-Day Price & Volume Chart Series for {len(results)} stocks!")


# =====================================================================
# 3. Macro Indicator, Oil Shock & RBI / Govt Sovereign Pulse Engine
# =====================================================================
class MacroNewsItem(BaseModel):
    source: str = Field(description="'RBI Official', 'Govt of India (PIB)', or 'Global Macro Shock'")
    title: str
    date: str
    category: str = Field(description="e.g., 'Monetary Policy & Liquidity', 'Oil & Commodity Shock', 'Sectoral Policy', 'FX & Rates'")
    impact_severity: str = Field(description="'High', 'Medium', or 'Monitor'")
    portfolio_transmission: str = Field(description="1-2 sentences explaining how this specifically impacts portfolio sectors (e.g. Banks: HDFCBANK/FEDERALBNK, IT: INFY/PERSISTENT, Chemicals/Paints/Energy: RELIANCE/PIIND, Infra: LT/POLYCAB).")
    affected_tickers: List[str]
    link: str

class MacroSynthesisBatch(BaseModel):
    items: List[MacroNewsItem]

def build_macro_pulse():
    print("\n[*] Step 3: Building Macro Indicators (Brent Oil, USD/INR, US 10Y, VIX) & RBI/PIB Official Feeds...")

    # A. Fetch 1-Month Macro Benchmark Series
    macro_Benchmarks = [
        {"id": "BRENT", "name": "Brent Crude Oil", "sym": "BZ=F", "unit": "$"},
        {"id": "USDINR", "name": "USD / INR", "sym": "INR=X", "unit": "₹"},
        {"id": "US10Y", "name": "US 10Y Treasury", "sym": "^TNX", "unit": "%"},
        {"id": "INDIAVIX", "name": "India VIX", "sym": "^INDIAVIX", "unit": ""},
        {"id": "NIFTY50", "name": "Nifty 50 Index", "sym": "^NSEI", "unit": ""},
    ]

    indicators_data = []
    for b in macro_Benchmarks:
        try:
            hist = yf.Ticker(b["sym"]).history(period="1mo", interval="1d").dropna()
            if len(hist) >= 2:
                curr = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                first = float(hist["Close"].iloc[0])
                d1_pct = ((curr - prev) / prev) * 100.0
                m1_pct = ((curr - first) / first) * 100.0
                series = [
                    {"date": idx.strftime("%d %b"), "value": round(float(val), 2)}
                    for idx, val in hist["Close"].items()
                ]
                indicators_data.append({
                    "id": b["id"],
                    "name": b["name"],
                    "unit": b["unit"],
                    "current": round(curr, 2),
                    "daily_pct": round(d1_pct, 2),
                    "month_pct": round(m1_pct, 2),
                    "series": series
                })
        except Exception as e:
            print(f"  [!] Macro benchmark {b['name']} error: {e}")

    # B. Scrape Official RBI Press Releases RSS & Govt of India PIB RSS
    raw_sovereign_items = []
    rss_sources = [
        ("RBI Official", "https://rbi.org.in/pressreleases_rss.xml"),
        ("RBI Notifications", "https://rbi.org.in/notifications_rss.xml"),
        ("Govt of India (PIB)", "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"),
    ]

    headers = {"User-Agent": "Mozilla/5.0"}
    for label, rss_url in rss_sources:
        try:
            r = requests.get(rss_url, headers=headers, timeout=10)
            if r.status_code == 200:
                root = ET.fromstring(r.content)
                for item in root.findall(".//item")[:6]:
                    title = item.findtext("title") or ""
                    link = item.findtext("link") or "https://www.rbi.org.in"
                    pub_date = item.findtext("pubDate") or "Oct 2026"
                    desc = item.findtext("description") or ""
                    if title:
                        raw_sovereign_items.append({
                            "source": label,
                            "title": title.strip(),
                            "date": pub_date[:16].strip(),
                            "description": desc[:300].strip(),
                            "link": link.strip()
                        })
        except Exception as e:
            print(f"  [~] RSS note for {label}: {e}")

    # Also pull official sovereign/macro/oil policy releases over the past month via search if RSS is sparse
    try:
        with DDGS() as ddgs:
            queries = [
                "site:rbi.org.in monetary policy repo rate liquidity circular 2026",
                "site:pib.gov.in Ministry of Finance Commerce PLI semiconductor pharma infrastructure 2026",
                "Brent crude oil shock OPEC EIA US Federal Reserve yield impact India macro October 2026"
            ]
            for q in queries:
                for r in list(ddgs.text(q, max_results=3)):
                    raw_sovereign_items.append({
                        "source": "RBI / PIB / Sovereign Feed",
                        "title": r.get("title", ""),
                        "date": "Oct 2026 (30D Macro Window)",
                        "description": r.get("body", ""),
                        "link": r.get("href", "https://www.rbi.org.in")
                    })
    except Exception:
        pass

    prompt = (
        f"Current 30-Day Macro Benchmarks: {json.dumps(indicators_data)}\n"
        f"Raw Sovereign, RBI, PIB, and Global Macro Items:\n{json.dumps(raw_sovereign_items[:15], indent=2)}\n\n"
        "Select and synthesize the 8 most material macro, RBI regulatory, sovereign policy, and global commodity/rate shock developments "
        "from the last 30 days. Map each item directly to how it transmits into our Indian institutional portfolio tickers: "
        "RELIANCE, HDFCBANK, INFY, SUNPHARMA, LT, POLYCAB, PERSISTENT, TRENT, DIXON, FEDERALBNK, PIIND, KAYNES, NEULANDLAB, CDSL, ANGELONE."
    )

    synthesized = call_gemini_structured(prompt, MacroSynthesisBatch)
    macro_feed = synthesized.get("items", []) if synthesized else []

    payload = {
        "updated_at": "30-Day Rolling Sovereign & Macro Horizon",
        "indicators": indicators_data,
        "macro_feed": macro_feed
    }

    with open(MACRO_PULSE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[+] Saved Macro Pulse ({len(indicators_data)} benchmarks, {len(macro_feed)} sovereign/macro briefs) to {MACRO_PULSE_PATH}!")


if __name__ == "__main__":
    fix_all_management_bios()
    upgrade_eod_30d_history(PORTFOLIO_TICKERS)
    build_macro_pulse()