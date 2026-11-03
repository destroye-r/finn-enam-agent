# FINN — Financial Investment & Primary-Signal Intelligence Agent

**Live Production App:** [https://finn-enam-agent.onrender.com](https://finn-enam-agent.onrender.com)  
**Author:** Niket Dhar Diwan  

---

## 1. System Thesis & Architectural Overview

Public-markets investors do not suffer from a lack of information; they suffer from an asymmetric signal-to-noise problem. Every trading day, the National Stock Exchange (NSE) broadcasts hundreds of routine compliance disclosures (investor grievances, newspaper clippings, duplicate share certificates, trading window closures) alongside a handful of thesis-altering corporate events (audited quarterly results, board restructurings, NCLAT orders, capital raises). Simultaneously, secondary market commentary—broker notes and financial media—frequently obscures whether management's narrative matches audited execution.

**FINN** is a mobile-first, primary-source institutional intelligence layer engineered around two tightly coupled pipelines:
1. **The Push Engine (Proactive Signal Filtering & Cross-Domain Fusion):** Collapses ~85% of routine exchange noise, surfaces material corporate actions across configurable time windows (`24H` default, `3D`, `7D`, `30D`), and fuses them with EOD price/volume anomalies (`yfinance` 20D SMA multiples) and sovereign/macro transmissions (RBI, PIB, Brent Crude).
2. **The Pull Engine (On-Demand Primary Research & Verbatim Evidence):** Generates structured fundamental audits for **any NSE ticker** strictly from audited financial statements (Screener.in) and primary exchange PDFs/concall transcripts—excluding broker commentary and secondary news—backed by expandable verbatim evidence drawers (`▸ Show the line`).

---

## 2. File-by-File Architecture & Execution Pipeline

```text
├── app.py                        # FastAPI server, REST routers, Ask FINN copilot & embedded Mobile-First SPA UI
├── pull_engine.py                # Live Screener/NSE PDF scraper, HTML table parser & LLM/Deterministic synthesizer
├── data_pipeline.py              # EOD price/volume (yfinance) & NSE corporate filing ingestion pipeline
├── push_engine.py                # Filing classifier, priority scorer & PDF deep-enrichment engine
├── push_engine_2.py              # Secondary batch enrichment worker for rate-limited / retry queues
├── upgrade_data_and_macro.py     # 30-day history backfiller & RBI/PIB/Macro benchmark aggregator
├── requirements.txt              # Production Python dependencies
└── data_cache/                   # Pre-computed EOD, filing, macro & Pull Research JSON snapshots
    ├── corporate_filings.json    # Raw ingested NSE corporate announcements with broadcast timestamps & PDF links
    ├── enriched_push_feed.json   # Classified, priority-scored (1-5) & Gemini-summarized filing feed
    ├── eod_metrics.json          # 30-day OHLCV histories, 1D moves & 20D volume SMA ratios per portfolio ticker
    ├── macro_pulse.json          # Sovereign & commodity 30D series + RBI/PIB policy transmission mapping
    └── pull_research/            # Cached & newly generated {TICKER}_research.json primary audit reports
```

---

### File 1: `data_pipeline.py` — Market & Exchange Ingestion Layer
Responsible for fetching raw primary market data and exchange announcements for the portfolio universe.

* **EOD Price & Volume Anomaly Calculation (`yfinance`):**
  * Pulls 30-day daily OHLCV histories for each NSE ticker (`{TICKER}.NS`).
  * Computes daily percentage price change ($\Delta P_{\%}$) and the **20-Day Volume Simple Moving Average (20D SMA)** to detect institutional accumulation or distribution:
    $$\Delta P_{\%} = \left( \frac{P_{\text{close}, t} - P_{\text{close}, t-1}}{P_{\text{close}, t-1}} \right) \times 100$$
    $$\text{Vol}_{\text{ratio}, t} = \frac{V_t}{\frac{1}{20}\sum_{i=1}^{20} V_{t-i}}$$
* **NSE Corporate Filing Ingestion:**
  * Queries the NSE Corporate Announcements feed using session-persisted browser headers (`User-Agent`, `Referer`, cookies) to extract the exact `broadcast_time`, `subject`, `details`, and official exchange attachment `pdf_url`, saving raw records to `data_cache/corporate_filings.json`.

---

### File 2: `push_engine.py` (and `push_engine_2.py`) — Noise Collapse & Deep PDF Enrichment
Transforms raw exchange announcements in `corporate_filings.json` into the structured, priority-ranked intelligence feed stored in `enriched_push_feed.json`.

* **Stage 1: Deterministic Noise Triage vs. Material Classification:**
  * Evaluates filing subjects and exchange categories to separate routine compliance noise from thesis-relevant disclosures.
  * **Priority 1–2 (Buried Routine Noise):** Duplicate share certificates (`Loss of Share Certificates`), newspaper advertisements (`Copy of Newspaper Publication`), trading window closures, and routine investor grievance reports.
  * **Priority 3–5 (Surfaced Material Signal):** Audited/unaudited quarterly financial results (`RESULTS`), key managerial or board changes (`EXEC_CHANGE`), capital raises/dividends/M&A (`CORP_ACTION`), litigation/NCLAT/tax orders (`REGULATORY_LEGAL`), and investor presentations/concall transcripts (`CONCALL_PPT`).
* **Stage 2: Selective PDF Parsing & Gemini Enrichment:**
  * Rather than wasting LLM tokens on Priority 1–2 noise, `push_engine.py` downloads the attached NSE PDF (`pypdf`) **only** for material filings (Priority $\ge 3$).
  * **For `RESULTS` Filings:** Uses Gemini structured output to parse the quarterly result PDF table into exact **Revenue, EBITDA/Operating Profit, and PAT** figures with **QoQ % and YoY %** deltas.
  * **For `EXEC_CHANGE` Filings:** Extracts the appointee/departing executive's role, effective date, and a 1-line **biographical pedigree summary** (prior institutional experience and governance relevance) from the board resolution annexure.
  * `push_engine_2.py` acts as a secondary batch/retry worker to process remaining filings cleanly across API rate-limit windows.

---

### File 3: `upgrade_data_and_macro.py` — Macro Pulse & 30-Day Time-Series Backfiller
Connects bottom-up stock monitoring with top-down macroeconomic and sovereign policy shocks (`macro_pulse.json`).

* **Macro Benchmark Time-Series (`yfinance`):**
  * Fetches 30-day trajectories, latest levels, and 1D/5D/30D percentage shifts for four core benchmarks:
    1. **Brent Crude (`BZ=F`)** — Input cost & margin transmission for paints, aviation, logistics, and O&G.
    2. **USD/INR (`INR=X`)** — Revenue tailwind/headwind for IT exporters (`INFY`, `TCS`) and pharma (`CIPLA`), vs. import cost pressure.
    3. **US 10Y / India 10Y Yield (`^TNX`)** — FII flow sensitivity and discount-rate pressure on high-P/E compounders.
    4. **India VIX (`^INDIAVIX`)** — Broad market volatility regime indicator.
* **RBI & Sovereign Policy Transmission (RBI / PIB):**
  * Scrapes and structures official Reserve Bank of India (RBI) notifications and Press Information Bureau (PIB) cabinet/ministry releases.
  * Maps each sovereign action to a **transmission mechanism** and explicitly tags **`affected_tickers`** in the portfolio (e.g., mapping RBI NBFC/banking liquidity circulars to `HDFCBANK` and defence procurement clearances to `HAL`/`AVANTEL`).

---

### File 4: `pull_engine.py` — On-Demand Primary Research, Chart Parser & Self-Healing Synthesizer
Powers the `Research` tab (`GET /api/pull/research/{ticker}`), capable of analysing **any valid NSE symbol** on demand.

* **1. Live Screener.in & Primary PDF Scraper (`scrape_screener_fundamentals` & `extract_pdf_excerpt`):**
  * Requests `https://www.screener.in/company/{ticker}/consolidated/` (automatically falling back to standalone if consolidated tables are sparse).
  * Extracts top valuation/efficiency ratios (`Market Cap`, `Stock P/E`, `ROCE`, `ROE`) and five complete HTML financial tables (`Quarters`, `Profit & Loss`, `Balance Sheet`, `Cash Flows`, `Ratios`).
  * Collects direct URLs to the latest **Concall Transcripts** and **Investor Presentations**, plus any matching PDF filings in `corporate_filings.json`, extracting verbatim text from the first 8 pages via `pypdf` (`extract_pdf_excerpt`).
* **2. Zero-Token Numeric Chart Parser (`parse_screener_numeric_charts`):**
  * Deterministically parses HTML `<table>` rows using `BeautifulSoup` and regex (`_parse_num`) without spending LLM tokens, aligning fiscal years across statements to construct four chart series:
    * **`quarters_parsed`**: Last 4 quarters of Revenue, OPM %, and Net Profit.
    * **`trajectory`**: 6-year annual Revenue (₹ Cr) and Operating Profit Margin (OPM %).
    * **`cash_vs_pat`**: 5-year aligned Operating Cash Flow (CFO) vs. Net Profit (PAT) to audit accounting profit conversion.
    * **`balance_sheet`**: 5-year Total Borrowings (₹ Cr), ROCE %, and Working Capital Days / Debtor Days.
* **3. Structured Gemini Synthesis with Smart Failover (`invoke_research_llm`):**
  * Passes the 5 markdown financial tables + primary concall transcript excerpts into **Google Gemini (`google-genai`)** using strict Pydantic schema enforcement (`response_schema=PullResearchReport`, `temperature=0.1`).
  * Enforces a strict system prompt: *Every single claim in `narrative_vs_numbers`, `cash_quality`, `forward_guidance`, `bull_case`, and `bear_case` must include an exact `verbatim_quote` and `source_citation` from the primary filings.*
  * Implements multi-model cascading across Gemini Flash models with automatic failover to **Groq (`llama-3.3-70b-versatile`)** in JSON mode if configured.
* **4. Self-Healing Deterministic Fallback (`build_deterministic_fallback_report`):**
  * If all external LLM APIs are offline, unconfigured, or rate-limited when a user searches an uncached stock (e.g., `SBCL`, `MARUTI`), FINN never throws a 500 error.
  * Instead, it mathematically computes QoQ revenue deltas, 3-year cumulative CFO-to-PAT conversion:
    $$\text{Cash Conversion}_{3\text{Y}} = \left( \frac{\sum_{t=-2}^{0} \text{CFO}_t}{\sum_{t=-2}^{0} \text{PAT}_t} \right) \times 100$$
    and constructs all 4 quarterly audit cards (`Beat` if $\text{QoQ} \ge +3\%$, `Missed` if $\le -3\%$, else `In-Line`), headline metrics, and bull/bear cases directly from the audited tables.

---

### File 5: `app.py` — FastAPI Backend, Signal Fusion Math, `Ask FINN` Router & Mobile SPA
Serves both the REST API endpoints and the mobile-first single-page frontend (`Chart.js` + vanilla JS).

* **Dataset Timestamp Anchoring (`get_dataset_anchor_dt`):**
  * Rather than filtering relative to `datetime.now()` (which would cause static cached filings to age out after 24 hours), `app.py` anchors time windows (`24H`, `3D`, `7D`, `30D`) to the **newest `broadcast_time` in `corporate_filings.json`**. This guarantees the app renders identically whether evaluated today or weeks from now.
* **Cross-Domain Fused Brief Scoring (`GET /api/push/fused-brief`):**
  * Joins EOD price/volume metrics (`eod_metrics.json`), corporate filings (`enriched_push_feed.json`), and macro transmissions (`macro_pulse.json`) by ticker and date.
  * Ranks every holding using a composite **Conviction Score**:
    $$\text{Score} = 1.5|\Delta P_{\%}| + 3.0\max(0, \text{Vol}_{\text{ratio}} - 1) + 2.5(\text{Filing Priority}) + 1.5(\mathbb{I}_{\text{Macro}}) + 6.0(\mathbb{I}_{\text{EOD Anomaly} \land \text{Material Filing}})$$
    where the $+6.0$ interaction bonus specifically elevates stocks experiencing *both* an EOD price/volume spike and a high-priority corporate filing on the same trading session.
* **Explainable Surfacing Generator (`build_why_surfaced`):**
  * Dynamically attaches a 1-line **`✦ Why surfaced`** badge to every digest card by combining its priority tier, category rationale, and concurrent EOD market reaction (e.g., *"Priority 5/5 • Audited quarterly financial statements • Coincided with +3.8% move on 2.4x vol"*).
* **`✦ Ask FINN` Grounded Copilot (`POST /api/ask`):**
  * Extracts candidate tickers from natural-language queries via regex token matching and an alias dictionary (`extract_candidate_tickers`).
  * Assembles a strict context bundle containing EOD metrics, recent NSE filings, and full Pull Research JSONs for matched tickers.
  * Invokes Gemini with `temperature=0.0` and `response_schema=AskResponse`, enforcing a hard refusal guardrail: if a requested metric or fact is absent from the provided primary filings, FINN explicitly responds **"Not in the provided primary filings/transcripts"** rather than hallucinating from general pre-training memory.

---

## 3. Complete Feature Walkthrough (UI Tabs)

| Tab / Component | Core Features & Investor Workflow |
| :--- | :--- |
| **1. `Brief` (30-Second Fused Brief)** | • **Signal Filter Funnel:** Live telemetry bar (`58 filings scanned → X surfaced • Y buried`).<br>• **Conviction vs. Reaction Matrix:** Dual-axis visual comparing **Filing Priority (1–5)** vs. **Volume Spike (x 20D SMA)** to spot under-the-radar material filings vs. unexplained volume spikes.<br>• **Fused Alert Cards:** Ranked cross-domain cards linking price action, exchange PDFs, and macro shocks in one view. |
| **2. `Digest` (Windowed Filing Feed)** | • **Window & Category Filters:** Defaults to `24H` with 1-tap toggles for `3D`, `7D`, `30D`, category pills (`RESULTS`, `EXEC_CHANGE`, `CORP_ACTION`, `LEGAL`), and ticker filter.<br>• **Inline Structured Tables & Bios:** Renders parsed Q2 Revenue/EBITDA/PAT QoQ & YoY tables and executive biographical pedigree directly inside the card.<br>• **`✦ Why surfaced` & `🔇 Bury like this`:** Explains why each card cleared the filter and lets the user locally mute recurring `Ticker : Category` pairs via `localStorage`.<br>• **Collapsed Noise Drawer:** Tucks away Priority 1–2 routine filings at the bottom of the feed for full auditability without clutter. |
| **3. `Alerts` (Watchdogs & Replay)** | • **Dynamic Threshold Sliders:** Real-time sliders for **Daily Price Move (`±0.5%` to `±6.0%`)** and **Volume Spike (`1.0x` to `3.5x` 20D SMA)**.<br>• **30-Day Sparkline Modal:** Tapping any triggered alert opens a 30-day dual price and volume chart.<br>• **`▶ Replay` Intraday Simulator:** Header button that steps chronologically through the selected day's market breaches and NSE filings every 2.5 seconds via top-of-screen push toast notifications. |
| **4. `Macro` (Sovereign & Commodity Pulse)** | • **30-Day Benchmark Cards:** Interactive 30D charts and 1D/5D/30D deltas for **Brent Crude**, **USD/INR**, **10Y Yield**, and **India VIX**.<br>• **RBI & PIB Transmission Feed:** Official sovereign circulars mapped directly to sector transmission channels and clickable portfolio ticker chips. |
| **5. `Research` (On-Demand Pull Audit)** | • **Universal NSE Search:** Instant pills for cached stocks + live search bar for **any NSE symbol**.<br>• **4 Sub-Views:** `Narrative vs #s` (4-quarter management promise vs. actual delivery audit), `Trajectory` (Multi-year Revenue & OPM % chart), `Cash & B/S` (3Y CFO vs. PAT conversion & Borrowings vs. Working Capital Days charts), and `Guidance & Case` (Forward guidance, Bull & Bear cases).<br>• **`▸ Show the line` Verbatim Drawers:** Expandable primary-source quotes and exact table citations under every claim. |
| **6. `✦ Ask FINN` (Primary-Source Copilot)** | • Slide-up bottom sheet accessible from every tab with 1-tap starter prompts, clickable source citation chips, and a strict *"Not in the filings"* anti-hallucination guardrail. |

---

## 4. Local Setup & Deployment

### 1. Install Dependencies
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/finn-enam-agent.git
cd finn-enam-agent
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_optional_groq_fallback_key
```

### 3. Start the Application
```bash
python app.py
```
Open **`http://127.0.0.1:8000`** in your browser (or use Chrome DevTools Device Toolbar `Ctrl + Shift + M` to preview the mobile viewport on desktop).

---

## 5. Known Trade-offs & Production Roadmap

1. **`Ask FINN` Routing Scope:** Currently uses regex token matching and an alias dictionary capped at 2 candidate tickers per query to stay within free-tier latency and token-per-minute (TPM) limits. In production, this upgrades to a 2-stage **Agentic Tool-Calling Router** backed by **DuckDB/PostgreSQL** for cross-portfolio SQL aggregations and a hybrid **BM25 + Vector index** over chunked filing PDFs.
2. **Snapshot vs. Live Streaming Push Feed:** NSE's corporate announcements endpoint blocks cloud datacenter IPs (Render/AWS) via Akamai TLS fingerprinting. The Push tabs therefore serve pre-enriched snapshots from `data_cache/`, while Pull Research scrapes Screener.in live on demand. Production deployment uses a residential-proxy cron worker pushing live NSE announcements via **WebSockets / Server-Sent Events (SSE)**.
3. **Scanned PDF OCR:** `pypdf` extracts text from the first 8 pages of digital text PDFs; scanned image PDFs (common in board meeting outcomes) require an asynchronous **Gemini 2.5 Flash multimodal PDF vision / OCR** queue in production.