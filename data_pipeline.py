import os
import json
import time
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import requests

# 1. Mock Portfolio (16 stocks across Large, Mid, and Small Cap & varied sectors)
PORTFOLIO_TICKERS = [
    "RELIANCE", "HDFCBANK", "INFY", "TATAMOTORS", "SUNPHARMA", "LT",     # Large Cap
    "POLYCAB", "PERSISTENT", "TRENT", "DIXON", "FEDERALBNK", "PIIND",    # Mid Cap
    "KAYNES", "NEULANDLAB", "CDSL", "ANGELONE"                           # Small / Growth
]

AD_HOC_TICKERS = ["ZOMATO", "HAL", "TITAN", "BSE"]

CACHE_DIR = "./data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# =====================================================================
# PART A: EOD Price & Volume Engine (yfinance)
# =====================================================================
def fetch_eod_metrics(tickers: list[str], vol_threshold: float = 2.0, price_pct_threshold: float = 3.0) -> list[dict]:
    """
    Fetches the last 30 trading days of EOD data for given NSE tickers.
    Calculates 1-day return (%) and Volume vs 20-day SMA ratio.
    """
    cache_path = os.path.join(CACHE_DIR, "eod_metrics.json")
    yf_symbols = [f"{t}.NS" for t in tickers]
    
    print(f"[*] Fetching EOD data for {len(tickers)} tickers from Yahoo Finance...")
    try:
        # Download 2 months of daily data to ensure at least 20 trading days for SMA
        data = yf.download(yf_symbols, period="2mo", interval="1d", group_by="ticker", progress=False)
        
        results = []
        for ticker in tickers:
            sym = f"{ticker}.NS"
            df = data[sym].dropna() if len(tickers) > 1 else data.dropna()
            
            if len(df) < 21:
                continue
                
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            close_curr = float(latest["Close"])
            close_prev = float(prev["Close"])
            vol_curr = float(latest["Volume"])
            
            # 20-day Volume Simple Moving Average (excluding today)
            vol_sma_20 = float(df["Volume"].iloc[-21:-1].mean())
            
            daily_return_pct = ((close_curr - close_prev) / close_prev) * 100.0
            vol_ratio = (vol_curr / vol_sma_20) if vol_sma_20 > 0 else 0.0
            
            # Flag triggered alerts based on configurable thresholds
            is_price_alert = abs(daily_return_pct) >= price_pct_threshold
            is_vol_alert = vol_ratio >= vol_threshold
            
            results.append({
                "ticker": ticker,
                "date": df.index[-1].strftime("%Y-%m-%d"),
                "close": round(close_curr, 2),
                "prev_close": round(close_prev, 2),
                "daily_return_pct": round(daily_return_pct, 2),
                "volume": int(vol_curr),
                "vol_sma_20": int(vol_sma_20),
                "vol_ratio": round(vol_ratio, 2),
                "triggered_alert": is_price_alert or is_vol_alert,
                "alert_reasons": [
                    *( [f"Price move: {daily_return_pct:+.2f}%"] if is_price_alert else [] ),
                    *( [f"Volume spike: {vol_ratio:.1f}x 20D Avg"] if is_vol_alert else [] )
                ]
            })
            
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    except Exception as e:
        print(f"[!] Error fetching live EOD data: {e}. Falling back to cache.")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return []


# =====================================================================
# PART B: NSE Corporate Announcements Scraper with Session Handshake
# =====================================================================
class NSEClient:
    """
    Handles NSE India's cookie protection by establishing a browser session
    on the homepage before hitting internal JSON APIs.
    """
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
            "Connection": "keep-alive"
        }
        self.session.headers.update(self.headers)
        self._init_cookies()

    def _init_cookies(self):
        try:
            # Hit main page to grab valid nsit & nseappid cookies
            self.session.get(self.base_url, timeout=10)
            time.sleep(1)
        except Exception as e:
            print(f"[!] Warning: NSE cookie initialization failed: {e}")

    def fetch_announcements(self, symbol: str, days_back: int = 7) -> list[dict]:
        to_date = datetime.now().strftime("%d-%m-%Y")
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%m-%Y")
        
        url = (
            f"{self.base_url}/api/corporate-announcements"
            f"?index=equities&symbol={symbol}&from_date={from_date}&to_date={to_date}"
        )
        
        for attempt in range(2):
            try:
                resp = self.session.get(url, timeout=10)
                if resp.status_code == 401 or resp.status_code == 403:
                    self._init_cookies()
                    continue
                resp.raise_for_status()
                raw_data = resp.json()
                
                cleaned = []
                for item in raw_data:
                    cleaned.append({
                        "ticker": item.get("symbol"),
                        "company_name": item.get("sm_name"),
                        "subject": item.get("desc"),                 # e.g., "Outcome of Board Meeting"
                        "details": item.get("attchmntText"),         # Brief description of the filing
                        "pdf_url": item.get("attchmntFile"),         # Direct link to the exchange PDF
                        "broadcast_time": item.get("an_dt"),         # e.g., "29-Oct-2026 18:45:12"
                        "exchange": "NSE"
                    })
                return cleaned
            except Exception as e:
                print(f"[!] Attempt {attempt+1} failed for {symbol}: {e}")
                time.sleep(1.5)
        return []


def fetch_portfolio_filings(tickers: list[str], days_back: int = 7) -> list[dict]:
    cache_path = os.path.join(CACHE_DIR, "corporate_filings.json")
    client = NSEClient()
    all_filings = []
    
    print(f"[*] Fetching last {days_back} days of NSE filings for {len(tickers)} stocks...")
    for idx, ticker in enumerate(tickers):
        filings = client.fetch_announcements(ticker, days_back=days_back)
        all_filings.extend(filings)
        print(f"  [{idx+1}/{len(tickers)}] {ticker}: {len(filings)} filings retrieved")
        time.sleep(0.6)  # Polite rate limit to avoid IP block
        
    if all_filings:
        with open(cache_path, "w") as f:
            json.dump(all_filings, f, indent=2)
        return all_filings
    else:
        print("[!] No live filings retrieved. Loading from local fallback cache...")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return []


if __name__ == "__main__":
    # 1. Test EOD Price & Volume Pipeline
    eod_data = fetch_eod_metrics(PORTFOLIO_TICKERS, vol_threshold=2.0, price_pct_threshold=3.0)
    alerts = [d for d in eod_data if d["triggered_alert"]]
    print(f"\n[+] EOD Sync Complete: {len(eod_data)} stocks processed, {len(alerts)} alerts triggered.")
    for a in alerts:
        print(f"    ⚡ {a['ticker']}: {', '.join(a['alert_reasons'])}")

    # 2. Test Corporate Filings Pipeline
    filings = fetch_portfolio_filings(PORTFOLIO_TICKERS, days_back=7)
    print(f"\n[+] Filings Sync Complete: Total {len(filings)} filings cached in {CACHE_DIR}/corporate_filings.json")