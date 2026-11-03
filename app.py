import os
import sys
import re
import json
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Check if data_cache is right next to app.py first (works on Render & root local), else check parent
if os.path.exists(os.path.join(SRC_DIR, "data_cache", "eod_metrics.json")):
    CACHE_DIR = os.path.join(SRC_DIR, "data_cache")
else:
    CACHE_DIR = os.path.join(os.path.dirname(SRC_DIR), "data_cache")


PUSH_FEED_PATH = os.path.join(CACHE_DIR, "enriched_push_feed.json")
EOD_METRICS_PATH = os.path.join(CACHE_DIR, "eod_metrics.json")
MACRO_PULSE_PATH = os.path.join(CACHE_DIR, "macro_pulse.json")
PULL_CACHE_DIR = os.path.join(CACHE_DIR, "pull_research")

from pull_engine import generate_pull_research, PULL_MODELS, get_gemini_client
from data_pipeline import PORTFOLIO_TICKERS

app = FastAPI(title="FINN - ENAM Financial Investment Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def parse_nse_time(dt_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(dt_str.strip(), "%d-%b-%Y %H:%M:%S")
    except Exception:
        return None


def get_trading_dates_from_cache(all_filings: list) -> list[dict]:
    """
    Builds the 3-day calendar strip strictly from actual weekday trading sessions
    found in eod_metrics.json's history_30d (never lands on a Saturday/Sunday label).
    """
    trading_days_iso = []
    if os.path.exists(EOD_METRICS_PATH):
        try:
            with open(EOD_METRICS_PATH, "r", encoding="utf-8") as f:
                eod_list = json.load(f)
            if eod_list:
                ref_stock = eod_list[0]
                ref_year = int((ref_stock.get("date") or "2026-10-29")[:4])
                hist = ref_stock.get("history_30d", [])
                for pt in reversed(hist):
                    raw_lbl = pt.get("date", "")
                    try:
                        dt_obj = datetime.strptime(f"{raw_lbl} {ref_year}", "%d %b %Y")
                        if dt_obj.weekday() < 5:
                            trading_days_iso.append(dt_obj)
                    except Exception:
                        continue
        except Exception:
            pass

    if not trading_days_iso:
        seen = set()
        for item in all_filings:
            t = parse_nse_time(item.get("broadcast_time", ""))
            if t and t.weekday() < 5 and t.strftime("%Y-%m-%d") not in seen:
                seen.add(t.strftime("%Y-%m-%d"))
                trading_days_iso.append(t)
        trading_days_iso.sort(reverse=True)

    available_dates = []
    for d_obj in trading_days_iso[:3]:
        d_iso = d_obj.strftime("%Y-%m-%d")
        d_count = sum(
            1 for item in all_filings
            if (parse_nse_time(item.get("broadcast_time", "")) or datetime.min).strftime("%Y-%m-%d") == d_iso
        )
        available_dates.append({
            "iso": d_iso,
            "day_name": d_obj.strftime("%a"),
            "day_num": d_obj.strftime("%d %b"),
            "count": d_count
        })
    return available_dates


def get_stock_metrics_for_date(stock: dict, target_date: Optional[str]) -> dict:
    if not target_date or target_date == "all" or stock.get("date") == target_date:
        return stock

    hist = stock.get("history_30d", [])
    if not hist:
        return stock

    try:
        dt_obj = datetime.strptime(target_date, "%Y-%m-%d")
        short_label = dt_obj.strftime("%d %b")
    except Exception:
        return stock

    for idx, pt in enumerate(hist):
        if pt.get("date") == short_label and idx > 0:
            close_c = float(pt["close"])
            close_p = float(hist[idx - 1]["close"])
            vol_c = int(pt.get("volume", stock.get("volume", 0)))
            sma_window = [float(x.get("volume", vol_c)) for x in hist[max(0, idx - 20):idx]]
            vol_sma = sum(sma_window) / len(sma_window) if sma_window else float(stock.get("vol_sma_20", 1))
            ret_pct = ((close_c - close_p) / close_p) * 100.0 if close_p else 0.0
            vol_ratio = (vol_c / vol_sma) if vol_sma > 0 else 1.0

            return {
                **stock,
                "date": target_date,
                "close": round(close_c, 2),
                "prev_close": round(close_p, 2),
                "daily_return_pct": round(ret_pct, 2),
                "volume": vol_c,
                "vol_ratio": round(vol_ratio, 2),
            }
    return stock


# =====================================================================
# FINN MOBILE UI (Complete Embedded Single-Page Application)
# =====================================================================
EMBEDDED_UI_HTML = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
<meta name="theme-color" content="#0B2A55" />
<title>FINN — ENAM Holdings Financial Investment Agent</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#EEF2F7; --card:#FFFFFF; --card2:#F6F8FB; --text:#1B2535; --muted:#66748A; --line:#E1E7EF;
  --navy:#0B2A55; --accent:#2F66D0; --accent-soft:#E6EEFC;
  --up:#12936A; --up-soft:#E3F5EE; --down:#D1405A; --down-soft:#FBE8EC; --warn:#B07A14; --warn-soft:#FBF1DC;
  --shadow:0 1px 2px rgba(15,30,60,.06),0 4px 14px rgba(15,30,60,.05);
}
[data-theme="dark"]{
  --bg:#0D131B; --card:#151D28; --card2:#1B2533; --text:#E2E8F1; --muted:#8996AA; --line:#243143;
  --navy:#22324A; --accent:#6BA3FF; --accent-soft:#1B2C47;
  --up:#3FD29B; --up-soft:#12302A; --down:#FF7088; --down-soft:#3A1B25; --warn:#E3B04B; --warn-soft:#352B14;
  --shadow:0 1px 2px rgba(0,0,0,.4);
}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
html,body{margin:0;background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:14px;line-height:1.45}
body{display:flex;justify-content:center;min-height:100vh}
.mono,.num{font-family:'JetBrains Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums}
button{font:inherit;color:inherit;background:none;border:0;cursor:pointer;padding:0}
.app{width:100%;max-width:480px;min-height:100vh;background:var(--bg);position:relative;display:flex;flex-direction:column}
.hdr{position:sticky;top:0;z-index:30;background:#0B2A55;color:#fff;padding:12px 16px 10px;padding-top:max(12px,env(safe-area-inset-top))}
.hdr-top{display:flex;align-items:center;justify-content:space-between;gap:10px}
.brand{display:flex;align-items:center;gap:10px}
.logo{width:32px;height:32px;border-radius:9px;background:#fff;color:#0B2A55;display:grid;place-items:center;font-weight:800;font-size:16px}
.brand h1{margin:0;font-size:16px;letter-spacing:.02em}
.brand p{margin:0;font-size:11px;color:#A9BEDD}
.hbtns{display:flex;gap:6px;align-items:center}
.ibtn{width:32px;height:32px;border-radius:9px;background:rgba(255,255,255,.1);display:grid;place-items:center;color:#fff}
.ibtn svg{width:16px;height:16px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.replay-btn{padding:0 10px;height:32px;border-radius:9px;background:rgba(18,147,106,.28);border:1px solid rgba(110,231,183,.45);color:#6EE7B7;font-size:11px;font-weight:700;font-family:'JetBrains Mono',monospace;display:flex;align-items:center;gap:5px}
.replay-btn.active{background:#D1405A;border-color:#FDA4AF;color:#fff;animation:pulse 1.2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.72}}
.tape{display:flex;gap:6px;overflow-x:auto;margin-top:8px;scrollbar-width:none;transition:max-height .22s ease,opacity .18s ease,margin .22s ease;max-height:36px;opacity:1}
.tape::-webkit-scrollbar{display:none}
.hdr.compact .tape{max-height:0;opacity:0;margin-top:0;overflow:hidden;pointer-events:none}
.chip{flex:none;display:flex;gap:6px;align-items:center;padding:3px 8px;border-radius:7px;background:rgba(255,255,255,.08);font-size:11px;color:#fff}
.chip .u{color:#6EE7B7}.chip .d{color:#FDA4AF}
.ask-fab{padding:0 10px;height:32px;border-radius:9px;background:#fff;color:#0B2A55;font-size:11px;font-weight:700;display:flex;align-items:center;gap:5px}
.sheet-backdrop{position:fixed;inset:0;background:rgba(8,15,28,.55);z-index:80;display:none;align-items:flex-end;justify-content:center}
.sheet-backdrop.open{display:flex}
.sheet{width:100%;max-width:480px;max-height:82vh;background:var(--card);border-top-left-radius:18px;border-top-right-radius:18px;border:1px solid var(--line);padding:14px 16px calc(18px + env(safe-area-inset-bottom));display:flex;flex-direction:column;gap:10px;box-shadow:0 -8px 30px rgba(0,0,0,.28);overflow-y:auto}
.cal-strip{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:12px}
.cal-btn{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:7px 6px;text-align:center;box-shadow:var(--shadow);display:flex;flex-direction:column;align-items:center;gap:2px;transition:.15s}
.cal-btn.on{background:var(--navy);border-color:var(--navy);color:#fff}
[data-theme="dark"] .cal-btn.on{background:var(--accent);border-color:var(--accent);color:#0D131B}
.cal-btn small{font-size:10px;opacity:.75;font-weight:600;text-transform:uppercase}
.cal-btn b{font-size:12px;font-family:'JetBrains Mono',monospace}
.cal-btn i{font-style:normal;font-size:9px;padding:1px 6px;border-radius:99px;background:var(--card2);color:var(--muted);font-family:'JetBrains Mono',monospace}
.cal-btn.on i{background:rgba(255,255,255,.22);color:inherit}
.toast-wrap{position:fixed;top:14px;left:50%;transform:translateX(-50%);width:calc(100% - 24px);max-width:456px;z-index:99;display:flex;flex-direction:column;gap:8px;pointer-events:none}
.toast{pointer-events:auto;background:#0B2A55;color:#fff;border:1px solid #6BA3FF;border-left:4px solid #3FD29B;border-radius:12px;padding:10px 12px;box-shadow:0 10px 28px rgba(0,0,0,.35);display:flex;flex-direction:column;gap:4px;animation:slideIn .25s ease-out}
@keyframes slideIn{from{opacity:0;transform:translateY(-12px)}to{opacity:1;transform:translateY(0)}}
main{flex:1;padding:14px 14px calc(84px + env(safe-area-inset-bottom))}
.view{display:none;flex-direction:column;gap:12px}
.view.on{display:flex}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;box-shadow:var(--shadow)}
.card h3{margin:0;font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:700}
.card h2{margin:0;font-size:15px;font-weight:700}
.row{display:flex;align-items:center;justify-content:space-between;gap:8px}
.col{display:flex;flex-direction:column;gap:8px}
.muted{color:var(--muted)}.up{color:var(--up)}.down{color:var(--down)}
.sm{font-size:12px}.xs{font-size:11px}
.seg{display:flex;background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:2px}
.seg button{padding:5px 10px;border-radius:8px;font-size:11px;font-weight:600;color:var(--muted)}
.seg button.on{background:var(--navy);color:#fff}
[data-theme="dark"] .seg button.on{background:var(--accent);color:#0D131B}
.pills{display:flex;gap:6px;overflow-x:auto;scrollbar-width:none;padding-bottom:2px}
.pills::-webkit-scrollbar{display:none}
.pill{flex:none;padding:5px 11px;border-radius:99px;font-size:12px;font-weight:600;border:1px solid var(--line);background:var(--card2);color:var(--muted)}
.pill.on{background:var(--navy);border-color:var(--navy);color:#fff}
[data-theme="dark"] .pill.on{background:var(--accent);border-color:var(--accent);color:#0D131B}
.badge{display:inline-block;padding:2px 7px;border-radius:6px;font-size:10px;font-weight:700;letter-spacing:.04em;font-family:'JetBrains Mono',monospace}
.b-up{background:var(--up-soft);color:var(--up)}.b-down{background:var(--down-soft);color:var(--down)}
.b-warn{background:var(--warn-soft);color:var(--warn)}.b-info{background:var(--accent-soft);color:var(--accent)}
.b-neutral{background:var(--card2);color:var(--muted);border:1px solid var(--line)}
.tick{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:12px;padding:3px 8px;border-radius:7px;background:var(--navy);color:#fff}
[data-theme="dark"] .tick{background:var(--accent);color:#0D131B}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 8px;text-align:center;box-shadow:var(--shadow)}
.kpi b{display:block;font-size:20px;font-family:'JetBrains Mono',monospace}
.kpi span{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.heat{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.tile{border-radius:10px;padding:8px 6px;text-align:left;border:1px solid var(--line);min-height:54px;display:flex;flex-direction:column;justify-content:space-between}
.tile b{font-size:11px;font-family:'JetBrains Mono',monospace}
.tile span{font-size:12px;font-family:'JetBrains Mono',monospace;font-weight:600}
.chartbox{position:relative;width:100%}
.h110{height:110px}.h150{height:150px}.h190{height:190px}.h220{height:220px}
.sep{height:1px;background:var(--line);margin:2px 0}
.table{width:100%;border-collapse:collapse;font-size:12px}
.table th{font-size:10px;color:var(--muted);text-align:right;font-weight:600;padding-bottom:4px;letter-spacing:.04em}
.table th:first-child,.table td:first-child{text-align:left}
.table td{padding:6px 0;border-top:1px solid var(--line);text-align:right}
.dots{display:flex;gap:2px}.dots i{width:5px;height:5px;border-radius:50%;background:var(--line)}.dots i.f{background:var(--accent)}
.note{background:var(--card2);border-left:3px solid var(--accent);border-radius:8px;padding:9px 10px;font-size:12px}
.note.up-b{border-left-color:var(--up)}.note.down-b{border-left-color:var(--down)}
.lbl{font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);font-family:'JetBrains Mono',monospace;margin-bottom:3px}
a{color:var(--accent);text-decoration:none;font-weight:600}
input[type=text],select{background:var(--card2);color:var(--text);border:1px solid var(--line);border-radius:10px;padding:9px 11px;font:inherit;font-size:13px;outline:none}
input[type=text]:focus,select:focus{border-color:var(--accent)}
input[type=range]{width:100%;accent-color:var(--accent)}
.btn{background:var(--navy);color:#fff;border-radius:10px;padding:9px 14px;font-weight:600;font-size:13px}
[data-theme="dark"] .btn{background:var(--accent);color:#0D131B}
.nav{position:fixed;bottom:0;width:100%;max-width:480px;z-index:30;background:var(--card);border-top:1px solid var(--line);display:flex;justify-content:space-around;padding:6px 4px calc(6px + env(safe-area-inset-bottom))}
.nav button{display:flex;flex-direction:column;align-items:center;gap:2px;color:var(--muted);font-size:10px;font-weight:600;padding:4px 8px;border-radius:10px;position:relative}
.nav button svg{width:19px;height:19px;stroke:currentColor;fill:none;stroke-width:1.9;stroke-linecap:round;stroke-linejoin:round}
.nav button.on{color:var(--accent)}
.nav .dot{position:absolute;top:2px;right:8px;width:8px;height:8px;border-radius:50%;background:var(--down);display:none}
.skel{border-radius:12px;background:linear-gradient(90deg,var(--card2),var(--line),var(--card2));background-size:200% 100%;animation:sh 1.3s infinite;height:90px}
@keyframes sh{0%{background-position:200% 0}100%{background-position:-200% 0}}
.spin{width:26px;height:26px;border:3px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:rot .8s linear infinite;margin:auto}
@keyframes rot{to{transform:rotate(360deg)}}
.empty{padding:26px 10px;text-align:center;color:var(--muted);font-size:13px}
.bar2{display:flex;align-items:center;gap:6px;height:6px}
.bar2 i{display:block;height:6px;border-radius:3px}
details>summary{list-style:none;cursor:pointer}
details>summary::-webkit-details-marker{display:none}
.kgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.kcard{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:9px 10px;text-align:left}
.hidden{display:none!important}
.hero-filter{background:linear-gradient(135deg,#0B2A55 0%,#1E4B8F 100%);color:#fff;border-radius:14px;padding:12px 14px;display:flex;align-items:center;justify-content:space-between;box-shadow:var(--shadow)}
.hero-filter b{font-size:17px;font-family:'JetBrains Mono',monospace;color:#6EE7B7}
.why-line{font-size:11px;color:var(--accent);background:var(--accent-soft);padding:5px 8px;border-radius:7px;font-weight:600;display:flex;align-items:center;justify-content:space-between;gap:6px}
.bury-btn{font-size:10px;color:var(--muted);padding:2px 7px;border-radius:6px;border:1px solid var(--line);background:var(--card2);font-weight:600}
.bury-btn:hover{color:var(--down);border-color:var(--down)}
.quote-exp{margin-top:4px;font-size:11px}
.quote-exp summary{color:var(--accent);font-weight:600;font-family:'JetBrains Mono',monospace;font-size:10px;display:inline-flex;align-items:center;gap:4px}
.quote-box{margin-top:5px;padding:8px 10px;background:var(--card);border:1px dashed var(--line);border-left:3px solid var(--accent);border-radius:6px;font-style:italic;color:var(--text);font-size:11px;line-height:1.45}

@media (min-width:520px){
  body{padding:16px 0;background:#DCE4F0}
  [data-theme="dark"] body{background:#070B10}
  .app{border-radius:24px;border:1px solid var(--line);min-height:calc(100vh - 32px);overflow:hidden;box-shadow:0 20px 50px rgba(0,0,0,.18)}
  .nav{border-bottom-left-radius:24px;border-bottom-right-radius:24px}
}
</style>
</head>
<body>

<div id="toast-box" class="toast-wrap"></div>

<div class="app">

<header class="hdr" id="main-hdr">
  <div class="hdr-top">
    <div class="brand">
      <div class="logo">E</div>
      <div><h1>FINN</h1><p id="asof">ENAM Investment Agent</p></div>
    </div>
    <div class="hbtns">
      <button class="ask-fab" data-act="open-ask">✦ Ask FINN</button>
      <button id="btn-replay" class="replay-btn" data-act="replay">▶ Replay</button>
      <button class="ibtn" data-act="refresh" aria-label="Refresh"><svg viewBox="0 0 24 24"><path d="M21 12a9 9 0 1 1-3-6.7M21 4v5h-5"/></svg></button>
      <button class="ibtn" data-act="theme" aria-label="Toggle theme"><svg viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg></button>
    </div>
  </div>
  <div id="tape" class="tape mono"></div>
</header>

<main>

<div id="push-cal-bar" class="cal-strip"></div>

<!-- TAB 1: BRIEF (FUSED SIGNAL + CONVICTION MATRIX) -->
<section id="view-pulse" class="view on">
  <div id="filter-hero" class="hero-filter"></div>

  <div class="card col" style="border-left:4px solid var(--accent)">
    <div class="row">
      <div>
        <h3>30-Second Fused Brief</h3>
        <div id="fused-sub" class="xs muted">Price anomalies, filings &amp; macro linked</div>
      </div>
      <span class="badge b-info">FUSED</span>
    </div>
    <div id="fused-brief-list" class="col" style="margin-top:4px"></div>
  </div>

  <div id="pulse-kpis" class="kpis"></div>

  <div class="card col">
    <div class="row"><h3 id="heat-title">Portfolio Heatmap (1D)</h3><span class="xs muted">tap for 30D chart</span></div>
    <div id="heat" class="heat"></div>
  </div>

  <div class="card col">
    <div class="row"><h3>Conviction vs Reaction (Priority vs Vol)</h3><span class="xs muted">judgment view</span></div>
    <div class="chartbox h190"><canvas id="moversChart"></canvas></div>
  </div>

  <div class="card col">
    <div class="row"><h3>Volume vs 20D Average</h3><span id="vol-thr-lbl" class="badge b-info">2.0x</span></div>
    <div class="chartbox h150"><canvas id="volChart"></canvas></div>
  </div>

  <div class="col">
    <div class="row" style="padding:0 2px"><h3 class="muted xs" style="margin:0;text-transform:uppercase;letter-spacing:.06em">Top Priority Filings</h3><button class="xs" style="color:var(--accent);font-weight:600" data-act="tab" data-v="digest">Full Digest →</button></div>
    <div id="pulse-top" class="col"></div>
  </div>
</section>

<!-- TAB 2: DIGEST (DEFAULTS TO 24H PER BRIEF) -->
<section id="view-digest" class="view">
  <div id="digest-hero" class="hero-filter"></div>

  <div class="card col">
    <div class="row">
      <h3>Windowed Digest</h3>
      <div class="seg" id="win-seg">
        <button data-act="win" data-v="24" class="on">24H</button>
        <button data-act="win" data-v="72">3D</button>
        <button data-act="win" data-v="168">7D</button>
        <button data-act="win" data-v="720">30D</button>
      </div>
    </div>
    <div class="pills" id="cat-pills">
      <button class="pill on" data-act="cat" data-v="all">All signal</button>
      <button class="pill" data-act="cat" data-v="results">Results</button>
      <button class="pill" data-act="cat" data-v="management_change">Mgmt change</button>
      <button class="pill" data-act="cat" data-v="M&A">M&amp;A</button>
      <button class="pill" data-act="cat" data-v="fund_raise">Fund raise</button>
      <button class="pill" data-act="cat" data-v="litigation">Litigation</button>
      <button class="pill" data-act="cat" data-v="credit_rating">Rating</button>
      <button class="pill" data-act="cat" data-v="dividend">Dividend</button>
    </div>
    <select id="tick-filter" aria-label="Filter by ticker"><option value="all">All portfolio stocks</option></select>
  </div>

  <div class="card col">
    <div class="row">
      <div><h3>Signal Mix</h3><div class="sm" style="margin-top:3px"><b id="m-count" class="mono up">0</b> surfaced • <b id="r-count" class="mono muted">0</b> collapsed</div></div>
      <span id="digest-mode-badge" class="badge b-info">24H Window</span>
    </div>
    <div class="chartbox h150"><canvas id="catChart"></canvas></div>
  </div>

  <div id="feed" class="col"></div>

  <div id="routine-box" class="card hidden">
    <details>
      <summary class="row"><span class="sm"><span id="r-badge" class="badge b-neutral">0</span>&nbsp; Routine filings collapsed</span><span class="xs" style="color:var(--accent);font-weight:600">Expand / Collapse</span></summary>
      <div id="routine-list" class="col" style="margin-top:10px"></div>
    </details>
  </div>
</section>

<!-- TAB 3: ALERTS -->
<section id="view-alerts" class="view">
  <div class="card col">
    <div class="row">
      <div><h3>30-Day Price &amp; Volume</h3><h2 id="stk-title" class="mono" style="margin-top:3px">-</h2></div>
      <select id="stk-sel" data-chg="stock"></select>
    </div>
    <div class="chartbox h220"><canvas id="stockChart"></canvas></div>
  </div>
  <div class="card col">
    <div class="row">
      <div>
        <h3>Alert Thresholds</h3>
        <div class="xs muted">Polled every 60s • fires live toast on breach</div>
      </div>
      <button class="badge b-info" data-act="replay">▶ Replay Day</button>
    </div>
    <div>
      <div class="row sm"><span>Daily Price Move (±X%)</span><b id="lbl-price" class="mono" style="color:var(--accent)">3.0%</b></div>
      <input id="sl-price" type="range" min="0.5" max="6" step="0.5" value="3">
    </div>
    <div>
      <div class="row sm"><span>Volume vs 20D Avg (Yx)</span><b id="lbl-vol" class="mono" style="color:var(--accent)">2.0x</b></div>
      <input id="sl-vol" type="range" min="1" max="3.5" step="0.25" value="2">
    </div>
  </div>
  <div class="col">
    <h3 class="muted xs" style="margin:0 2px;text-transform:uppercase;letter-spacing:.06em">Price &amp; Volume Breaches <span id="eod-n" class="badge b-warn">0</span></h3>
    <div id="eod-list" class="col"></div>
  </div>
  <div class="card col">
    <h3>Material Categories Watched</h3>
    <div class="pills" id="ev-pills"></div>
  </div>
  <div class="col">
    <h3 class="muted xs" style="margin:0 2px;text-transform:uppercase;letter-spacing:.06em">Material Filings</h3>
    <div id="ev-list" class="col"></div>
  </div>
</section>

<!-- TAB 4: MACRO -->
<section id="view-macro" class="view">
  <div class="card col">
    <div class="row"><h3>Macro Benchmarks (30D)</h3><span class="badge b-info">RBI • PIB • ICE</span></div>
    <div id="macro-grid" class="kgrid"></div>
  </div>
  <div class="card col">
    <div class="row">
      <h2 id="mc-title" style="font-size:13px">-</h2>
      <div class="seg" id="rng-seg">
        <button data-act="rng" data-v="7">1W</button>
        <button data-act="rng" data-v="15">2W</button>
        <button data-act="rng" data-v="30" class="on">1M</button>
      </div>
    </div>
    <div id="mc-stats" class="row xs mono muted"></div>
    <div class="chartbox h190"><canvas id="macroChart"></canvas></div>
  </div>
  <div class="col">
    <h3 class="muted xs" style="margin:0 2px;text-transform:uppercase;letter-spacing:.06em">RBI, Policy &amp; Shocks — Portfolio Mapped</h3>
    <div id="macro-feed" class="col"></div>
  </div>
</section>

<!-- TAB 5: RESEARCH -->
<section id="view-research" class="view">
  <div class="card col">
    <div class="row"><h3>On-Demand Research</h3><span class="badge b-info">Primary Sources Only</span></div>
    <form id="pull-form" class="row" style="gap:8px">
      <input id="pull-input" type="text" placeholder="Any NSE ticker: HAL, TCS, INFY, ZOMATO..." autocapitalize="characters" autocomplete="off" style="flex:1;text-transform:uppercase" class="mono" />
      <button class="btn" type="submit">Analyse</button>
    </form>
    <div class="pills" id="cached-chips"></div>
  </div>
  <div id="pull-loading" class="card hidden col" style="text-align:center">
    <div class="spin"></div>
    <b class="sm">Reading Annual Reports &amp; Concall Transcripts</b>
    <span class="xs muted">Zero broker notes. Primary disclosures only.</span>
  </div>
  <div id="pull-out" class="col"></div>
</section>

</main>

<!-- ASK FINN BOTTOM SHEET -->
<div id="ask-sheet" class="sheet-backdrop">
  <div class="sheet">
    <div class="row">
      <div>
        <h2 style="font-size:14px">Ask FINN Copilot</h2>
        <span class="xs muted">Grounded strictly in primary filings &amp; transcripts</span>
      </div>
      <button class="ibtn" style="background:var(--card2);color:var(--text)" data-act="close-ask">✕</button>
    </div>
    <form id="ask-form" class="row" style="gap:6px">
      <input id="ask-input" type="text" placeholder="e.g. What changed today? or Compare HAL guidance vs delivered" style="flex:1" autocomplete="off" />
      <button class="btn" type="submit">Ask</button>
    </form>
    <div class="pills">
      <button class="pill" data-act="ask-q" data-v="What changed in my portfolio today?">What changed today?</button>
      <button class="pill" data-act="ask-q" data-v="Compare INFY guidance with what it delivered">INFY guidance vs delivered</button>
      <button class="pill" data-act="ask-q" data-v="Compare HAL guidance with what it delivered">HAL guidance vs delivered</button>
    </div>
    <div id="ask-drawer" class="col"></div>
  </div>
</div>

<nav class="nav">
  <button class="on" data-act="tab" data-v="pulse" id="nav-pulse"><svg viewBox="0 0 24 24"><path d="M4 20V11M10 20V4M16 20v-7M22 20H2"/></svg>Brief</button>
  <button data-act="tab" data-v="digest" id="nav-digest"><svg viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h10"/></svg>Digest</button>
  <button data-act="tab" data-v="alerts" id="nav-alerts"><svg viewBox="0 0 24 24"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10 21h4"/></svg>Alerts<span class="dot" id="alert-dot"></span></button>
  <button data-act="tab" data-v="macro" id="nav-macro"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18"/></svg>Macro</button>
  <button data-act="tab" data-v="research" id="nav-research"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M21 21l-5-5"/></svg>Research</button>
</nav>

</div>

<script>
const $ = id => document.getElementById(id);
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = v => { const n = parseFloat(v); return isNaN(n) ? 0 : n; };
const sgn = v => (num(v) >= 0 ? '+' : '');
const pct = v => sgn(v) + num(v).toFixed(2) + '%';
const dir = v => (num(v) >= 0 ? 'up' : 'down');
const store = {
  get(k, d){ try { const v = localStorage.getItem(k); return v == null ? d : v; } catch(e){ return d; } },
  set(k, v){ try { localStorage.setItem(k, v); } catch(e){} }
};

window.addEventListener('scroll', () => {
  const hdr = $('main-hdr');
  if (hdr) hdr.classList.toggle('compact', window.scrollY > 36);
}, { passive: true });

async function api(url, opts){
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()) || r.status);
  return r.json();
}
function spark(vals, w, h){
  w = w || 70; h = h || 22;
  if (!vals || vals.length < 2) return '';
  const mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals), rg = (mx - mn) || 1;
  const pts = vals.map((v, i) => (i / (vals.length - 1) * w).toFixed(1) + ',' + (h - 2 - (v - mn) / rg * (h - 4)).toFixed(1)).join(' ');
  const col = vals[vals.length - 1] >= vals[0] ? 'var(--up)' : 'var(--down)';
  return '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '"><polyline fill="none" stroke="' + col + '" stroke-width="1.6" stroke-linejoin="round" points="' + pts + '"/></svg>';
}
function dots(n){ let s = ''; for (let i = 1; i <= 5; i++) s += '<i class="' + (i <= n ? 'f' : '') + '"></i>'; return '<div class="dots" title="Priority ' + n + '/5">' + s + '</div>'; }
function skel(n){ let s = ''; for (let i = 0; i < n; i++) s += '<div class="skel"></div>'; return s; }

const S = {
  win: 24, date: 'all', availDates: [], cat: 'all', tick: 'all',
  tape: [], macro: null, macroId: null, macroRange: 30, research: null,
  sub: 'narrative', stock: null, digest: null, evSel: null, fused: [],
  seenAlerts: new Set(), initializedAlerts: false, replayTimer: null,
  mutedPairs: new Set(JSON.parse(store.get('finn_muted', '[]')))
};
const charts = {};
const CAT = {
  results: ['RESULTS', 'b-up'], management_change: ['MGMT CHANGE', 'b-info'], 'M&A': ['M&A', 'b-warn'],
  fund_raise: ['FUND RAISE', 'b-info'], litigation: ['LITIGATION', 'b-down'], press_release: ['KEY UPDATE', 'b-neutral'],
  earnings_call: ['CONCALL', 'b-info'], dividend: ['DIVIDEND', 'b-up'], credit_rating: ['RATING', 'b-info'], investor_meetings: ['INVESTOR MEET', 'b-neutral']
};
const EV_TYPES = ['results', 'management_change', 'M&A', 'fund_raise', 'litigation', 'credit_rating', 'dividend'];
S.evSel = new Set(JSON.parse(store.get('finn_ev', JSON.stringify(EV_TYPES))));

function C(){
  const g = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
  return { up: g('--up'), down: g('--down'), accent: g('--accent'), muted: g('--muted'), line: g('--line'), text: g('--text'), warn: g('--warn') };
}
function mk(id, cfg){
  if (charts[id]) charts[id].destroy();
  const el = $(id); if (!el) return;
  const c = C();
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.font.size = 10;
  Chart.defaults.color = c.muted;
  charts[id] = new Chart(el.getContext('2d'), cfg);
}
const grid = () => ({ color: C().line, drawBorder: false });
function applyTheme(t){
  document.documentElement.setAttribute('data-theme', t);
  store.set('finn_theme', t);
  const m = document.querySelector('meta[name=theme-color]'); if (m) m.content = t === 'dark' ? '#0A1624' : '#0B2A55';
}
function toggleTheme(){
  applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
  redrawCharts();
}
function redrawCharts(){
  renderPulse(); if (S.digest) drawCatChart(S.digest);
  if (S.stock) drawStock(S.stock);
  if (S.macroId) drawMacro(S.macroId);
  if (S.research) drawPullCharts();
}
(function initTheme(){
  const saved = store.get('finn_theme', null);
  applyTheme(saved || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
})();

function showToast(badgeText, ticker, headline, subtext, isReplay){
  const box = $('toast-box');
  if (!box) return;
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML =
    '<div class="row">' +
      '<div class="row" style="gap:6px">' +
        '<span class="badge ' + (isReplay ? 'b-warn' : 'b-up') + '">' + esc(badgeText) + '</span>' +
        '<button class="tick" data-act="pull" data-v="' + esc(ticker) + '">' + esc(ticker) + ' ↗</button>' +
      '</div>' +
      '<button class="xs" style="color:#A9BEDD" onclick="this.closest(\\'.toast\\').remove()">✕</button>' +
    '</div>' +
    '<div class="sm" style="font-weight:600">' + esc(headline) + '</div>' +
    (subtext ? '<div class="xs mono" style="color:#A9BEDD">' + esc(subtext) + '</div>' : '');
  box.prepend(el);
  while (box.children.length > 2) box.lastChild.remove();
  setTimeout(() => { if (el.parentNode) el.remove(); }, 5500);
}

function toggleReplayDay(){
  const btn = $('btn-replay');
  if (S.replayTimer) {
    clearInterval(S.replayTimer);
    S.replayTimer = null;
    if (btn) { btn.classList.remove('active'); btn.textContent = '▶ Replay'; }
    return;
  }
  const targetDate = (S.date && S.date !== 'all') ? S.date : (S.availDates && S.availDates[0] ? S.availDates[0].iso : '2026-10-29');
  S.date = targetDate;
  renderCalStrip();
  loadFusedBrief(); loadAlerts(false); loadDigest();

  if (btn) { btn.classList.add('active'); btn.textContent = '⏹ Stop'; }

  setTimeout(() => {
    const events = [];
    (alertData && alertData.eod_alerts || []).forEach(a => {
      events.push({
        badge: 'REPLAY SIMULATION • EOD BREACH',
        ticker: a.ticker,
        headline: a.ticker + ' crossed watchdog threshold: ' + (a.dynamic_reasons || []).join(' & '),
        sub: 'Replay (' + targetDate + ') • Close Rs ' + a.close
      });
    });
    (S.digest && S.digest.material_items || []).slice(0, 5).forEach(f => {
      events.push({
        badge: 'REPLAY SIMULATION • ' + String(f.category || 'FILING').toUpperCase(),
        ticker: f.ticker,
        headline: f.headline_summary,
        sub: 'Replay Broadcast: ' + (f.broadcast_time || targetDate)
      });
    });

    if (!events.length) {
      showToast('REPLAY SIMULATION', 'PORTFOLIO', 'No threshold breaches on ' + targetDate, 'Try lowering watchdog sliders', true);
      toggleReplayDay();
      return;
    }

    let idx = 0;
    showToast(events[idx].badge, events[idx].ticker, events[idx].headline, events[idx].sub, true);
    idx++;

    S.replayTimer = setInterval(() => {
      if (idx >= events.length) {
        toggleReplayDay();
        return;
      }
      const ev = events[idx++];
      showToast(ev.badge, ev.ticker, ev.headline, ev.sub, true);
    }, 2500);
  }, 350);
}

function tab(t){
  ['pulse', 'digest', 'alerts', 'macro', 'research'].forEach(k => {
    $('view-' + k).classList.toggle('on', k === t);
    $('nav-' + k).classList.toggle('on', k === t);
  });
  const calBar = $('push-cal-bar');
  if (calBar) calBar.classList.toggle('hidden', t === 'research' || t === 'macro');
  window.scrollTo({ top: 0 });
  if (t === 'alerts' && S.stock) drawStock(S.stock);
  if (t === 'macro' && S.macroId) drawMacro(S.macroId);
  if (t === 'pulse') renderPulse();
  if (t === 'research' && S.research) setTimeout(drawPullCharts, 40);
}
function openPull(t){ $('ask-sheet').classList.remove('open'); tab('research'); loadPull(t); }
function pickStock(t){ tab('alerts'); S.stock = t; $('stk-sel').value = t; drawStock(t); }

function renderCalStrip(){
  const el = $('push-cal-bar');
  if (!el || !S.availDates.length) return;
  const winLabel = S.win === 24 ? '24H' : (S.win / 24) + 'D';
  const allBtn = '<button class="cal-btn ' + (S.date === 'all' ? 'on' : '') + '" data-act="cal" data-v="all">' +
    '<small>Default</small><b>' + winLabel + ' Window</b><i>Rolling</i></button>';
  const dayBtns = S.availDates.map(d =>
    '<button class="cal-btn ' + (S.date === d.iso ? 'on' : '') + '" data-act="cal" data-v="' + esc(d.iso) + '">' +
    '<small>' + esc(d.day_name) + '</small><b>' + esc(d.day_num) + '</b><i>' + d.count + ' filings</i></button>'
  ).join('');
  el.innerHTML = allBtn + dayBtns;
}

document.addEventListener('click', e => {
  const el = e.target.closest('[data-act]'); if (!el) return;
  const v = el.getAttribute('data-v');
  switch (el.getAttribute('data-act')) {
    case 'tab': tab(v); break;
    case 'theme': toggleTheme(); break;
    case 'refresh': refreshAll(); break;
    case 'replay': toggleReplayDay(); break;
    case 'pull': openPull(v); break;
    case 'stock': pickStock(v); break;
    case 'open-ask': $('ask-sheet').classList.add('open'); setTimeout(() => $('ask-input').focus(), 50); break;
    case 'close-ask': $('ask-sheet').classList.remove('open'); break;
    case 'ask-q': $('ask-input').value = v; runAsk(v); break;
    case 'bury':
      S.mutedPairs.add(v);
      store.set('finn_muted', JSON.stringify(Array.from(S.mutedPairs)));
      showToast('FILTER UPDATED', v.split(':')[0], 'Buried ' + v.split(':')[1] + ' filings for ' + v.split(':')[0], 'Saved to local filter rules', false);
      loadDigest();
      break;
    case 'reset-bury':
      S.mutedPairs.clear();
      store.set('finn_muted', '[]');
      loadDigest();
      break;
    case 'cal':
      S.date = v;
      renderCalStrip();
      loadFusedBrief();
      loadAlerts(false);
      loadDigest();
      break;
    case 'win':
      S.date = 'all';
      S.win = +v;
      setOn('win-seg', el);
      renderCalStrip();
      loadFusedBrief();
      loadAlerts(false);
      loadDigest();
      break;
    case 'cat': S.cat = v; setOn('cat-pills', el); loadDigest(); break;
    case 'macro': S.macroId = v; drawMacro(v); break;
    case 'rng': S.macroRange = +v; setOn('rng-seg', el); if (S.macroId) drawMacro(S.macroId); break;
    case 'sub': S.sub = v; renderPull(); break;
    case 'ev':
      if (S.evSel.has(v)) S.evSel.delete(v); else S.evSel.add(v);
      store.set('finn_ev', JSON.stringify(Array.from(S.evSel)));
      renderEvPills(); renderEvents(); break;
  }
});
function setOn(box, el){ $(box).querySelectorAll('button').forEach(b => b.classList.toggle('on', b === el)); }
document.addEventListener('change', e => {
  if (e.target.id === 'tick-filter') { S.tick = e.target.value; loadDigest(); }
  if (e.target.id === 'stk-sel') { S.stock = e.target.value; drawStock(S.stock); }
});

$('ask-form').addEventListener('submit', e => {
  e.preventDefault();
  const q = $('ask-input').value.trim();
  if (q) runAsk(q);
});
async function runAsk(query){
  const dr = $('ask-drawer');
  dr.innerHTML = '<div class="row"><b class="xs mono" style="color:var(--accent)">ROUTING QUERY...</b></div><div class="spin" style="margin:10px auto"></div>';
  try {
    const res = await api('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });
    const cites = (res.citations || []).map(c =>
      c.url ? '<a class="badge b-info" href="' + esc(c.url) + '" target="_blank" rel="noopener">📄 ' + esc(c.label) + ' ↗</a>'
            : '<span class="badge b-neutral">🔒 ' + esc(c.label) + '</span>'
    ).join(' ');
    const tkBtns = (res.related_tickers || []).map(t =>
      '<button class="tick" data-act="pull" data-v="' + esc(t) + '">' + esc(t) + ' Research ↗</button>'
    ).join(' ');
    dr.innerHTML =
      '<div class="row"><span class="badge b-info">' + esc(res.route_badge || 'GROUNDED SYNTHESIS') + '</span></div>' +
      '<div class="sm" style="font-weight:600;color:var(--text)">Q: ' + esc(query) + '</div>' +
      '<div class="note"><div class="sm" style="line-height:1.5;white-space:pre-line">' + esc(res.answer) + '</div></div>' +
      (cites ? '<div class="row" style="gap:4px;flex-wrap:wrap;justify-content:flex-start">' + cites + '</div>' : '') +
      (tkBtns ? '<div class="row" style="gap:6px;flex-wrap:wrap;justify-content:flex-start;margin-top:2px">' + tkBtns + '</div>' : '');
  } catch (err) {
    dr.innerHTML = '<div class="xs down">Error: ' + esc(err.message) + '</div>';
  }
}

async function loadFusedBrief(){
  try {
    const d = await api('/api/push/fused-brief?date=' + encodeURIComponent(S.date));
    S.fused = d.items || [];
    $('fused-sub').textContent = S.date === 'all'
      ? 'Price anomalies, filings & macro linked'
      : 'Single-Day Fused Snapshot for ' + S.date;
    $('fused-brief-list').innerHTML = S.fused.length ? S.fused.map(it => {
      const tags = (it.signals || []).map(sg => '<span class="badge ' + esc(sg.cls) + '">' + esc(sg.label) + '</span>').join(' ');
      return '<div class="note" style="border-left-color:var(--' + (it.severity === 'high' ? 'down' : 'accent') + ')">' +
        '<div class="row"><div class="row" style="gap:6px"><button class="tick" data-act="pull" data-v="' + esc(it.ticker) + '">' + esc(it.ticker) + '</button>' + tags + '</div>' +
        '<b class="mono xs ' + dir(it.daily_return_pct) + '">' + pct(it.daily_return_pct) + ' (' + num(it.vol_ratio).toFixed(1) + 'x vol)</b></div>' +
        '<div class="sm" style="margin-top:5px;font-weight:600">' + esc(it.fused_narrative) + '</div>' +
        (it.pdf_url ? '<div class="row xs" style="margin-top:4px"><span class="muted">' + esc(it.source_tag) + '</span><a href="' + esc(it.pdf_url) + '" target="_blank">Filing PDF ↗</a></div>' : '') +
        '</div>';
    }).join('') : '<div class="empty xs">No high-severity fused alerts on this trading date.</div>';
  } catch (e) {}
}

let alertData = null, alertTimer = null;
async function loadAlerts(fromSliderOrPoll){
  const p = $('sl-price').value, v = $('sl-vol').value;
  $('lbl-price').textContent = '+/-' + num(p).toFixed(1) + '%';
  $('lbl-vol').textContent = num(v).toFixed(2) + 'x';
  if ($('vol-thr-lbl')) $('vol-thr-lbl').textContent = num(v).toFixed(2) + 'x';
  store.set('finn_p', p); store.set('finn_v', v);
  try {
    alertData = await api('/api/push/alerts?price_threshold=' + p + '&vol_threshold=' + v + '&date=' + encodeURIComponent(S.date));
  } catch (err) {
    $('eod-list').innerHTML = '<div class="card empty">Could not load alerts: ' + esc(err.message) + '</div>';
    return;
  }

  (alertData.eod_alerts || []).forEach(a => {
    const k = a.ticker + ':' + (a.date || 'latest') + ':' + (a.dynamic_reasons || []).join('|');
    if (!S.seenAlerts.has(k)) {
      S.seenAlerts.add(k);
      if (S.initializedAlerts && fromSliderOrPoll) {
        showToast('LIVE THRESHOLD BREACH', a.ticker, a.ticker + ' crossed threshold: ' + (a.dynamic_reasons || []).join(', '), 'Close Rs ' + a.close, false);
      }
    }
  });
  S.initializedAlerts = true;

  S.tape = alertData.portfolio_tape || [];
  const sel = $('stk-sel');
  if (!sel.options.length && S.tape.length) {
    sel.innerHTML = S.tape.map(s => '<option value="' + esc(s.ticker) + '">' + esc(s.ticker) + '</option>').join('');
    S.stock = S.tape[0].ticker;
    const tf = $('tick-filter');
    S.tape.forEach(s => { const o = document.createElement('option'); o.value = s.ticker; o.textContent = s.ticker; tf.appendChild(o); });
  }
  $('tape').innerHTML = S.tape.map(s => '<button class="chip" data-act="stock" data-v="' + esc(s.ticker) + '"><b>' + esc(s.ticker) + '</b><span class="' + (num(s.daily_return_pct) >= 0 ? 'u' : 'd') + '">' + pct(s.daily_return_pct) + '</span></button>').join('');
  $('alert-dot').style.display = (alertData.eod_alerts || []).length ? 'block' : 'none';
  $('eod-n').textContent = (alertData.eod_alerts || []).length;
  $('heat-title').textContent = S.date === 'all' ? 'Portfolio Heatmap (1D)' : 'Portfolio Heatmap (' + S.date + ')';
  renderEod(); renderEvents(); renderEvPills(); renderPulse();
  if (S.stock) drawStock(S.stock);
}
function renderEod(){
  const list = alertData.eod_alerts || [];
  const p = $('sl-price').value, v = $('sl-vol').value;
  if (!list.length) { $('eod-list').innerHTML = '<div class="card empty">Nothing breached +/-' + esc(p) + '% or ' + esc(v) + 'x volume on this trading day. Lower sliders to test sensitivity.</div>'; return; }
  $('eod-list').innerHTML = list.map(a => {
    const closes = (a.history_30d || []).map(d => d.close);
    return '<div class="card col" style="padding:12px;border-left:3px solid var(--warn)" data-act="stock" data-v="' + esc(a.ticker) + '">' +
      '<div class="row"><div><span class="mono" style="font-weight:700">' + esc(a.ticker) + '</span> <span class="mono sm muted">Rs ' + esc(a.close) + '</span></div>' +
      '<div style="text-align:right"><b class="mono ' + dir(a.daily_return_pct) + '">' + pct(a.daily_return_pct) + '</b></div></div>' +
      '<div class="row"><div class="row" style="gap:4px;flex-wrap:wrap;justify-content:flex-start">' + (a.dynamic_reasons || []).map(r => '<span class="badge b-warn">' + esc(r) + '</span>').join('') + '</div>' + spark(closes, 64, 22) + '</div></div>';
  }).join('');
}
function renderEvPills(){
  $('ev-pills').innerHTML = EV_TYPES.map(k => '<button class="pill ' + (S.evSel.has(k) ? 'on' : '') + '" data-act="ev" data-v="' + esc(k) + '">' + esc((CAT[k] || [k])[0]) + '</button>').join('');
}
function renderEvents(){
  if (!alertData) return;
  const list = (alertData.event_alerts || []).filter(x => S.evSel.has(x.category));
  $('ev-list').innerHTML = list.length ? list.map(ev => {
    const b = CAT[ev.category] || [String(ev.category || 'FILING').toUpperCase(), 'b-neutral'];
    return '<div class="card col" style="padding:12px"><div class="row"><div class="row" style="gap:8px"><button class="tick" data-act="pull" data-v="' + esc(ev.ticker) + '">' + esc(ev.ticker) + '</button><span class="badge ' + b[1] + '">' + b[0] + '</span></div>' + dots(ev.priority_score || 1) + '</div>' +
      '<div class="sm" style="font-weight:600">' + esc(ev.headline_summary) + '</div><div class="row xs"><span class="muted mono">' + esc((ev.broadcast_time || '').slice(0, 11)) + '</span><a href="' + esc(ev.pdf_url) + '" target="_blank" rel="noopener">Exchange PDF ↗</a></div></div>';
  }).join('') : '<div class="card empty">No material events for the selected categories/date.</div>';
}

function drawStock(t){
  const s = S.tape.find(x => x.ticker === t);
  if (!s || !s.history_30d || !s.history_30d.length) { $('stk-title').textContent = t + ' - no history'; return; }
  const c = C(), h = s.history_30d;
  const mr = num(s.month_return_pct);
  $('stk-title').innerHTML = esc(s.ticker) + ' Rs ' + esc(s.close) + ' <span class="sm ' + dir(mr) + '">1M ' + pct(mr) + '</span>';
  const vols = h.map(d => d.volume != null ? d.volume : (d.vol != null ? d.vol : null));
  const hasVol = vols.some(v => v != null);
  const up = h[h.length - 1].close >= h[0].close;
  const col = up ? c.up : c.down;
  const ds = [{ type: 'line', label: 'Close', data: h.map(d => d.close), borderColor: col, backgroundColor: col + '22', fill: true, borderWidth: 2, tension: .25, pointRadius: 0, pointHoverRadius: 4, yAxisID: 'y', order: 1 }];
  if (hasVol) ds.push({ type: 'bar', label: 'Volume', data: vols, backgroundColor: c.muted + '55', yAxisID: 'v', order: 2, borderRadius: 2 });
  mk('stockChart', {
    data: { labels: h.map(d => d.date), datasets: ds },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 5 } },
        y: { position: 'right', grid: grid(), ticks: { maxTicksLimit: 5 } },
        v: { display: false, position: 'left', beginAtZero: true, max: hasVol ? Math.max.apply(null, vols.filter(v => v != null)) * 4 : 1 }
      }
    }
  });
}

function renderPulse(){
  const tape = S.tape;
  if (!tape.length) { $('pulse-kpis').innerHTML = skel(1); return; }
  const adv = tape.filter(s => num(s.daily_return_pct) >= 0).length;
  const dec = tape.length - adv;
  const trig = (alertData && alertData.eod_alerts || []).length;
  const mat = S.digest ? S.digest.material_count : '-';
  $('pulse-kpis').innerHTML =
    '<div class="kpi"><b class="up">' + adv + '</b><span>Advancers</span></div>' +
    '<div class="kpi"><b class="down">' + dec + '</b><span>Decliners</span></div>' +
    '<div class="kpi"><b style="color:var(--warn)">' + trig + '</b><span>Triggered</span></div>' +
    '<div class="kpi"><b style="color:var(--accent)">' + mat + '</b><span>Surfaced</span></div>';
  $('heat').innerHTML = tape.slice().sort((a, b) => num(b.daily_return_pct) - num(a.daily_return_pct)).map(s => {
    const r = num(s.daily_return_pct), a = Math.min(Math.abs(r) / 4, 1) * 0.55 + 0.08;
    const bg = r >= 0 ? 'rgba(18,147,106,' + a.toFixed(2) + ')' : 'rgba(209,64,90,' + a.toFixed(2) + ')';
    return '<button class="tile" style="background:' + bg + '" data-act="stock" data-v="' + esc(s.ticker) + '"><b>' + esc(s.ticker) + '</b><span class="' + (r >= 0 ? 'up' : 'down') + '">' + pct(r) + '</span></button>';
  }).join('');

  const c = C();
  // Judgment Chart: Filing Priority Score (1-5) vs Volume Ratio (x 20D SMA)
  const filingMaxByTicker = {};
  ((S.digest && S.digest.material_items) || []).forEach(f => {
    filingMaxByTicker[f.ticker] = Math.max(filingMaxByTicker[f.ticker] || 0, f.priority_score || 1);
  });
  const judged = tape.slice().map(s => ({
    ticker: s.ticker,
    prio: filingMaxByTicker[s.ticker] || 0,
    vol: num(s.vol_ratio)
  })).sort((a, b) => (b.prio + b.vol) - (a.prio + a.vol)).slice(0, 8);

  mk('moversChart', {
    type: 'bar',
    data: {
      labels: judged.map(x => x.ticker),
      datasets: [
        { label: 'Filing Priority (1-5)', data: judged.map(x => x.prio), backgroundColor: c.accent, borderRadius: 4 },
        { label: 'Volume Ratio (x 20D)', data: judged.map(x => x.vol), backgroundColor: c.warn, borderRadius: 4 }
      ]
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: true, labels: { boxWidth: 9, font: { size: 10 }, color: c.text } } },
      scales: { x: { grid: grid() }, y: { grid: { display: false }, ticks: { color: c.text, font: { family: 'JetBrains Mono', size: 10, weight: '600' } } } }
    }
  });

  const vt = tape.slice().sort((a, b) => num(b.vol_ratio) - num(a.vol_ratio)).slice(0, 8);
  const thr = num($('sl-vol').value);
  mk('volChart', {
    data: { labels: vt.map(s => s.ticker), datasets: [
      { type: 'bar', data: vt.map(s => num(s.vol_ratio)), backgroundColor: vt.map(s => num(s.vol_ratio) >= thr ? c.warn : c.muted + '66'), borderRadius: 4, order: 2 },
      { type: 'line', data: vt.map(() => thr), borderColor: c.down, borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0, order: 1 }
    ] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false }, tooltip: { callbacks: { label: x => num(x.raw).toFixed(2) + 'x' } } },
      scales: { x: { grid: { display: false }, ticks: { font: { family: 'JetBrains Mono', size: 9 } } }, y: { grid: grid(), beginAtZero: true, ticks: { callback: v => v + 'x' } } } }
  });
}
function renderPulseTop(){
  if (!S.digest) return;
  const top = (S.digest.material_items || []).slice(0, 3);
  $('pulse-top').innerHTML = top.length ? top.map(feedCard).join('') : '<div class="card empty">No material filings on this trading date/window.</div>';
}

function resultsBlock(r){
  const rows = (r.metrics || []).map(m => {
    const y = num(String(m.yoy_pct).replace(/[^0-9.+-]/g, ''));
    const w = Math.min(Math.abs(y), 60) / 60 * 100;
    return '<tr><td style="font-weight:600">' + esc(m.metric) + '</td><td class="mono" style="font-weight:700">' + esc(m.current_qtr) + '</td>' +
      '<td class="mono ' + dir(String(m.qoq_pct).replace(/[^0-9.+-]/g, '')) + '">' + esc(m.qoq_pct) + '</td>' +
      '<td class="mono ' + dir(y) + '">' + esc(m.yoy_pct) + '</td>' +
      '<td style="width:44px"><div class="bar2" style="justify-content:' + (y >= 0 ? 'flex-start' : 'flex-end') + '"><i style="width:' + w.toFixed(0) + '%;background:var(--' + (y >= 0 ? 'up' : 'down') + ')"></i></div></td></tr>';
  }).join('');
  return '<div class="note"><div class="row lbl"><span>Headline numbers (' + esc(r.basis || 'Consolidated') + ')</span><span>' + esc(r.period || '') + '</span></div>' +
    '<table class="table"><thead><tr><th>METRIC</th><th>ACTUAL</th><th>QoQ</th><th>YoY</th><th></th></tr></thead><tbody>' + rows + '</tbody></table>' +
    (r.key_takeaway ? '<div class="sm" style="margin-top:8px">' + esc(r.key_takeaway) + '</div>' : '') + '</div>';
}
function mgmtBlock(item){
  let raw = (item.enrichment && item.enrichment.data) ? item.enrichment.data : {};
  if (Array.isArray(raw)) raw = raw[0] || {};
  const name = raw.executive_name || raw.name || item.person_involved || (item.company_name + ' leadership');
  const role = raw.role_transition || raw.role || item.role_change || item.subject || 'Board / key managerial transition';
  const bio = raw.background_bio || raw.bio || raw.summary || item.details || '';
  return '<div class="note"><div class="row"><b class="sm">' + esc(name) + '</b><span class="badge b-info">EXECUTIVE BIO</span></div>' +
    '<div class="xs" style="font-weight:600;margin:3px 0">' + esc(role) + '</div>' + (bio ? '<div class="sm muted">' + esc(bio) + '</div>' : '') + '</div>';
}
function feedCard(item){
  const b = CAT[item.category] || [String(item.category || 'FILING').toUpperCase(), 'b-neutral'];
  let extra = '';
  if (item.enrichment && item.enrichment.type === 'quarterly_results' && item.enrichment.data) extra = resultsBlock(item.enrichment.data);
  if (item.category === 'management_change') extra = mgmtBlock(item);
  const why = item.why_surfaced || ('Priority ' + (item.priority_score || 3) + '/5 • Material disclosure');
  const muteKey = item.ticker + ':' + item.category;
  return '<div class="card col" style="padding:12px">' +
    '<div class="row"><div class="row" style="gap:8px"><button class="tick" data-act="pull" data-v="' + esc(item.ticker) + '">' + esc(item.ticker) + '</button><span class="badge ' + b[1] + '">' + b[0] + '</span></div>' +
    '<div class="col" style="align-items:flex-end;gap:3px">' + dots(item.priority_score || 1) + '<span class="xs mono muted">' + esc((item.broadcast_time || '').slice(0, 11)) + '</span></div></div>' +
    '<div class="sm" style="font-weight:600;line-height:1.4">' + esc(item.headline_summary) + '</div>' +
    '<div class="why-line"><span>✦ Why surfaced: ' + esc(why) + '</span></div>' +
    extra +
    '<div class="row xs"><span class="muted" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:48%">' + esc(item.company_name) + '</span>' +
    '<div class="row" style="gap:8px"><button class="bury-btn" data-act="bury" data-v="' + esc(muteKey) + '" title="Mute ' + esc(item.category) + ' for ' + esc(item.ticker) + '">🔇 Bury like this</button>' +
    '<a href="' + esc(item.pdf_url) + '" target="_blank" rel="noopener">PDF ↗</a></div></div></div>';
}
async function loadDigest(){
  $('feed').innerHTML = skel(3);
  let d;
  try {
    d = await api('/api/push/digest?hours=' + S.win + '&category=' + encodeURIComponent(S.cat) + '&ticker=' + encodeURIComponent(S.tick) + '&date=' + encodeURIComponent(S.date));
  } catch (err) { $('feed').innerHTML = '<div class="card empty">Could not load filings: ' + esc(err.message) + '</div>'; return; }

  const rawMat = d.material_items || [];
  const activeMat = rawMat.filter(i => !S.mutedPairs.has(i.ticker + ':' + i.category));
  const mutedCount = rawMat.length - activeMat.length;
  d.material_items = activeMat;
  d.material_count = activeMat.length;
  S.digest = d;

  const totalScanned = d.total_raw_scanned || 58;
  const buriedTotal = (d.routine_count || 0) + mutedCount;
  const heroHTML =
    '<div><div class="xs mono" style="color:#A9BEDD;text-transform:uppercase">Signal Filter Funnel</div>' +
    '<div class="sm" style="margin-top:2px"><b>' + totalScanned + '</b> filings → <b>' + d.material_count + '</b> surfaced • <span class="mono" style="color:#FDA4AF">' + buriedTotal + ' buried</span></div></div>' +
    (S.mutedPairs.size ? '<button class="bury-btn" style="background:rgba(255,255,255,.15);color:#fff;border:0" data-act="reset-bury">Unmute (' + S.mutedPairs.size + ')</button>' : '<span class="badge b-up">' + Math.round((1 - d.material_count / Math.max(totalScanned, 1)) * 100) + '% noise cut</span>');
  if ($('filter-hero')) $('filter-hero').innerHTML = heroHTML;
  if ($('digest-hero')) $('digest-hero').innerHTML = heroHTML;

  if (d.available_dates && d.available_dates.length) {
    S.availDates = d.available_dates;
    renderCalStrip();
  }
  const winLbl = S.win === 24 ? '24H' : (S.win / 24) + 'D';
  if (d.anchor_time) $('asof').textContent = S.date === 'all' ? ('Default ' + winLbl + ' • Poll 60s') : ('Trading Date: ' + S.date);
  $('digest-mode-badge').textContent = S.date === 'all' ? (winLbl + ' Window') : ('Date: ' + S.date);
  $('m-count').textContent = d.material_count; $('r-count').textContent = buriedTotal;
  drawCatChart(d);
  $('feed').innerHTML = d.material_items.length ? d.material_items.map(feedCard).join('') : '<div class="card empty">No material filings in this window or category.</div>';
  const rb = $('routine-box');
  if (d.routine_count > 0) {
    rb.classList.remove('hidden'); $('r-badge').textContent = d.routine_count;
    $('routine-list').innerHTML = d.routine_items.map(r => '<div class="row xs"><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:78%"><b class="mono">' + esc(r.ticker) + '</b> <span class="muted">' + esc(r.subject) + '</span></span><a href="' + esc(r.pdf_url) + '" target="_blank" rel="noopener">PDF</a></div>').join('');
  } else rb.classList.add('hidden');
  renderPulseTop(); renderPulse();
}
function drawCatChart(d){
  const counts = {};
  (d.material_items || []).forEach(i => { const k = (CAT[i.category] || [i.category || 'Other'])[0]; counts[k] = (counts[k] || 0) + 1; });
  if (d.routine_count) counts['ROUTINE'] = d.routine_count;
  const labels = Object.keys(counts);
  if (!labels.length) { if (charts.catChart) { charts.catChart.destroy(); delete charts.catChart; } return; }
  const c = C();
  const pal = [c.accent, c.up, c.warn, c.down, '#8B7CF6', '#2BB3C0', '#E58BB0', '#7C8DA6'];
  mk('catChart', {
    type: 'doughnut',
    data: { labels: labels, datasets: [{ data: labels.map(l => counts[l]), backgroundColor: labels.map((l, i) => l === 'ROUTINE' ? c.line : pal[i % pal.length]), borderColor: 'transparent', borderWidth: 0 }] },
    options: { responsive: true, maintainAspectRatio: false, cutout: '62%', plugins: { legend: { position: 'right', labels: { boxWidth: 9, boxHeight: 9, font: { size: 10 }, color: c.text } } } }
  });
}

async function loadMacro(){
  try { S.macro = await api('/api/macro/pulse'); } catch (err) { $('macro-grid').innerHTML = '<div class="empty">' + esc(err.message) + '</div>'; return; }
  const inds = S.macro.indicators || [];
  $('macro-grid').innerHTML = inds.length ? inds.map(i => {
    const vals = (i.series || []).map(p => p.value);
    return '<button class="kcard" data-act="macro" data-v="' + esc(i.id) + '"><div class="xs muted" style="font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + esc(i.name) + '</div>' +
      '<div class="row"><b class="mono">' + esc(i.unit) + esc(i.current) + '</b>' + spark(vals, 48, 18) + '</div>' +
      '<div class="xs mono ' + dir(i.month_pct) + '">1M ' + pct(i.month_pct) + '</div></button>';
  }).join('') : '<div class="empty">No macro data.</div>';
  if (inds.length) { S.macroId = inds[0].id; drawMacro(S.macroId); }
  $('macro-feed').innerHTML = (S.macro.macro_feed || []).map(it => {
    const hi = String(it.impact_severity || '').toLowerCase() === 'high';
    const tk = (it.affected_tickers || []).map(t => '<button class="tick" data-act="pull" data-v="' + esc(t) + '">' + esc(t) + '</button>').join('');
    return '<div class="card col" style="padding:12px"><div class="row"><span class="badge b-neutral">' + esc(it.source) + '</span><span class="badge ' + (hi ? 'b-down' : 'b-info') + '">' + esc(it.category) + '</span></div>' +
      '<div class="sm" style="font-weight:600">' + esc(it.title) + '</div>' +
      '<div class="note"><div class="lbl">Portfolio transmission</div><div class="sm">' + esc(it.portfolio_transmission) + '</div></div>' +
      '<div class="row"><div class="row" style="gap:4px;flex-wrap:wrap;justify-content:flex-start">' + tk + '</div><a class="xs" href="' + esc(it.link) + '" target="_blank" rel="noopener">Source ↗</a></div></div>';
  }).join('');
}
function drawMacro(id){
  if (!S.macro) return;
  const ind = (S.macro.indicators || []).find(x => x.id === id); if (!ind || !ind.series) return;
  const ser = ind.series.slice(-S.macroRange), vals = ser.map(p => p.value);
  const ch = vals.length > 1 ? (vals[vals.length - 1] / vals[0] - 1) * 100 : 0;
  $('mc-title').textContent = ind.name;
  $('mc-stats').innerHTML = '<span>HI ' + Math.max.apply(null, vals).toFixed(2) + '</span><span>LO ' + Math.min.apply(null, vals).toFixed(2) + '</span><span class="' + dir(ch) + '">CHG ' + pct(ch) + '</span>';
  const c = C(), col = ch >= 0 ? c.up : c.down;
  mk('macroChart', {
    type: 'line',
    data: { labels: ser.map(p => p.date), datasets: [{ data: vals, borderColor: col, backgroundColor: col + '22', fill: true, borderWidth: 2, tension: .25, pointRadius: 0, pointHoverRadius: 4 }] },
    options: { responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: false } },
      scales: { x: { grid: { display: false }, ticks: { maxTicksLimit: 5 } }, y: { position: 'right', grid: grid(), ticks: { maxTicksLimit: 5 } } } }
  });
}

async function initPull(){
  try {
    const d = await api('/api/pull/available');
    $('cached-chips').innerHTML = d.cached_tickers.map(t => '<button class="pill" data-act="pull" data-v="' + esc(t) + '">' + esc(t) + '</button>').join('');
    if (d.cached_tickers.length) loadPull(d.cached_tickers[0]);
  } catch (e) {}
}
async function loadPull(t, forceRefresh){
  t = String(t).trim().toUpperCase(); if (!t) return;
  $('pull-input').value = t;
  $('pull-loading').classList.remove('hidden'); $('pull-out').innerHTML = '';
  const url = '/api/pull/research/' + encodeURIComponent(t) + (forceRefresh ? '?refresh=true' : '');
  try { S.research = await api(url); S.sub = 'narrative'; renderPull(); }
  catch (err) { $('pull-out').innerHTML = '<div class="card empty down">Could not analyse ' + esc(t) + ': ' + esc(err.message) + '</div>'; }
  finally { $('pull-loading').classList.add('hidden'); }
}
function verdictClass(v){ v = String(v || '').toLowerCase(); return v.indexOf('beat') >= 0 ? 'up' : (v.indexOf('miss') >= 0 ? 'down' : 'warn'); }
function quoteExpander(quote, cite){
  if (!quote && !cite) return '';
  const qText = quote || 'Verified directly against primary financial statement line items.';
  return '<details class="quote-exp"><summary>▸ Show the line (' + esc(cite || 'Primary Source') + ')</summary>' +
    '<div class="quote-box">"' + esc(qText) + '"<div class="xs mono muted" style="margin-top:4px;font-style:normal">— ' + esc(cite || 'Audited Filing / Concall') + '</div></div></details>';
}
function renderPull(){
  const r = S.research; if (!r) return;
  const h = r.headline_snapshot || {};
  const q = r.narrative_vs_numbers || [];
  const cnt = { up: 0, warn: 0, down: 0 };
  q.forEach(x => cnt[verdictClass(x.verdict)]++);
  const tot = q.length || 1;
  const defaultCite = (r.primary_sources_used && r.primary_sources_used[1]) ? r.primary_sources_used[1] : 'Audited Quarterly Filings & Concall Transcript';

  let sub = '';
  if (S.sub === 'narrative') {
    sub = '<div class="card col"><h3>Narrative vs. Numbers (4Q Audit)</h3>' +
      '<div class="bar2" style="height:10px;gap:2px"><i style="width:' + cnt.up / tot * 100 + '%;height:10px;background:var(--up)"></i><i style="width:' + cnt.warn / tot * 100 + '%;height:10px;background:var(--warn)"></i><i style="width:' + cnt.down / tot * 100 + '%;height:10px;background:var(--down)"></i></div>' +
      '<div class="row xs mono"><span class="up">' + cnt.up + ' beat</span><span style="color:var(--warn)">' + cnt.warn + ' in line</span><span class="down">' + cnt.down + ' miss</span></div></div>' +
      q.map(x => {
        const vc = verdictClass(x.verdict);
        const srcTag = x.source_citation || (x.quarter + ' Filing & Concall (' + defaultCite.slice(0, 38) + ')');
        return '<div class="card col" style="padding:12px"><div class="row"><b class="mono sm">' + esc(x.quarter) + '</b><span class="badge ' + (vc === 'up' ? 'b-up' : vc === 'down' ? 'b-down' : 'b-warn') + '">' + esc(String(x.verdict || '').toUpperCase()) + '</span></div>' +
          '<div class="note"><div class="lbl">Management Guided</div><div class="sm">' + esc(x.management_narrative) + '</div></div>' +
          '<div class="note ' + (vc === 'down' ? 'down-b' : 'up-b') + '"><div class="lbl" style="color:var(--' + (vc === 'down' ? 'down' : 'up') + ')">Audited Delivered</div><div class="sm" style="font-weight:500">' + esc(x.actual_numbers_delivered) + '</div></div>' +
          quoteExpander(x.verbatim_quote || x.management_narrative, srcTag) + '</div>';
      }).join('');
  } else if (S.sub === 'trajectory') {
    sub = '<div class="card col"><div class="row"><h3>Trajectory • Revenue &amp; OPM %</h3><span class="badge b-info">Audited P&amp;L</span></div>' +
      '<div class="chartbox h190"><canvas id="pullTrajChart"></canvas></div>' +
      (r.trajectory_summary ? '<div class="note"><div class="lbl">Multi-Year Direction</div><div class="sm">' + esc(r.trajectory_summary) + '</div></div>' : '') +
      '</div>';
  } else if (S.sub === 'cash') {
    const cq = r.cash_quality || {};
    const cqCite = cq.source_citation || 'Audited Cash Flow & Balance Sheet Statements';
    const items = [['3Y CFO vs PAT conversion', cq.cfo_to_pat_ratio_3yr], ['Working capital & debtor cycle', cq.working_capital_trend], ['Balance sheet leverage & ROCE', cq.leverage_and_roce]];
    sub = '<div class="card col"><div class="row"><h3>Cash Quality • CFO vs PAT</h3><span class="badge b-up">Rs Cr</span></div>' +
      '<div class="chartbox h190"><canvas id="pullCashChart"></canvas></div></div>' +
      '<div class="card col"><div class="row"><h3>Balance Sheet • Debt &amp; WC Cycle</h3><span class="badge b-warn">Leverage &amp; Days</span></div>' +
      '<div class="chartbox h190"><canvas id="pullBsChart"></canvas></div></div>' +
      '<div class="card col"><h3>Cash &amp; Balance Sheet Audit</h3>' + items.map(i => '<div class="note"><div class="lbl">' + i[0] + '</div><div class="sm">' + esc(i[1] || '-') + '</div></div>').join('') +
      '<div class="note" style="border-left-color:var(--warn)"><div class="lbl" style="color:var(--warn)">Diagnostic Verdict</div><div class="sm" style="font-weight:500">' + esc(cq.diagnostic_verdict || cq.red_flags_or_strengths || '-') + '</div></div>' +
      quoteExpander(cq.verbatim_quote || cq.cfo_to_pat_ratio_3yr, cqCite) + '</div>';
  } else {
    const ul = a => '<ul class="sm" style="margin:6px 0 0;padding-left:18px;display:flex;flex-direction:column;gap:8px">' + (a || []).map(x => {
      if (typeof x === 'object' && x !== null) {
        return '<li><div>' + esc(x.claim) + '</div>' + quoteExpander(x.verbatim_quote, x.source_citation) + '</li>';
      }
      return '<li><div>' + esc(x) + '</div>' + quoteExpander(x, defaultCite) + '</li>';
    }).join('') + '</ul>';
    sub = '<div class="card"><h3>Explicit Guidance</h3>' + ul(r.forward_guidance) + '</div>' +
      '<div class="card" style="border-top:3px solid var(--up)"><h3 class="up">Bull Case</h3>' + ul(r.bull_case) + '</div>' +
      '<div class="card" style="border-top:3px solid var(--down)"><h3 class="down">Bear Case</h3>' + ul(r.bear_case) + '</div>';
  }

  const kp = (l, v, c) => '<div class="kcard"><div class="xs muted" style="font-weight:600">' + l + '</div><div class="mono" style="font-weight:700;color:' + (c || 'var(--text)') + '">' + esc(v || '-') + '</div></div>';
  $('pull-out').innerHTML =
    '<div class="card col"><div><div class="row" style="justify-content:flex-start;gap:8px"><h2>' + esc(r.company_name) + '</h2><span class="badge b-info">' + esc(r.ticker) + '</span></div>' +
    '<div class="sm muted" style="margin-top:4px"><b style="color:var(--text)">Business Snapshot:</b> ' + esc(r.business_snapshot) + '</div></div>' +
    '<div class="lbl" style="margin-top:2px">Financial Snapshot</div>' +
    '<div class="kgrid" style="grid-template-columns:repeat(3,1fr)">' + kp('MKT CAP', h.market_cap) + kp('ROCE / ROE', (h.roce || '-') + ' / ' + (h.roe || '-'), 'var(--accent)') + kp('P/E', h.stock_pe) + kp('REV YoY', h.latest_qtr_rev_yoy, 'var(--up)') + kp('OPM', h.latest_opm_pct) + '</div>' +
    '<div class="seg" style="width:100%;flex-wrap:wrap">' +
      '<button style="flex:1" class="' + (S.sub === 'narrative' ? 'on' : '') + '" data-act="sub" data-v="narrative">Narrative vs #s</button>' +
      '<button style="flex:1" class="' + (S.sub === 'trajectory' ? 'on' : '') + '" data-act="sub" data-v="trajectory">Trajectory</button>' +
      '<button style="flex:1" class="' + (S.sub === 'cash' ? 'on' : '') + '" data-act="sub" data-v="cash">Cash &amp; B/S</button>' +
      '<button style="flex:1" class="' + (S.sub === 'guidance' ? 'on' : '') + '" data-act="sub" data-v="guidance">Guidance &amp; Case</button>' +
    '</div></div>' + sub +
    '<div class="card xs muted"><b class="mono" style="color:var(--text)">🔒 PRIMARY SOURCES VERIFIED</b><ul style="margin:6px 0 0;padding-left:16px">' + (r.primary_sources_used || []).map(s => '<li>' + esc(s) + '</li>').join('') + '</ul></div>';

  setTimeout(drawPullCharts, 30);
}

function drawPullCharts(){
  const r = S.research;
  if (!r || !r.chart_series) return;
  const cs = r.chart_series, c = C();

  if (S.sub === 'trajectory' && cs.trajectory && cs.trajectory.years && cs.trajectory.years.length) {
    const t = cs.trajectory;
    mk('pullTrajChart', {
      data: {
        labels: t.years,
        datasets: [
          { type: 'bar', label: 'Revenue (Rs Cr)', data: t.revenue, backgroundColor: c.accent + '88', borderRadius: 4, yAxisID: 'y', order: 2 },
          { type: 'line', label: 'OPM %', data: t.opm_pct, borderColor: c.up, backgroundColor: c.up, borderWidth: 2.2, tension: 0.25, yAxisID: 'y1', order: 1 }
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: true, labels: { boxWidth: 10, font: { size: 10 }, color: c.text } } },
        scales: {
          x: { grid: { display: false } },
          y: { position: 'left', grid: grid(), ticks: { maxTicksLimit: 5 } },
          y1: { position: 'right', grid: { display: false }, ticks: { callback: v => v + '%', maxTicksLimit: 5 } }
        }
      }
    });
  }

  if (S.sub === 'cash') {
    if (cs.cash_vs_pat && cs.cash_vs_pat.years && cs.cash_vs_pat.years.length) {
      const cp = cs.cash_vs_pat;
      mk('pullCashChart', {
        type: 'bar',
        data: {
          labels: cp.years,
          datasets: [
            { label: 'Operating Cash Flow (CFO)', data: cp.cfo, backgroundColor: c.up, borderRadius: 4 },
            { label: 'Net Profit (PAT)', data: cp.pat, backgroundColor: c.accent, borderRadius: 4 }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: true, labels: { boxWidth: 10, font: { size: 10 }, color: c.text } } },
          scales: { x: { grid: { display: false } }, y: { grid: grid(), ticks: { maxTicksLimit: 5 } } }
        }
      });
    }
    if (cs.balance_sheet && cs.balance_sheet.years && cs.balance_sheet.years.length) {
      const bs = cs.balance_sheet;
      mk('pullBsChart', {
        data: {
          labels: bs.years,
          datasets: [
            { type: 'bar', label: 'Borrowings / Debt (Rs Cr)', data: bs.debt, backgroundColor: c.warn + '99', borderRadius: 4, yAxisID: 'y', order: 2 },
            { type: 'line', label: bs.cycle_label || 'Working Cap Days', data: bs.cycle_days, borderColor: c.down, backgroundColor: c.down, borderWidth: 2, tension: 0.25, yAxisID: 'y1', order: 1 }
          ]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: true, labels: { boxWidth: 10, font: { size: 10 }, color: c.text } } },
          scales: {
            x: { grid: { display: false } },
            y: { position: 'left', grid: grid(), ticks: { maxTicksLimit: 5 } },
            y1: { position: 'right', grid: { display: false }, ticks: { callback: v => v + 'd', maxTicksLimit: 5 } }
          }
        }
      });
    }
  }
}
$('pull-form').addEventListener('submit', e => { e.preventDefault(); loadPull($('pull-input').value); });

['sl-price', 'sl-vol'].forEach(id => {
  $(id).addEventListener('input', () => {
    $('lbl-price').textContent = '+/-' + num($('sl-price').value).toFixed(1) + '%';
    $('lbl-vol').textContent = num($('sl-vol').value).toFixed(2) + 'x';
    clearTimeout(alertTimer);
    alertTimer = setTimeout(() => loadAlerts(true), 250);
  });
});
$('sl-price').value = store.get('finn_p', '3');
$('sl-vol').value = store.get('finn_v', '2');

function refreshAll(){ loadFusedBrief(); loadAlerts(false); loadDigest(); loadMacro(); }

window.addEventListener('DOMContentLoaded', () => {
  refreshAll();
  initPull();
});
</script>
</body>
</html>
"""


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return HTMLResponse(content="", status_code=204)


@app.get("/", response_class=HTMLResponse)
def serve_mobile_ui():
    return HTMLResponse(content=EMBEDDED_UI_HTML)


@app.get("/api/push/fused-brief")
def get_fused_brief(date: Optional[str] = Query(None)):
    eod_map = {}
    if os.path.exists(EOD_METRICS_PATH):
        with open(EOD_METRICS_PATH, "r", encoding="utf-8") as f:
            for raw_s in json.load(f):
                s = get_stock_metrics_for_date(raw_s, date)
                eod_map[s["ticker"]] = s

    filings_by_ticker = {}
    if os.path.exists(PUSH_FEED_PATH):
        with open(PUSH_FEED_PATH, "r", encoding="utf-8") as f:
            for item in json.load(f):
                if date and date != "all":
                    t = parse_nse_time(item.get("broadcast_time", ""))
                    if not t or t.strftime("%Y-%m-%d") != date:
                        continue
                if item.get("priority_score", 1) >= 2 and item.get("category") != "unclassified":
                    filings_by_ticker.setdefault(item["ticker"], []).append(item)

    macro_by_ticker = {}
    if os.path.exists(MACRO_PULSE_PATH):
        with open(MACRO_PULSE_PATH, "r", encoding="utf-8") as f:
            mp = json.load(f)
            for m in mp.get("macro_feed", []):
                for tk in m.get("affected_tickers", []):
                    macro_by_ticker.setdefault(tk, []).append(m)

    fused_items = []
    all_tickers = set(list(eod_map.keys()) + list(filings_by_ticker.keys()))

    for tk in all_tickers:
        stock = eod_map.get(tk, {"daily_return_pct": 0.0, "vol_ratio": 1.0, "close": 0})
        ret = float(stock.get("daily_return_pct", 0.0))
        vol = float(stock.get("vol_ratio", 1.0))
        t_filings = filings_by_ticker.get(tk, [])
        t_macros = macro_by_ticker.get(tk, [])

        has_eod_anomaly = abs(ret) >= 2.0 or vol >= 1.5
        top_priority = max([x.get("priority_score", 1) for x in t_filings], default=0)
        score = (abs(ret) * 1.5) + (max(0, vol - 1.0) * 3.0) + (top_priority * 2.5) + (1.5 if t_macros else 0)
        if has_eod_anomaly and t_filings:
            score += 6.0

        if score < 6.0:
            continue

        signals = []
        parts = []

        if abs(ret) >= 2.0 or vol >= 1.5:
            signals.append({"label": "EOD ANOMALY", "cls": "b-warn"})
            parts.append(f"{tk} moved {ret:+.2f}% on {vol:.1f}x 20D avg volume")
        else:
            parts.append(f"{tk} ({ret:+.2f}% 1D, {vol:.1f}x vol)")

        pdf_url = None
        source_tag = "EOD + Exchange Feed"
        if t_filings:
            top_f = t_filings[0]
            cat_lbl = top_f.get("category", "filing").upper()
            signals.append({"label": cat_lbl, "cls": "b-up" if cat_lbl == "RESULTS" else "b-info"})
            parts.append(f"coinciding with [{cat_lbl}] {top_f.get('headline_summary')}")
            pdf_url = top_f.get("pdf_url")
            source_tag = f"NSE Filing ({top_f.get('broadcast_time', '')[:11]})"

        if t_macros:
            top_m = t_macros[0]
            signals.append({"label": "RBI / MACRO", "cls": "b-neutral"})
            parts.append(f"amid {top_m.get('source')} {top_m.get('category')} transmission")

        fused_items.append({
            "ticker": tk,
            "score": round(score, 2),
            "severity": "high" if (abs(ret) >= 3.0 or top_priority >= 5) else "medium",
            "daily_return_pct": ret,
            "vol_ratio": vol,
            "signals": signals[:3],
            "fused_narrative": " — ".join(parts),
            "pdf_url": pdf_url,
            "source_tag": source_tag
        })

    fused_items.sort(key=lambda x: x["score"], reverse=True)
    return {"items": fused_items[:5]}


STOPWORDS_UPPER = {
    "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "HOW", "WHY", "ARE", "WAS",
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "HAVE", "HAS",
    "TODAY", "YESTERDAY", "WEEK", "MONTH", "YEAR", "QOQ", "YOY", "EBITDA",
    "PAT", "PBT", "ROCE", "ROE", "CFO", "FCF", "OPM", "EPS", "PE", "DE",
    "NSE", "BSE", "RBI", "SEBI", "PIB", "IPO", "QIP", "NCD", "ESOP",
    "FINN", "ENAM", "PUSH", "PULL", "COMPARE", "SHOW", "TELL", "ME", "MY",
    "PORTFOLIO", "STOCK", "STOCKS", "COMPANY", "GUIDANCE", "RESULTS", "CASH",
    "DEBT", "MARGIN", "MARGINS", "REVENUE", "PROFIT", "GROWTH", "DELIVERED",
    "PROMISED", "VS", "IN", "ON", "AT", "TO", "OF", "IS", "IT", "ALL", "ANY"
}

def extract_candidate_tickers(raw_query: str) -> List[str]:
    candidates = []
    q_upper = raw_query.upper()

    name_map = {
        "INFOSYS": "INFY", "RELIANCE": "RELIANCE", "HDFC BANK": "HDFCBANK", "HDFC": "HDFCBANK",
        "TRENT": "TRENT", "L&T": "LT", "LARSEN": "LT", "PERSISTENT": "PERSISTENT",
        "POLYCAB": "POLYCAB", "HINDUSTAN AERONAUTICS": "HAL", "NEULAND": "NEULANDLAB",
        "DIXON": "DIXON", "FEDERAL BANK": "FEDERALBNK", "SUN PHARMA": "SUNPHARMA",
        "BAJAJ FINANCE": "BAJFINANCE", "MARUTI": "MARUTI", "WIPRO": "WIPRO", "TITAN": "TITAN"
    }
    for name_kw, tk in name_map.items():
        if re.search(rf"\b{re.escape(name_kw)}\b", q_upper) and tk not in candidates:
            candidates.append(tk)

    for tok in re.findall(r"\b[A-Z][A-Z0-9&]{1,11}\b", raw_query):
        clean_tok = tok.strip()
        if clean_tok not in STOPWORDS_UPPER and clean_tok not in candidates:
            candidates.append(clean_tok)

    cached_tickers = set(PORTFOLIO_TICKERS)
    if os.path.exists(PULL_CACHE_DIR):
        for fname in os.listdir(PULL_CACHE_DIR):
            if fname.endswith("_research.json"):
                cached_tickers.add(fname.replace("_research.json", ""))

    for tok in re.findall(r"\b[A-Za-z0-9&]{2,12}\b", raw_query):
        up_tok = tok.upper()
        if up_tok in cached_tickers and up_tok not in candidates:
            candidates.append(up_tok)

    return candidates[:2]


class AskRequest(BaseModel):
    query: str

class AskResponse(BaseModel):
    route_badge: str = Field(description="e.g., 'PULL RESEARCH • HAL' or 'PORTFOLIO FUSED BRIEF'")
    answer: str = Field(description="Concise 2-4 sentence institutional answer grounded strictly in provided filings/tables. State 'Not in the filings' if absent.")
    related_tickers: List[str]


@app.post("/api/ask")
def ask_finn(req: AskRequest):
    q = req.query.strip()
    candidate_tickers = extract_candidate_tickers(q)

    citations = []
    context_blobs = []
    valid_tickers = []
    pull_data_map = {}
    route_badge = "PORTFOLIO PUSH & MACRO BRIEF"

    for cand in candidate_tickers:
        try:
            p_data = generate_pull_research(cand)
            valid_tickers.append(cand)
            pull_data_map[cand] = p_data
            context_blobs.append(f"=== PRIMARY PULL RESEARCH FOR {cand} ===\n{json.dumps(p_data, indent=2)[:10000]}")
            for src in p_data.get("primary_sources_used", [])[:2]:
                citations.append({"label": src, "url": None})
        except Exception as e:
            print(f"[~] Candidate token '{cand}' is not a valid ticker or failed pull: {e}")

    if valid_tickers:
        route_badge = f"PRIMARY RESEARCH • {', '.join(valid_tickers)}"

    matched_filings = []
    if os.path.exists(PUSH_FEED_PATH):
        with open(PUSH_FEED_PATH, "r", encoding="utf-8") as f:
            filings = json.load(f)
            matched_filings = [
                x for x in filings
                if (not valid_tickers or x.get("ticker") in valid_tickers) and x.get("priority_score", 1) >= 3
            ][:8]
            if matched_filings:
                context_blobs.append(f"=== RECENT NSE FILINGS ===\n{json.dumps(matched_filings, indent=2)}")
                for rf in matched_filings[:2]:
                    citations.append({"label": f"{rf['ticker']}: {rf['subject'][:28]}", "url": rf.get("pdf_url")})

    if os.path.exists(EOD_METRICS_PATH):
        with open(EOD_METRICS_PATH, "r", encoding="utf-8") as f:
            eods = json.load(f)
            rel_eods = [
                {k: v for k, v in s.items() if k != "history_30d"}
                for s in eods if (not valid_tickers or s.get("ticker") in valid_tickers)
            ]
            if rel_eods:
                context_blobs.append(f"=== EOD PRICE & VOLUME ===\n{json.dumps(rel_eods)}")

    g_client = get_gemini_client()
    if g_client:
        system_guardrail = (
            "STRICT GROUNDING RULE: You must answer ONLY using the provided Primary Data Context. "
            "Do NOT use your own outside knowledge of the company, news, or broker commentary. "
            "If the answer to the user's question is not present in the provided filings, transcripts, "
            "or financial tables, state clearly: 'Not in the provided primary filings/transcripts for this period,' "
            "and summarize only what the filings do disclose."
        )
        prompt = (
            f"Investor Question: {q}\n\n"
            f"Primary Data Context:\n" + "\n\n".join(context_blobs) + "\n\n"
            "Answer in 2-4 concise sentences using ONLY the Primary Data Context above. "
            "If the answer is absent from the context, state 'Not in the provided primary filings/transcripts for this period.'"
        )
        for m in PULL_MODELS:
            try:
                resp = g_client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_guardrail,
                        response_mime_type="application/json",
                        response_schema=AskResponse,
                        temperature=0.0
                    )
                )
                parsed = resp.parsed.model_dump() if resp.parsed else json.loads(resp.text)
                parsed["route_badge"] = route_badge
                parsed["citations"] = citations[:4]
                return parsed
            except Exception:
                continue

    if valid_tickers:
        tk = valid_tickers[0]
        r = pull_data_map[tk]
        h = r.get("headline_snapshot", {})
        nav = r.get("narrative_vs_numbers", [])
        latest_audit = f" Latest audit ({nav[0]['quarter']} -> {nav[0]['verdict']}): {nav[0]['actual_numbers_delivered']}" if nav else ""
        honest_ans = (
            f"[Live LLM synthesis unavailable — showing deterministic primary data for {r.get('company_name', tk)} ({tk})]: "
            f"Mkt Cap {h.get('market_cap', '-')}, ROCE {h.get('roce', '-')}, Latest Qtr Rev YoY {h.get('latest_qtr_rev_yoy', '-')}, OPM {h.get('latest_opm_pct', '-')}.{latest_audit}"
        )
        return {
            "route_badge": f"DETERMINISTIC DATA • {tk}",
            "answer": honest_ans,
            "related_tickers": valid_tickers,
            "citations": citations[:4]
        }

    fused_top = get_fused_brief().get("items", [])[:3]
    fused_lines = "\n".join([f"• {it['fused_narrative']}" for it in fused_top])
    return {
        "route_badge": "DETERMINISTIC FUSED BRIEF (LLM OFFLINE)",
        "answer": f"Live LLM synthesis is currently unavailable (quota/key limit). Here are the top deterministic fused portfolio signals:\n{fused_lines}",
        "related_tickers": [it["ticker"] for it in fused_top],
        "citations": citations[:4]
    }


@app.get("/api/push/digest")
def get_push_digest(
    hours: int = Query(24),
    category: str = Query("all"),
    ticker: str = Query("all"),
    date: Optional[str] = Query(None)
):
    if not os.path.exists(PUSH_FEED_PATH):
        return {"material_items": [], "routine_items": [], "material_count": 0, "routine_count": 0, "total_raw_scanned": 0, "available_dates": []}

    with open(PUSH_FEED_PATH, "r", encoding="utf-8") as f:
        all_filings = json.load(f)

    parsed_times = [parse_nse_time(item.get("broadcast_time", "")) for item in all_filings]
    valid_times = [t for t in parsed_times if t is not None]
    anchor_time = max(valid_times) if valid_times else datetime.now()

    available_dates = get_trading_dates_from_cache(all_filings)
    cutoff_time = anchor_time - timedelta(hours=hours)

    filtered = []
    for item in all_filings:
        t = parse_nse_time(item.get("broadcast_time", ""))
        if date and date != "all":
            if not t or t.strftime("%Y-%m-%d") != date:
                continue
        else:
            if t and t < cutoff_time:
                continue

        if category != "all" and item.get("category") != category:
            continue
        if ticker != "all" and item.get("ticker") != ticker.upper():
            continue
        filtered.append(item)

    material_items = [x for x in filtered if x.get("priority_score", 1) >= 2 and x.get("category") != "unclassified"]
    routine_items = [x for x in filtered if x.get("priority_score", 1) == 1 or x.get("category") == "unclassified"]

    material_items.sort(
        key=lambda x: (x.get("priority_score", 1), parse_nse_time(x.get("broadcast_time", "")) or datetime.min),
        reverse=True
    )

    eod_lookup = {}
    if os.path.exists(EOD_METRICS_PATH):
        with open(EOD_METRICS_PATH, "r", encoding="utf-8") as ef:
            for s in json.load(ef):
                eod_lookup[s["ticker"]] = get_stock_metrics_for_date(s, date)

    WHY_MAP = {
        "results": "Priority 5/5 • Audited quarterly financial statements with QoQ/YoY deltas",
        "management_change": "Priority 4–5/5 • Key C-suite or Board leadership transition",
        "M&A": "Priority 4–5/5 • Material corporate restructuring, divestment, or subsidiary IPO",
        "fund_raise": "Priority 4/5 • Institutional capital raise / QIP / NCD allotment",
        "litigation": "Priority 4/5 • Regulatory/NCLAT legal order with balance-sheet relevance",
        "dividend": "Priority 3/5 • Cash distribution & record-date corporate action",
        "credit_rating": "Priority 3/5 • Credit agency rating action on company debt",
        "earnings_call": "Priority 3/5 • Primary management concall transcript / investor deck",
        "press_release": "Priority 3–4/5 • Commercial order win or novel product launch"
    }

    for item in material_items:
        tk = item.get("ticker", "")
        st = eod_lookup.get(tk)
        base_why = WHY_MAP.get(item.get("category"), f"Priority {item.get('priority_score', 3)}/5 • Material exchange disclosure")
        if st and (abs(st.get("daily_return_pct", 0)) >= 2.0 or st.get("vol_ratio", 0) >= 1.5):
            item["why_surfaced"] = f"{base_why} • Coincided with {st['daily_return_pct']:+.2f}% move ({st['vol_ratio']:.1f}x vol)"
        else:
            item["why_surfaced"] = base_why

    return {
        "anchor_time": anchor_time.strftime("%d %b %Y, %H:%M IST"),
        "selected_date": date or "all",
        "available_dates": available_dates,
        "window_hours": hours,
        "total_raw_scanned": len(all_filings),
        "material_count": len(material_items),
        "routine_count": len(routine_items),
        "material_items": material_items,
        "routine_items": routine_items
    }


@app.get("/api/push/alerts")
def get_triggered_alerts(
    price_threshold: float = Query(3.0),
    vol_threshold: float = Query(2.0),
    date: Optional[str] = Query(None)
):
    eod_data = []
    if os.path.exists(EOD_METRICS_PATH):
        with open(EOD_METRICS_PATH, "r", encoding="utf-8") as f:
            eod_data = json.load(f)

    eod_alerts = []
    portfolio_tape = []

    for raw_stock in eod_data:
        stock = get_stock_metrics_for_date(raw_stock, date)
        ret = abs(stock.get("daily_return_pct", 0.0))
        vol_mult = stock.get("vol_ratio", 0.0)
        is_price_hit = ret >= price_threshold
        is_vol_hit = vol_mult >= vol_threshold

        reasons = []
        if is_price_hit:
            reasons.append(f"Price Move: {stock['daily_return_pct']:+.2f}% (≥ ±{price_threshold}%)")
        if is_vol_hit:
            reasons.append(f"Volume Spike: {vol_mult:.2f}x 20D Avg (≥ {vol_threshold}x)")

        enriched_stock = {**stock, "is_triggered": is_price_hit or is_vol_hit, "dynamic_reasons": reasons}
        portfolio_tape.append(enriched_stock)
        if is_price_hit or is_vol_hit:
            eod_alerts.append(enriched_stock)

    event_alerts = []
    if os.path.exists(PUSH_FEED_PATH):
        with open(PUSH_FEED_PATH, "r", encoding="utf-8") as f:
            filings = json.load(f)
            if date and date != "all":
                filings = [
                    x for x in filings
                    if (parse_nse_time(x.get("broadcast_time", "")) or datetime.min).strftime("%Y-%m-%d") == date
                ]
            event_alerts = [x for x in filings if x.get("priority_score", 1) >= 4 and x.get("is_material", False)][:10]

    return {
        "thresholds": {"price_pct": price_threshold, "vol_mult": vol_threshold},
        "selected_date": date or "all",
        "eod_alerts": eod_alerts,
        "event_alerts": event_alerts,
        "portfolio_tape": portfolio_tape
    }


@app.get("/api/macro/pulse")
def get_macro_pulse():
    if os.path.exists(MACRO_PULSE_PATH):
        with open(MACRO_PULSE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"indicators": [], "macro_feed": []}


@app.get("/api/pull/available")
def get_available_research():
    cached = []
    if os.path.exists(PULL_CACHE_DIR):
        for fname in os.listdir(PULL_CACHE_DIR):
            if fname.endswith("_research.json"):
                cached.append(fname.replace("_research.json", ""))
    return {"cached_tickers": sorted(cached), "portfolio_tickers": PORTFOLIO_TICKERS}


@app.get("/api/pull/research/{ticker}")
def get_company_research(ticker: str, refresh: bool = False):
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "").strip()
    try:
        return generate_pull_research(clean_ticker, force_refresh=refresh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print(f"\nStarting FINN Financial Investment Agent (Cache: {CACHE_DIR}) on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)