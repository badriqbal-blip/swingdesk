# SwingDesk

A multi-factor swing-trade research and risk engine for **US equities/ETFs, UAE equities (DFM/ADX) and crypto spot**, sized for a **20,000 AED cash account**, long-only, no CFDs or leverage, 2–4 week holding horizon.

It does what a systematic desk actually does: detect the market regime, rank every name against its peers on trend, momentum, entry timing, volatility compression, volume, fundamentals and news, classify the setup, size the position off a fixed risk budget with a structure-based stop, and enforce portfolio constraints. Then it turns all of that into **one plan for the whole account** — exactly how many AED go into which tickers across US, UAE and crypto, how much stays in cash and why — and tells you, honestly, how much information the ranking has had historically.

**What it is not:** a crystal ball. No program predicts prices. Good desks make money with small, persistent edges plus strict risk control — the edge here is a disciplined, repeatable process, and the risk engine is the part that keeps you in the game. This is a research tool, not investment advice.

---

## Quick start

```bash
pip install -r requirements.txt

python run.py scan --demo                 # offline run on synthetic data — see the output format in 20 seconds
python run.py validate                    # which tickers your data sources actually cover (run this first with real data)
python run.py scan                        # scan us, uae, crypto → terminal + ./output/ (CSV, JSON, HTML briefing)
python run.py scan --markets us,crypto --top 12 --ml --with-backtest
python run.py backtest --markets us       # event-driven backtest + signal diagnostics
python run.py regime                      # regime read-out only
python run.py size --market us --entry 190 --stop 182 --regime RISK_ON   # ad-hoc sizing
```

Python 3.10+. Open `output/briefing_<date>.html` on your phone or laptop — it is one self-contained file.

Typical run time with real data: 1–3 minutes per market (fundamentals and news are fetched for the top-ranked names only and cached for 24 h). `--no-fundamentals --no-news` makes a pure-technical scan in ~20 s.

---

## How the engine works

| Layer | Module | What it does |
|---|---|---|
| Regime | `regime.py` | Benchmark vs 200/50-day, 50-day slope, realised-vol percentile, breadth (% above 50-day), VIX bands and HYG/IEF credit momentum (US) → `RISK_ON / NEUTRAL / RISK_OFF / CRISIS`. Regime sets factor weights, risk per trade and max positions. CRISIS = no new longs. |
| Features | `indicators.py`, `factors.py` | 40+ causal features per name: EMA stack, Clenow regression momentum (slope×R²), ADX/DI, RSI(14/2), MACD, Bollinger %B & bandwidth percentile, Keltner squeeze, Donchian 20/55, OBV slope, CMF, up/down volume, relative volume, 12-1/6/3-month momentum, relative strength vs benchmark, 52-week-high proximity, ATR%, realised vol, dollar volume. |
| Ranking | `scoring.py` | Each factor is z-scored **cross-sectionally within its market** (crypto is never compared with UAE banks), clipped at ±3, averaged into groups (trend, momentum, timing, volatility, volume, fundamentals, sentiment), then combined with regime-dependent weights. Weights renormalise over whatever groups a name has data for. |
| Fundamentals | `factors.fundamental_factors` | Quality (ROE, gross margin, leverage, FCF yield), growth (revenue, EPS), value (earnings yield, EV/EBITDA), street (recommendation mean, target upside). Snapshot only — it is deliberately **excluded** from the backtest/ML because point-in-time history isn't available for free. |
| Sentiment | `sentiment.py`, `data.py` | Headlines (Yahoo) scored with VADER plus a finance phrase lexicon, recency-weighted; CNN Fear & Greed (US) and alternative.me Fear & Greed (crypto) as contrarian size multipliers at extremes; perp funding rates (crypto) as a crowding penalty. |
| Setup playbook | `scoring.classify_setup` | Breakout · Squeeze (pre-breakout) · Pullback in uptrend · Mean-reversion bounce · Momentum continuation · Extended (wait) · Counter-trend (speculative) · No clean setup. Each carries its own entry tactic and holding period. |
| Event risk | `events.py` | Earnings inside the holding window are flagged, halved or vetoed (`events.earnings_policy`). Macro dates (FOMC) inside the window are shown as notes. |
| Risk engine | `risk.py` | Fixed-fractional sizing (default 1.0 / 0.75 / 0.5 / 0 % of capital by regime). Stop = the lower of entry−2 ATR and the 10-day swing low−¼ ATR, capped at 3.5 ATR. T1 = +1.5R (scale half), T2 = +3R, breakeven at +1R, 2.5-ATR chandelier trail after +1R, time-stop after 15 bars if < +0.5R. Caps: 25 % of capital per position, 5 % total open risk, 100 % deployed, 2 names per sector, 60-day correlation ≤ 0.70. Whole shares for UAE, fractional for US/crypto. |
| ML (optional) | `ml.py` | `--ml`: gradient boosting on cross-sectional feature ranks predicting top-30 % forward performers, trained walk-forward with purge/embargo (no label overlaps the test date). Reports out-of-sample AUC and top-decile lift; blends 30 % into conviction. |
| Validation | `backtest.py` | Weekly re-ranking, fills at the next open ± slippage, the same stop/trail/time rules, equity curve, R-multiples, profit factor, drawdown vs benchmark. Plus two diagnostics that run on every scan: forward-return **deciles** of the composite and the rank **information coefficient** (IC) with a t-stat. If the IC is ~0 for a universe, the ranking has no edge there — believe that number over the pretty table. |

**Conviction** is a logistic transform of the composite (a ranking, not a probability), minus penalties for non-actionable setups, earnings inside the window, and thin liquidity.

---

## Research layers

| Layer | Source (free, no keys) | Where it shows up |
|---|---|---|
| News | Yahoo headlines per name, VADER + finance lexicon, recency-weighted | sentiment factor, "why" text |
| Social mentions | StockTwits streams (US: message rate, bullish share, watchers); CoinGecko (crypto: community sentiment votes, trending list, Reddit activity). No free X/Reddit API exists; UAE has no usable source | small sentiment factors, flags such as "crowd euphoric on social (contrarian caution)" |
| Events | Earnings dates, ex-dividend dates, FOMC dates from config | flags, size halving or veto by policy |
| Macro & commodities | WTI, Brent, gold, silver, copper, natural gas, US 10-year yield, dollar index, VIX, S&P, Nasdaq, long Treasuries, high-yield credit, EM, the UAE equities ETF, Bitcoin | dashboard table + plain-English cross-asset reads (oil → Gulf, copper/gold → growth, dollar → crypto/EM, yields → growth multiples, credit → risk appetite) |
| Supply & demand of futures money | CFTC Commitments of Traders: net speculative position as % of open interest with a 3-year percentile for crude, gold, silver, copper, gas, bitcoin, S&P, dollar, 10-year | positioning table; "crowded long/short" notes |
| Sector rotation | US sector ETFs relative to the S&P over 1 week / 1 month / 3 months | rotation table |
| Micro | Quality, growth, value, street ratings, 30-day EPS estimate revisions | fundamentals factor group |

Macro feeds the plan transparently: an oil downtrend trims the UAE budget ×0.85, a strong dollar trims crypto ×0.9, spiking yields with credit risk-off trims US ×0.9 — each shown in the plan's notes. Physical inventories (EIA) and options positioning are the natural next additions; both need API keys.

## The portfolio plan (where the 20,000 AED goes)

`allocate.py` sits on top of the three market scans:

1. **Market sleeves.** Capital is split from `allocation.market_base_weights` (default US 50 / UAE 20 / crypto 30), then each sleeve is scaled by its regime (RISK_ON ×1.0, NEUTRAL ×0.7, RISK_OFF ×0.35, CRISIS ×0) and by whether the ranking has shown information in that market (rank-IC t-stat ≥ 1.5 keeps the full budget; weaker evidence trims it). Budget a sleeve doesn't earn stays in cash — cash is a position, not a failure.
2. **Names within a sleeve.** The best eligible, uncorrelated names (sector cap, 60-day correlation cap) get inverse-volatility weights tilted by conviction (`conviction_tilt`), so a quiet large-cap gets more AED than a volatile alt-coin for the same risk. Every position is then capped by its own risk budget (AED lost if the stop is hit, from `risk_per_trade_pct`) and by `max_position_pct`.
3. **Account limits.** At most `max_total_positions` across all markets, and total open risk capped at `max_portfolio_heat_pct` (every position is scaled down proportionally if needed).
4. **Live sizing.** With `--live` (the default in the hosted workflow) signals are computed on completed bars only, while sizes, stops and targets of market-entry setups are shifted to the latest quote.
5. **Holdings.** Put what you already own in `holdings.yaml` and the plan reviews it (HOLD / HOLD-ADD / EXIT with the reason and the stop to use), counts its value against the sleeve, and only proposes new money for the remainder:

```yaml
holdings:
  - {market: us, ticker: NVDA, units: 3, entry: 150.0, stop: 140.0}
  - {market: uae, ticker: EMAAR.AE, units: 300, entry: 8.2}
  - {market: crypto, ticker: BTC/USDT, units: 0.01, entry: 90000}
```

The plan is the first section of the briefing and the terminal output, and is saved as `plan_<stamp>.csv` plus the `plan` object in `summary_<stamp>.json`. The risk line under it is the number to respect: what you lose if every stop is hit, and the typical one-day swing of the whole book.

## Reading the output

Terminal and HTML show, per market: the regime line, then the ranked names with setup, conviction, last price, planned entry, stop (and %), T1/T2, units and AED size, AED at risk, ATR%, a plain-English "why", flags, and a tick for the names the portfolio constraints actually select. Below that: the selected portfolio with open risk vs cap, and the signal-quality line.

`output/signals_<market>_<stamp>.csv` has every column (factor group scores, raw indicators, plan fields, skip reasons). `summary_<stamp>.json` is machine-readable for automation. `trades_<market>_<stamp>.csv` appears with `--with-backtest`.

A sensible weekly routine: Sunday night run `scan --with-backtest`; place limit/stop orders Monday morning per `entry_note`; manage with `exit_rule`; re-run mid-week only to check regime changes. Don't add a new position the day before earnings or an FOMC decision unless that is the thesis.

---

## Configuration (`config.yaml`)

Everything is a knob: capital, AED peg, horizon, risk per regime, position/heat caps, stop and trail multiples, liquidity floors, earnings policy, macro dates, factor-group weights per regime, ML blend, crypto exchange, benchmarks, cache TTLs and the three universes. Edit the universes freely — the validate command tells you which symbols return data.

Note that `max_positions × risk_per_trade` can exceed `max_portfolio_heat_pct` on purpose (6 × 1 % > 5 %): the heat cap, not the slot count, decides how many full-size positions you can hold in RISK_ON.

---

## Data sources and their limits

- **Yahoo Finance via `yfinance`** — free, delayed, unofficial. Adjusted daily bars; fundamentals and news are patchy for UAE names. UAE symbols are `<exchange symbol>.AE` (e.g. `EMAAR.AE`, `FAB.AE`).
- **ccxt** — public daily OHLCV from Binance by default (falls back to OKX → Kraken → Yahoo `BTC-USD`). If your location blocks an exchange, change `crypto.exchange`.
- **CNN Fear & Greed** is an unofficial endpoint and may stop responding; the scan degrades gracefully. **alternative.me** (crypto) is stable.
- Everything network-facing is wrapped: a failing source removes a factor group (weights renormalise) instead of killing the scan.

Known biases: the universe is hand-picked current constituents (survivorship bias in backtests); fills are modelled at the next open ± 0.05 % slippage and 0.10 % commission per side — check these against your broker; gap-throughs beyond the stop are modelled on the open only.

---

## Account notes for 20k AED

- 20,000 AED ≈ 5,446 USD at the 3.6725 peg; USD/USDT positions carry no FX risk for an AED account.
- At 1 % risk (200 AED) and a 7 % stop, a position is ~2,900 AED — five to six of those fill the account, which is exactly why the heat and exposure caps exist.
- Fractional shares make US names workable at this size; if your broker lacks them, set `fractional_shares_us: false` and some high-priced names will be skipped as "too small for minimum lot".
- Costs matter at this size: the planner flags any trade where round-trip costs exceed 15 % of the risk budget.
- No shorting (no CFDs): in RISK_OFF the system shrinks and in CRISIS it goes to cash. Defensive ETFs (GLD, TLT, XLU, XLP) are in the US universe so the momentum ranking can rotate into them.

---

## Deploying so it runs by itself

`.github/workflows/scan.yml` runs the scan on GitHub's servers every weekday after the US close and on Sunday (with the backtest), then publishes the HTML briefing to GitHub Pages. No computer of yours needs to be on, and it works from a phone:

1. Create a public repository (Pages is free on public repos; the config holds no secrets) and enable Pages once: Settings → Pages → Source: GitHub Actions.
2. Upload `swingdesk.zip` to the repository root (Add file → Upload files, or `github.com/<user>/<repo>/upload/main`).
3. Create `.github/workflows/scan.yml` with the contents of the workflow file in this package (Add file → Create new file). The run unpacks the zip into the repo, commits the code, scans and publishes.

**Save it like an app.** The page ships a web-app manifest and icon. On iPhone: open the briefing in Safari → Share → "Add to Home Screen". On Android: Chrome menu → "Add to Home screen" / "Install app". The icon opens the latest briefing full-screen; the address never changes.

Updating later is the same move: upload a new `swingdesk.zip` and the next run unpacks it, keeping your `config.yaml` and `holdings.yaml` untouched. The hosted schedule scans every 4 hours (crypto never closes; equities use completed bars) and runs the backtest on Sundays. The briefing lives at `https://<user>.github.io/<repo>/`, downloads under `/files.html`, and the latest outputs are also committed to `history/latest/` in the repo. Running it from your own machine instead: `python run.py scan --live` on a cron/Task Scheduler entry.

The program never connects to a broker or places orders; it produces a plan for you to execute.

## Extending it

Point-in-time fundamentals (e.g. a paid provider) would let fundamentals enter the backtest and ML honestly; intraday bars would allow tighter entries; an options-implied-move check before earnings; broker API hooks for order placement. The code is modular so each is a single file.

---

*Research tool for your own decisions. Not investment, legal or tax advice. Past performance — real or backtested — is not indicative of future results.*
