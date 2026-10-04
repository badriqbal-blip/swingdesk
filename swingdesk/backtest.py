"""Backtest: weekly re-ranking, next-open entries, ATR/structure stops, breakeven + chandelier trail, time stops.

Also: signal diagnostics (forward-return deciles, rank information coefficient) that test whether the ranking
has any information at all — the honest first question before trusting a model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wide(panel: pd.DataFrame, col: str, dates: pd.DatetimeIndex, tickers: list[str]) -> np.ndarray:
    return panel[col].unstack("ticker").reindex(index=dates, columns=tickers).to_numpy(dtype=float)


def run_backtest(scored: pd.DataFrame, prices: dict[str, pd.DataFrame], regime_df: pd.DataFrame, cfg: dict,
                 min_adv: float = 0.0, periods_per_year: int = 252, bench_close: pd.Series | None = None) -> dict | None:
    rk = cfg["risk"]
    reb = int(cfg.get("rebalance_days", 5))
    tickers = sorted(prices.keys())
    tix = {t: j for j, t in enumerate(tickers)}
    dates = scored.index.get_level_values(0).unique().sort_values()
    O = pd.DataFrame({t: prices[t]["Open"] for t in tickers}).reindex(dates).to_numpy(dtype=float)
    L = pd.DataFrame({t: prices[t]["Low"] for t in tickers}).reindex(dates).to_numpy(dtype=float)
    C = pd.DataFrame({t: prices[t]["Close"] for t in tickers}).reindex(dates).to_numpy(dtype=float)
    ATR = _wide(scored, "atr", dates, tickers)
    COMP = _wide(scored, "composite", dates, tickers)
    UP = _wide(scored, "uptrend", dates, tickers)
    ADV = _wide(scored, "adv20", dates, tickers)
    reg = regime_df["regime"].reindex(dates).ffill().bfill().fillna("NEUTRAL").to_numpy()

    valid = (~np.isnan(COMP)).sum(axis=1)
    ok = np.where(valid >= max(5, int(0.6 * len(tickers))))[0]
    if len(ok) == 0 or len(dates) - ok[0] < 60:
        return None
    start_i = int(ok[0])

    comm, slip = float(rk["commission_pct"]) / 100.0, float(rk["slippage_pct"]) / 100.0
    cash, positions, pending, trades = 1.0, {}, [], []
    equity = np.full(len(dates), np.nan)

    def mtm(i):
        tot = cash
        for t, p in positions.items():
            px = C[i, tix[t]]
            tot += p["units"] * (px if np.isfinite(px) else p["last"])
        return tot

    for i in range(start_i, len(dates)):
        # 1) fill yesterday's selections at today's open
        if pending:
            eq_now = mtm(i - 1) if i > 0 else cash
            r = reg[i]
            risk_pct = float(rk["risk_per_trade_pct"].get(r, 0.0)) / 100.0
            for t in pending:
                j = tix[t]
                px, a = O[i, j], ATR[i - 1, j]
                if not (np.isfinite(px) and np.isfinite(a) and a > 0 and px > 0) or risk_pct <= 0:
                    continue
                entry = px * (1 + slip)
                stop = entry - float(rk["stop_atr_mult"]) * a
                R = entry - stop
                units = eq_now * risk_pct / R
                units = min(units, eq_now * float(rk["max_position_pct"]) / 100.0 / entry)
                cost = units * entry * (1 + comm)
                if cost > cash:
                    units = cash / (entry * (1 + comm))
                    cost = units * entry * (1 + comm)
                if units <= 0:
                    continue
                cash -= cost
                positions[t] = {"units": units, "entry": entry, "stop": stop, "R": R, "hi": entry, "bars": 0,
                                "i0": i, "last": entry, "regime": r, "cost_basis": cost}
            pending = []

        # 2) manage open positions on today's bar
        for t in list(positions):
            p = positions[t]
            j = tix[t]
            o, lo, c, a = O[i, j], L[i, j], C[i, j], ATR[i, j]
            if not np.isfinite(c):
                continue
            p["last"] = c
            exit_px = reason = None
            if np.isfinite(o) and o <= p["stop"]:
                exit_px, reason = o, "gap_stop"
            elif np.isfinite(lo) and lo <= p["stop"]:
                exit_px, reason = p["stop"], "stop"
            if exit_px is None:
                p["bars"] += 1
                p["hi"] = max(p["hi"], c)
                gain_r = (c - p["entry"]) / p["R"]
                if gain_r >= float(rk["breakeven_at_r"]):
                    p["stop"] = max(p["stop"], p["entry"])
                if gain_r >= float(rk["trail_after_r"]) and np.isfinite(a):
                    p["stop"] = max(p["stop"], p["hi"] - float(rk["trail_atr_mult"]) * a)
                if p["bars"] >= int(rk["time_stop_days"]) and gain_r < 0.5:
                    exit_px, reason = c, "time_stop"
                elif p["bars"] >= int(rk["max_hold_days"]):
                    exit_px, reason = c, "max_hold"
                elif reg[i] == "CRISIS":
                    exit_px, reason = c, "regime_exit"
            if exit_px is not None:
                px = exit_px * (1 - slip)
                proceeds = p["units"] * px * (1 - comm)
                cash += proceeds
                trades.append({"ticker": t, "entry_date": dates[p["i0"]], "exit_date": dates[i], "entry": p["entry"],
                               "exit": px, "pnl": proceeds - p["cost_basis"], "ret_pct": 100 * (proceeds / p["cost_basis"] - 1),
                               "r_multiple": (px - p["entry"]) / p["R"], "bars": p["bars"], "reason": reason,
                               "regime": p["regime"]})
                del positions[t]

        equity[i] = mtm(i)

        # 3) weekly re-rank → queue entries for the next open
        if (i - start_i) % reb == 0 and i + 1 < len(dates):
            r = reg[i]
            slots = int(rk["max_positions"].get(r, 0)) - len(positions)
            if slots > 0 and r != "CRISIS":
                row = COMP[i].copy()
                good = np.isfinite(row) & (UP[i] == 1) & (np.nan_to_num(ADV[i], nan=0.0) >= min_adv)
                row[~good] = -np.inf
                for j in np.argsort(-row):
                    if slots <= 0 or row[j] == -np.inf:
                        break
                    t = tickers[j]
                    if t in positions or t in pending:
                        continue
                    pending.append(t)
                    slots -= 1

    eq = pd.Series(equity, index=dates).dropna()
    tr = pd.DataFrame(trades)
    metrics = compute_metrics(eq, tr, periods_per_year)
    if bench_close is not None:
        b = bench_close.reindex(eq.index).ffill().dropna()
        if len(b) > 10:
            yrs = (b.index[-1] - b.index[0]).days / 365.25
            metrics["bench_cagr_pct"] = 100 * ((b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else np.nan
            metrics["bench_max_dd_pct"] = 100 * float((b / b.cummax() - 1).min())
    return {"equity": eq, "trades": tr, "metrics": metrics}


def compute_metrics(eq: pd.Series, tr: pd.DataFrame, periods_per_year: int = 252) -> dict:
    m: dict = {"n_trades": int(len(tr))}
    if len(eq) > 2:
        yrs = (eq.index[-1] - eq.index[0]).days / 365.25
        ret = eq.pct_change().dropna()
        m["total_return_pct"] = 100 * (eq.iloc[-1] / eq.iloc[0] - 1)
        m["cagr_pct"] = 100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) if yrs > 0 else np.nan
        m["max_dd_pct"] = 100 * float((eq / eq.cummax() - 1).min())
        m["sharpe"] = float(ret.mean() / ret.std() * np.sqrt(periods_per_year)) if ret.std() > 0 else np.nan
        m["time_in_market_pct"] = 100 * float((ret != 0).mean())
        m["start"], m["end"] = str(eq.index[0].date()), str(eq.index[-1].date())
    if len(tr):
        wins, losses = tr[tr["r_multiple"] > 0], tr[tr["r_multiple"] <= 0]
        m["win_rate"] = float(len(wins) / len(tr))
        m["avg_r"] = float(tr["r_multiple"].mean())
        m["avg_win_r"] = float(wins["r_multiple"].mean()) if len(wins) else np.nan
        m["avg_loss_r"] = float(losses["r_multiple"].mean()) if len(losses) else np.nan
        gp, gl = tr.loc[tr["pnl"] > 0, "pnl"].sum(), -tr.loc[tr["pnl"] < 0, "pnl"].sum()
        m["profit_factor"] = float(gp / gl) if gl > 0 else np.inf
        m["expectancy_r"] = m["avg_r"]
        m["avg_bars"] = float(tr["bars"].mean())
        m["exit_reasons"] = tr["reason"].value_counts().to_dict()
        m["trades_per_year"] = float(len(tr) / max(((eq.index[-1] - eq.index[0]).days / 365.25), 1e-9)) if len(eq) > 2 else np.nan
    return m


def decile_analysis(scored: pd.DataFrame) -> pd.DataFrame | None:
    df = scored[["comp_rank", "fwd_ret"]].dropna()
    if len(df) < 200:
        return None
    dec = (df["comp_rank"] * 10).clip(upper=9.999).astype(int) + 1
    tbl = df.groupby(dec)["fwd_ret"].agg(mean="mean", median="median", hit_rate=lambda s: float((s > 0).mean()), n="count")
    tbl.index.name = "decile"
    tbl[["mean", "median"]] *= 100
    return tbl


def ic_analysis(scored: pd.DataFrame) -> dict | None:
    df = scored[["composite", "fwd_ret"]].dropna()
    if len(df) < 200:
        return None
    ic = df.groupby(level=0).apply(lambda g: g["composite"].corr(g["fwd_ret"], method="spearman") if len(g) >= 8 else np.nan).dropna()
    if len(ic) < 20:
        return None
    # overlapping 15-day labels inflate t-stats; shrink by sqrt(horizon) as a crude Newey-West stand-in
    return {"mean_ic": float(ic.mean()), "ic_std": float(ic.std()), "ic_t_stat": float(ic.mean() / ic.std() * np.sqrt(len(ic) / 15.0)),
            "pct_positive": float((ic > 0).mean()), "n_dates": int(len(ic))}
