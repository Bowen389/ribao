import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import baostock as bs
import numpy as np
import pandas as pd

import ensemble

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST,adjustflag"
MAIN = ("sh.600", "sh.601", "sh.603", "sh.605", "sz.000", "sz.001", "sz.002", "sz.003")


def read(result):
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock {result.error_code}: {result.error_msg}")
    frame = result.get_data()
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock {result.error_code}: {result.error_msg}")
    return frame


def features(raw, adjusted):
    x = raw.merge(adjusted[["date", "close", "high", "low"]].rename(columns={"close": "adj", "high": "adj_high", "low": "adj_low"}), on="date", validate="1:1")
    for key in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "adj", "adj_high", "adj_low"]:
        x[key] = pd.to_numeric(x[key], errors="coerce")
    x = x[(x.tradestatus == "1") & (x.open > 0) & (x.preclose > 0)].copy()
    x["gap"] = x.open / x.preclose - 1
    x["chg"] = x.close / x.preclose - 1
    x["up"] = x.close >= ((x.preclose * 110 + 0.500001).apply(np.floor) / 100 - 0.00001)
    x["open_up"] = x.open >= ((x.preclose * 110 + 0.500001).apply(np.floor) / 100 - 0.00001)
    x["liquid"] = x.amount.shift().rolling(20).mean() >= 30_000_000
    x["relative_amount"] = x.amount / x.amount.shift().rolling(20).median()
    x["strength"] = x.adj.pct_change(20)
    eligible = (x.isST == "0") & (x.close >= 3) & x.liquid & (np.arange(len(x)) >= 60)
    x["leader"] = eligible & (x.strength > .15) & (x.chg < -.045) & (x.relative_amount > 1.5)
    up = x.up & (x.isST == "0")
    x["four_board"] = eligible & up & up.shift().rolling(3).sum().eq(3) & up.shift(4).eq(False)
    x["first_board"] = up & up.shift().eq(False)
    x["float_cap"] = x.close * x.volume / (x.turn.where(x.turn > 0) / 100)
    age = (pd.to_datetime(x.date) - pd.Timestamp(x.date.iloc[0])).dt.days if len(x) else pd.Series(dtype=int)
    amplitude = x.adj_high.rolling(5).max() / x.adj_low.rolling(5).min() - 1
    x["min_gap"] = x.first_board & age.ge(50) & x.amount.between(3e8, 19e8) & amplitude.le(.20) & x.float_cap.between(1e9, 20e9)
    high, low = x.adj_high.rolling(10).max(), x.adj_low.rolling(10).min()
    span = (high - low).where(high > low)
    red = (200 * (x.adj - low) / span - 105).ewm(alpha=.2, adjust=False).mean() + 15
    green = (200 * (high - x.adj) / span - 125).ewm(alpha=.2, adjust=False).mean() + 15
    x["zhuoyao"] = ((x.isST == "0") & span.notna() & span.shift().notna()
                    & red.gt(green) & red.shift().le(green.shift()) & x.float_cap.lt(7.5e9)
                    & x.turn.ge(5) & x.turn.lt(15) & x.volume.ge(3e7) & x.volume.lt(8e7))
    return ensemble.tags(x).set_index("date")


def simulate(frames, sessions, mode, count, start_i, end_i):
    cash = 60_000.0
    holdings = {}
    equity = [cash]
    fills = 0
    for i in range(start_i, end_i):
        day, signal = sessions[i], sessions[i - 1]
        candidates = []
        if i + 2 < end_i:
            for code, x in frames.items():
                if code in holdings or signal not in x.index or day not in x.index or not x.at[signal, "leader"]:
                    continue
                row = x.loc[day]
                if row.isST != "0" or not 0 <= row.gap < .01 or row.open_up:
                    continue
                score = x.at[signal, "chg"] if mode == "跌幅" else -x.at[signal, "relative_amount"]
                candidates.append((score, code, row.open, row.adj / row.close))
        for _, code, op, factor in sorted(candidates):
            if count == 0:
                break
            if len(holdings) >= 6:
                break
            price = op * 1.0002
            qty = int((min(cash, 10_000) - 5) / price / 100) * 100
            if qty < 100:
                continue
            gross = qty * price
            cash -= gross + max(5, gross * .0003)
            holdings[code] = [qty, factor, i + 2, op]
            fills += 1
            if sum(h[2] == i + 2 for h in holdings.values()) >= count:
                break
        for code, holding in list(holdings.items()):
            x = frames[code]
            if day not in x.index:
                continue
            row = x.loc[day]
            qty, factor, due, _ = holding
            holding[3] = row.adj / factor
            limit_down = np.floor(row.preclose * 90 + .500001) / 100
            if i >= due and row.close > limit_down + 0.00001:
                gross = qty * holding[3] * .9998
                cash += gross - max(5, gross * .0003) - gross * .0005
                del holdings[code]
        equity.append(cash + sum(h[0] * h[3] for h in holdings.values()))
    nav = np.asarray(equity)
    returns = nav[1:] / nav[:-1] - 1
    risk = returns.std(ddof=1)
    return (
        (nav[-1] / nav[0]) ** (252 / len(returns)) - 1,
        max(1 - nav / np.maximum.accumulate(nav)),
        returns.mean() / risk * np.sqrt(252) if risk > 0 else 0,
        fills,
        len(holdings),
    )


def run(asof):
    start = (date.fromisoformat(asof) - timedelta(days=440)).isoformat()
    cal = read(bs.query_trade_dates(start_date=start, end_date=asof))
    if not len(cal) or cal.iloc[-1].calendar_date != asof:
        raise RuntimeError("交易日历未覆盖请求日期")
    sessions = cal.loc[cal.is_trading_day == "1", "calendar_date"].tolist()
    if asof not in sessions:
        print(f"{asof} 休市")
        return
    universe = read(bs.query_all_stock(asof))
    universe = universe[universe.code.str.startswith(MAIN) & (universe.tradeStatus == "1")]
    codes = dict(zip(universe.code, universe.code_name))
    if len(codes) < 2000:
        raise RuntimeError(f"主板日股票池仅 {len(codes)} 只，拒绝使用不完整快照")
    frames, failed = {}, []
    for code in sorted(codes):
        for attempt in range(2):
            try:
                raw = read(bs.query_history_k_data_plus(code, FIELDS, start, asof, "d", "3"))
                adj = read(bs.query_history_k_data_plus(code, "date,close,high,low", start, asof, "d", "2"))
                if raw.empty or adj.empty or raw.date.iloc[-1] != asof or adj.date.iloc[-1] != asof:
                    raise RuntimeError("当日日线未齐")
                frames[code] = features(raw, adj)
                break
            except (RuntimeError, ValueError, OSError) as exc:
                if attempt:
                    failed.append((code, str(exc)))
    if len(frames) / len(codes) < .98:
        raise RuntimeError(f"{asof} 日线齐备 {len(frames)}/{len(codes)}，拒绝产出信号")
    first = max(len(sessions) - 252, 64)
    mid = (first + len(sessions)) // 2
    variants = []
    for mode in ("跌幅", "放量倍数"):
        for count in (1, 2):
            train = simulate(frames, sessions, mode, count, first, mid)
            test = simulate(frames, sessions, mode, count, mid, len(sessions))
            variants.append((mode, count, train, test))
    eligible = [v for v in variants if v[2][1] <= .15 and v[2][3] >= 15 and v[2][2] > 0]
    chosen = max(eligible, key=lambda v: (v[2][2], v[2][0]), default=None)
    report = [
        f"# {asof} 盘后日报", "",
        f"BaoStock 自取行情 · 日线齐备 {len(frames)}/{len(codes)} · 失败 {len(failed)} 只", "",
        "## 一、历史夏普超过 2 的策略", "",
        "### 07 领涨股放量回撤", "",
        "下一交易日开盘缺口须在 [0%, +1%)。约 1 万元/笔，整手，买入后第 2 个交易日收盘计划退出。",
        "次日停牌、开盘封涨停或无法买入整手时跳过。下表为盘后条件命中，开盘确认后按历史比较选出的排序执行。", "",
        "|代码|股票|收盘|次日参考买价下限|次日参考买价上限（不含）|信号日涨跌|成交额倍数|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    candidates = [(code, x.loc[asof]) for code, x in frames.items() if asof in x.index and x.at[asof, "leader"]]
    candidates.sort(key=lambda item: ((-item[1].relative_amount if chosen and chosen[0] == "放量倍数" else item[1].chg), item[0]))
    report.extend(f"|{code}|{codes[code]}|{r.close:.2f}|{r.close:.2f}|{r.close * 1.01:.4f}|{r.chg:.2%}|{r.relative_amount:.2f}|" for code, r in candidates)
    if not candidates:
        report.extend(["", "07 在已覆盖股票内零候选。"])
    report.extend(["", "### 买入方案历史比较", "",
        f"开发段 {sessions[first]}～{sessions[mid-1]}，留出段 {sessions[mid]}～{asof}。两段各从 6 万元起步。",
        "开发段要求回撤不超过 15%、至少 15 笔且夏普为正，再按夏普排序。每次运行窗口随日期移动。",
        "回测使用当期可取的今日主板池回溯，存在幸存者偏差。留出结果仅作研究验证。", "",
        "|排序/每日上限|开发年化/回撤/夏普/开仓/未平|留出年化/回撤/夏普/开仓/未平|",
        "|---|---|---|",
    ])
    def metrics(v):
        return f"{v[0]:.2%} / {v[1]:.2%} / {v[2]:.2f} / {v[3]} / {v[4]}"
    report.extend(f"|{mode} / {count}|{metrics(train)}|{metrics(test)}|" for mode,count,train,test in variants)
    report.extend(["", f"07 开发段首选：{chosen[0]}，每日最多 {chosen[1]} 笔。" if chosen else "07 开发段暂无达到门槛的方案。"])
    core, opportunity = ensemble.markdown(frames, codes, sessions, asof, start, bs, read)
    report.extend(core)
    report.extend(["", "## 二、机会主义信号", ""])
    report.extend(opportunity)
    if failed:
        report.extend(["### 缺失证券", "", ", ".join(code for code, _ in failed), ""])
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / f"{asof}.md").write_text("\n".join(report), encoding="utf-8")
    print(f"已生成 {asof} 日报")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat())
    args = parser.parse_args()
    date.fromisoformat(args.date)
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {login.error_msg}")
    try:
        run(args.date)
    finally:
        bs.logout()


if __name__ == "__main__":
    main()
