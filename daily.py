import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import baostock as bs
import numpy as np
import pandas as pd

FIELDS = "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST,adjustflag"
PENDING = "01–03 BF组合、08–09/13 四象限、11 H1、14 G3、17 BA"
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
    return x.set_index("date")


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


def route_sections(frames, codes, asof, start):
    rows = [(code, x.loc[asof]) for code, x in frames.items() if asof in x.index]

    def section(title, instructions, picks):
        body = ["", f"### {title}", "", instructions, "",
                "|代码|名称|收盘参考|成交额（亿元）|换手率|", "|---|---|---:|---:|---:|"]
        body.extend(f"|{code}|{codes[code]}|{row.close:.2f}|{row.amount / 1e8:.2f}|{row.turn:.2f}%|"
                    for code, row in sorted(picks, key=lambda item: item[0]))
        if not picks:
            body.extend(["", "已覆盖样本中零盘后候选。"])
        return body

    out = section("06 首板最低缺口 Top1",
                  "次日开盘缺口 0%～6%，优先最低缺口，同缺口按信号日成交额降序、代码升序。原回测最多三仓、每笔约权益三分之一；买入后次日收盘首次评估，封板续持。流通市值使用成交额字段外的量价换手估计。",
                  [(code, r) for code, r in rows if r.min_gap])
    out += section("20-A 主力捉妖 Top2",
                   "次日先选 [-2%, -1%)，再用 [0%, +3%) 补足，同档从低到高、代码升序。最多两只、四仓、单笔权益 25%，买入后下一交易日收盘退出，停牌或收盘跌停顺延。",
                   [(code, r) for code, r in rows if r.zhuoyao])
    try:
        index = read(bs.query_history_k_data_plus("sh.000852", "date,close", start, asof, "d", "3"))
        closes = pd.to_numeric(index.close, errors="coerce").dropna()
        if len(closes) < 11 or index.date.iloc[-1] != asof:
            raise RuntimeError("中证1000 当日指数缺失")
        ma = closes.rolling(10).mean()
        rising = ma.iloc[-1] > ma.iloc[-2]
        falling = ma.iloc[-1] < ma.iloc[-2]
        ratio = closes.iloc[-1] / ma.iloc[-1]
        strategy15 = (falling and ratio <= .98) or (rising and ratio < 1)
        strategy16 = rising and ratio > 1
        instructions = "次日高开 2%～5%（含端点），按高开幅度降序、代码升序，最多两只、每只约 1 万元。买入日封板则下一交易日开盘卖，否则下一交易日收盘卖。"
        pool = [(code, r) for code, r in rows if r.first_board and 3 <= r.close < 5]
        for title, active, picks, dest in (
            ("15 2j3 原版", strategy15, pool, out),
            ("16 2j3 强市补充", strategy16, [(c, r) for c, r in pool if 3e8 <= r.amount < 6e8 and r.turn >= 6], None),
        ):
            block = section(title, instructions, picks) if active else ["", f"### {title}", "", "中证1000 市场门控关闭。"]
            if dest is not None:
                dest.extend(block)
            else:
                opportunity = block
    except (RuntimeError, ValueError, OSError) as exc:
        out.extend(["", "### 15 2j3 原版", "", f"数据待核验：{exc}"])
        opportunity = ["", "### 16 2j3 强市补充", "", f"数据待核验：{exc}"]
    return out, opportunity


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
    report.extend(["", f"开发段首选：{chosen[0]}，每日最多 {chosen[1]} 笔。" if chosen else "开发段暂无达到门槛的方案，候选仅观察。", "",
        f"待接入的策略：{PENDING}。以上为 07 内部买法比较，跨策略组合回测待各路线接入。", "",
        "## 二、机会主义信号", "",
        "51 上游形态观察。下列标的为今日恰好四连板，供核对原低吸条件。四连板形态本身不足以确认 51 的买点。", ""])
    four = [f"- {code} {codes[code]}" for code, x in frames.items() if asof in x.index and x.at[asof, "four_board"]]
    report.extend(four or ["本次未见满足观察条件的四连板标的。"])
    report.extend(["", "51 原低吸规则、32-A、33-A、39 和 55 的独立执行条件尚未接入，其余机会主义路线本次暂无可确认的买单计划。", ""])
    extra, opportunity = route_sections(frames, codes, asof, start)
    section_two = report.index("## 二、机会主义信号")
    report[section_two:section_two] = extra + [""]
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
