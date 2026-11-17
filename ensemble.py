"""Standalone reconstructions and shared-cash replay for the ribao signal universe."""
import numpy as np
import pandas as pd

RULES = {
    "01": ("BF 原版", -0.10, .0801),
    "02": ("BF 三仓双路线", -0.10, .0801),
    "03": ("BF 四仓轻仓", -0.10, .0801),
    "06": ("最低缺口 Top1", 0, .06001),
    "07": ("领涨股放量回撤", 0, .01),
    "08": ("四象限收紧版", None, None),
    "09": ("四象限收紧前", None, None),
    "11": ("H1 多家族低开", -.099, -.01),
    "13": ("四象限仅放宽 R4", None, None),
    "14": ("G3＋MA20", -.099, .0801),
    "15": ("2j3 原版", .019999, .050001),
    "17": ("低开强涨榜 BA", -.05, .01),
    "20": ("主力捉妖", None, None),
    "16": ("2j3 强市补充", .019999, .050001),
    "31": ("S18 首板", -.07, -.02),
    "32-A": ("S18 连板窄缺口", -.04, -.03),
    "33-A": ("S18 深跌窄缺口", -.06, -.05),
    "39": ("二进三回封后接力", -.02, .02),
    "51": ("四连板后 MA5 附近低吸", -.07, -.01),
    "55": ("多因子共振", -.05, .02),
}
OPPORTUNITY = {"16", "31", "32-A", "33-A", "39", "51", "55"}
REBUILT = {"01", "02", "03", "08", "09", "11", "13", "14", "17", "31", "32-A", "33-A", "39", "51", "55"}
FAMILY = {"01": "BF", "02": "BF", "03": "BF", "14": "BF",
          "08": "四象限", "09": "四象限", "13": "四象限", "15": "2j3", "16": "2j3"}


def tags(x):
    up = x.up & x.isST.eq("0")
    ma5 = x.adj.rolling(5).mean()
    prev_median = x.adj.shift(2).rolling(60).median()
    x["below_median"] = x.adj.shift().lt(prev_median)
    x["ma5_ref"] = ma5 / x.adj * x.close
    x["high_ret"] = x.high / x.preclose - 1
    x["prior5"] = x.adj.shift().pct_change(5)
    x["ma10_bias"] = x.adj / x.adj.rolling(10).mean() - 1
    x["amount20"] = x.amount.shift().rolling(20).mean()
    x["amount_ratio"] = x.amount / x.amount.shift(1)
    x["prior_red"] = x.close.shift().lt(x.open.shift())
    x["prior_touch"] = x.high.shift().ge(np.floor(x.preclose.shift() * 110 + .500001) / 100) & ~up.shift().eq(True)
    x["prior_limit_count"] = up.shift(2).rolling(5).sum()
    x["body_break"] = x.adj.gt(pd.concat([x.adj.shift(), (x.adj / x.close * x.open).shift()], axis=1).max(axis=1))
    x["first_clean"] = x.first_board & ~x.open_up
    x["two_board"] = up & up.shift().eq(True) & up.shift(2).eq(False)
    x["third_reseal"] = up & x.open_up & x.low.lt(x.open) & up.shift().rolling(2).sum().eq(2) & up.shift(3).eq(False)
    x["deep"] = x.chg.le(-.06) & x.relative_amount.ge(1)
    x["strong"] = x.chg.ge(.04) & x.relative_amount.ge(1.5)
    x["breakout"] = x.adj.ge(x.adj.shift().rolling(60).max()) & x.relative_amount.ge(1.5)
    x["eligible"] = x.isST.eq("0") & x.close.ge(3) & x.liquid & (np.arange(len(x)) >= 60)
    return x


def index_data(bs, read, code, start, end):
    x = read(bs.query_history_k_data_plus(code, "date,close", start, end, "d", "3"))
    return pd.to_numeric(x.set_index("date").close, errors="coerce")


def build_book(frames, sessions, sh, small):
    book = {}
    advances = pd.DataFrame({c: x.chg for c, x in frames.items()}).reindex(sessions)
    width = advances.gt(0).sum(axis=1) / advances.count(axis=1)
    prior_width = width.shift()
    sh_return = sh.pct_change()
    sh_ma = sh.rolling(20).mean()
    s_ma10, s_ma20 = small.rolling(10).mean(), small.rolling(20).mean()
    previous_ma10 = s_ma10.shift()
    for day in sessions:
        if day not in sh.index or day not in small.index:
            continue
        w = width.at[day]
        trend = sh.at[day] > sh_ma.at[day] and sh_ma.at[day] > sh_ma.shift(5).at[day]
        quadrant = (1 if w >= .55 else 2) + (0 if trend else 2)
        below = small.at[day] < s_ma20.at[day]
        current = [(c, x.loc[day]) for c, x in frames.items() if day in x.index]
        first = [(c, r) for c, r in current if r.first_clean and r.eligible]
        median = np.median([r.amount for _, r in first]) if first else np.nan
        selected = {}

        def add(code, rid, score, bounds=None, ref=None):
            lo, hi = bounds or RULES[rid][1:]
            selected.setdefault(code, {})[rid] = (float(score), lo, hi, ref)

        for c, r in current:
            if r.min_gap:
                add(c, "06", r.amount)
            if r.leader:
                add(c, "07", -r.chg)
            if r.zhuoyao:
                add(c, "20", r.amount)
            if r.first_board and 3 <= r.close < 5:
                if (s_ma10.at[day] < previous_ma10.at[day] and small.at[day] <= .98 * s_ma10.at[day]) or (s_ma10.at[day] > previous_ma10.at[day] and small.at[day] < s_ma10.at[day]):
                    add(c, "15", r.amount)
                if s_ma10.at[day] > previous_ma10.at[day] and small.at[day] > s_ma10.at[day] and 3e8 <= r.amount < 6e8 and r.turn >= 6:
                    add(c, "16", r.amount)
            if not r.eligible:
                continue
            if r.up and r.body_break and r.prior_limit_count >= 1 and (r.prior_red or r.prior_touch) and sh.at[day] >= sh_ma.at[day]:
                points = int(r.prior_limit_count == 1) + int(r.amount20 >= 2.5e8) + int(r.gap >= 0)
                if points >= 2:
                    add(c, "14", points * 1e10 + r.amount20)
                    if r.prior_red and not r.prior_touch and r.amount20 >= 2e8:
                        for rid in ("01", "02", "03"):
                            add(c, rid, points * 1e10 + r.amount20)
            if r.first_clean and below and r.below_median:
                add(c, "31", r.amount)
            if r.two_board and below:
                add(c, "32-A", r.amount)
            if r.deep and below:
                add(c, "33-A", -r.chg)
            if r.strong:
                add(c, "17", r.chg + .02 * min(r.relative_amount, 5))
            if r.third_reseal:
                add(c, "39", r.amount)
            if r.four_board and np.isfinite(r.ma5_ref):
                add(c, "51", r.amount, ref=r.ma5_ref)
            if r.first_clean:
                qual = (
                    quadrant == 1 and len(first) <= 85 and 3 <= r.turn <= 22 and r.amount >= 2.5e8 and -.015 <= r.gap <= .05
                    or quadrant == 2 and prior_width.at[day] >= .45 and r.amount >= max(1.6e8, .85 * median) and r.amount_ratio >= 1.25
                    or quadrant == 3 and sh_return.at[day] >= .001 and r.amount >= 1.8e8 and r.prior5 <= .025 and r.ma10_bias <= .08
                    or quadrant == 4 and sh.at[day] <= .994 * sh_ma.at[day] and -.005 <= r.gap <= .055 and r.amount >= max(2.2e8, 1.4 * median)
                )
                if qual:
                    bounds = {1: (0, .015), 2: (-.04, -.018), 3: (-.055, -.015), 4: (-.04, -.022)}[quadrant]
                    for rid in ("08", "09", "13"):
                        lo, hi = bounds
                        if rid == "09":
                            lo = {1: -.023, 2: -.055, 3: -.055, 4: -.055}[quadrant]
                        if rid == "13" and quadrant == 4:
                            lo = -.065
                        add(c, rid, np.log1p(r.amount / median), (lo, hi))
        ranked = []
        lookup = dict(current)
        for c in set(selected) | {c for c, r in current if r.breakout and r.eligible}:
            r = lookup[c]
            routes = selected.get(c, {})
            families = {FAMILY.get(k, k) for k in routes if k not in OPPORTUNITY}
            if r.breakout and r.eligible:
                families.add("突破")
            ranked.append((len(families), int(r.breakout) + int(r.amount_ratio >= 2.5) * 2 + int(10 <= r.turn < 15), r.amount, c))
        for n, score, amount, c in sorted(ranked, key=lambda x: (-x[0], -x[1], -x[2], x[3]))[:5]:
            if n:
                add(c, "11", n * 10 + score)
            if n >= 2:
                add(c, "55", n * 10 + score)
        book[day] = selected
    return book


def matching(routes, row):
    matched = {}
    for k, (score, lo, hi, ref) in routes.items():
        if k == "20":
            gap_ok = -.02 <= row.gap < -.01 or 0 <= row.gap < .03
        else:
            gap_ok = lo <= row.gap < hi
        if gap_ok and (ref is None or abs(row.open / ref - 1) <= .02):
            matched[k] = score
    return matched


def compare(frames, sessions, book):
    start = max(1, len(sessions) - 252)
    cut = (start + len(sessions)) // 2

    def run(left, right, ordering, capacity):
        cash, positions, nav, fills = 60_000., {}, [60_000.], 0
        for i in range(left, right):
            day, yesterday = sessions[i], sessions[i - 1]
            orders = []
            for code, routes in book.get(yesterday, {}).items():
                if code in positions or day not in frames[code].index or i + 1 >= right:
                    continue
                r = frames[code].loc[day]
                if r.open_up or r.gap <= -.095 or r.isST != "0":
                    continue
                matched = matching({k: v for k, v in routes.items() if k not in OPPORTUNITY}, r)
                if not matched:
                    continue
                overlaps = len({FAMILY.get(k, k) for k in matched})
                key = {"多家族优先": (-overlaps, r.gap), "深低开优先": (r.gap, -overlaps),
                       "历史排名优先": (min(int(k) for k in matched), -overlaps)}[ordering]
                orders.append((key, code, r))
            opened = 0
            for _, code, r in sorted(orders, key=lambda t: (t[0], t[1])):
                if len(positions) >= 6 or opened >= capacity:
                    break
                price = r.open * 1.001
                qty = max(0, int((min(cash, 10_000) - 5) / price / 100)) * 100
                if qty < 100:
                    continue
                gross = qty * price
                cash -= gross + max(5., gross * .0003)
                positions[code] = [qty, r.adj / r.close, price, i + 1]
                opened += 1
                fills += 1
            for code, pos in list(positions.items()):
                if day not in frames[code].index:
                    continue
                r = frames[code].loc[day]
                pos[2] = r.adj / pos[1]
                if i >= pos[3] and r.close > np.floor(r.preclose * 90 + .500001) / 100 + .00001:
                    gross = pos[0] * pos[2] * .999
                    cash += gross - max(5., gross * .0003) - gross * .0005
                    del positions[code]
            nav.append(cash + sum(p[0] * p[2] for p in positions.values()))
        x = np.asarray(nav)
        ret = x[1:] / x[:-1] - 1
        std = ret.std(ddof=1)
        return (float((x[-1] / x[0]) ** (252 / len(ret)) - 1),
                float(max(1 - x / np.maximum.accumulate(x))),
                float(ret.mean() / std * np.sqrt(252)) if std > 0 else 0., fills)

    rows = [(rule, cap, run(start, cut, rule, cap), run(cut, len(sessions), rule, cap))
            for rule in ("多家族优先", "深低开优先", "历史排名优先") for cap in (1, 2)]
    valid = [row for row in rows if row[2][1] <= .15 and row[2][3] >= 15 and row[2][2] > 0]
    return rows, max(valid, key=lambda row: (row[2][2], row[2][0]), default=None), sessions[start], sessions[cut]


def markdown(frames, codes, sessions, asof, start, bs, read):
    sh = index_data(bs, read, "sh.000001", start, asof)
    small = index_data(bs, read, "sh.000852", start, asof)
    if asof not in sh.index or asof not in small.index or not np.isfinite(sh.at[asof]) or not np.isfinite(small.at[asof]):
        raise RuntimeError("当日上证或中证1000指数缺失")
    book = build_book(frames, sessions, sh, small)
    rows, choice, begin, cut = compare(frames, sessions, book)
    out = ["", "### 跨策略共用资金对照", "",
           f"观察期自 {begin} 起，{cut} 起为独立留出段。每段 6 万起步、每笔最多 1 万、最多 6 仓。统一买入后下一交易日收盘卖，跌停顺延。该统一卖点用于比较买入排序，与个别策略原卖点不同。",
           "同股只成交一次，BF 变体和四象限变体分别合并计一个家族。开发段以回撤≤15%、开仓≥15、夏普为正筛选，再按夏普排序。",
           "回测股票来自报告日可取股票池，存在幸存者偏差；重建规则的原历史夏普不能直接沿用。", "",
           "|方法|每日上限|开发 年化／回撤／夏普／笔数|留出 年化／回撤／夏普／笔数|",
           "|---|---:|---|---|"]
    metric = lambda x: f"{x[0]:.2%}／{x[1]:.2%}／{x[2]:.2f}／{x[3]}"
    out.extend(f"|{r}|{n}|{metric(a)}|{metric(b)}|" for r,n,a,b in rows)
    out.extend(["", (f"开发段首选：{choice[0]}，每日最多 {choice[1]} 笔。留出段结果见表。"
                     if choice else "暂无通过开发门槛的综合方案。"), ""])
    opportunity = ["", "盘后预选须等次日真实开盘确认。默认每笔最多 1 万、整手，买入后下一交易日收盘退出；16 按封板状态决定下一交易日开盘或收盘卖。", ""]
    latest = book.get(asof, {})
    for k, (name, _, _) in RULES.items():
        target = opportunity if k in OPPORTUNITY else out
        label = "（重建验证版）" if k in REBUILT else ""
        found = [(c, v[k]) for c, v in latest.items() if k in v]
        target.extend([f"#### {k} {name}{label}", ""])
        for c, (_, lo, hi, ref) in sorted(found, key=lambda t: (-t[1][0], t[0])):
            band = "[-2%, -1%) 优先，[0%, +3%) 补位" if k == "20" else f"{lo:.2%} 至 {hi:.2%}"
            more = f"，并距今日 MA5 参考价 {ref:.2f} 不超过 2%" if ref else ""
            target.append(f"- {c} {codes[c]}：次日开盘缺口 {band}{more}。")
        if not found:
            target.append("零盘后候选。")
        target.append("")
    return out, opportunity
