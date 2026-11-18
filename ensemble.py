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
    from pathlib import Path
    from time import sleep
    target = Path(".ribao-cache") / end / f"index-{code}.csv"
    if target.is_file():
        x = pd.read_csv(target, dtype=str)
        if not x.empty and x.date.iloc[-1] == end:
            return pd.to_numeric(x.set_index("date").close, errors="coerce")
    for attempt in range(3):
        try:
            bs.logout()
            login = bs.login()
            if login.error_code != "0":
                raise RuntimeError(login.error_msg)
            x = read(bs.query_history_k_data_plus(code, "date,close", start, end, "d", "3"))
            if x.empty or x.date.iloc[-1] != end:
                raise RuntimeError(f"{code} 指数未覆盖 {end}")
            target.parent.mkdir(parents=True, exist_ok=True)
            x.to_csv(target, index=False)
            return pd.to_numeric(x.set_index("date").close, errors="coerce")
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"指数 {code} 第 {attempt + 1} 次读取失败：{exc}", flush=True)
            if attempt == 2:
                raise
            sleep(2 ** attempt)


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


def legacy_compare(frames, sessions, book):
    start = max(1, len(sessions) - 252)
    cut = (start + len(sessions)) // 2

    # Only observations that have already reached their exit date inform rankings.
    observations = {}
    for i in range(1, len(sessions) - 1):
        signal, buy = sessions[i - 1:i + 1]
        for code, routes in book.get(signal, {}).items():
            x = frames[code]
            if buy not in x.index:
                continue
            entry = x.loc[buy]
            if entry.open_up or entry.gap <= -.095 or entry.isST != "0":
                continue
            for route in matching(routes, entry):
                j = i + (2 if route == "07" else 1)
                if j >= len(sessions) or sessions[j] not in x.index:
                    continue
                exit_row = x.loc[sessions[j]]
                price = exit_row.open if route in {"15", "16"} and entry.up else exit_row.close
                if price <= 0 or exit_row.chg <= -.095:
                    continue
                result = price * (exit_row.adj / exit_row.close) / (entry.open * entry.adj / entry.close) - 1 - .004
                observations.setdefault(j, []).append((route, result))

    def run(left, right, ordering, capacity):
        cash, positions, nav, fills = 60_000., {}, [60_000.], 0
        history = {}
        for i in range(max(1, left - 60), right):
            for route, result in observations.get(i - 1, ()):
                history.setdefault(route, []).append((i - 1, result))
            if i < left:
                continue
            day, yesterday = sessions[i], sessions[i - 1]
            orders = []
            for code, routes in book.get(yesterday, {}).items():
                if code in positions or day not in frames[code].index or i + 1 >= right:
                    continue
                r = frames[code].loc[day]
                if r.open_up or r.gap <= -.095 or r.isST != "0":
                    continue
                matched = matching(routes, r)
                if not matched:
                    continue
                def evidence(route):
                    sample = [v for j, v in history.get(route, ()) if j >= i - 60]
                    return sum(sample) / (len(sample) + 20)
                if ordering == "常规首笔＋机会竞争":
                    route = max(matched, key=lambda k: (evidence(k), k not in OPPORTUNITY, k))
                    regular = [k for k in matched if k not in OPPORTUNITY]
                    regular_route = max(regular, key=lambda k: (evidence(k), k), default=None)
                else:
                    regular_route = None
                    route = max(matched, key=lambda k: (evidence(k), k))
                overlaps = len({FAMILY.get(k, k) for k in matched})
                key = ((-overlaps, -evidence(route), r.gap) if ordering == "多家族优先"
                       else (-evidence(route), -overlaps, r.gap))
                orders.append((key, code, r, route, regular_route))
            opened = 0
            while orders:
                first_regular = ordering == "常规首笔＋机会竞争" and opened == 0
                available = [o for o in orders if not first_regular or o[4] is not None]
                if not available:
                    break
                pick = min(available, key=lambda o: ((-evidence(o[4]), o[0]) if first_regular else (o[0],), o[1]))
                orders.remove(pick)
                _, code, r, route, regular_route = pick
                if first_regular:
                    route = regular_route
                if len(positions) >= 6 or opened >= capacity:
                    break
                price = r.open * 1.001
                qty = max(0, int((min(cash, 10_000) - 5) / price / 100)) * 100
                if qty < 100:
                    continue
                gross = qty * price
                cash -= gross + max(5., gross * .0003)
                positions[code] = [qty, r.adj / r.close, price, i + (2 if route == "07" else 1), route, 0, route in {"15", "16"} and r.up]
                opened += 1
                fills += 1
            for code, pos in list(positions.items()):
                if day not in frames[code].index:
                    continue
                r = frames[code].loc[day]
                pos[2] = r.adj / pos[1]
                if i < pos[3]:
                    continue
                use_open = pos[6] and r.open > np.floor(r.preclose * 90 + .500001) / 100 + .00001
                if pos[6] and not use_open:
                    continue
                if not use_open and r.close <= np.floor(r.preclose * 90 + .500001) / 100 + .00001:
                    continue
                if not use_open and r.up and (pos[4] == "06" or (pos[4] in {"01", "02", "03", "14"} and pos[5] < 3)):
                    pos[5] += 1
                    continue
                exit_value = r.open * r.adj / r.close / pos[1] if use_open else pos[2]
                gross = pos[0] * exit_value * .999
                cash += gross - max(5., gross * .0003) - gross * .0005
                del positions[code]
            nav.append(cash + sum(p[0] * p[2] for p in positions.values()))
        x = np.asarray(nav)
        ret = x[1:] / x[:-1] - 1
        std = ret.std(ddof=1)
        return (float((x[-1] / x[0]) ** (252 / len(ret)) - 1),
                float(max(1 - x / np.maximum.accumulate(x))),
                float(ret.mean() / std * np.sqrt(252)) if std > 0 else 0., fills, float(x[-1] - x[0]))

    rows = [(rule, cap, run(start, cut, rule, cap), run(cut, len(sessions), rule, cap))
            for rule in ("滚动路线优先", "多家族优先", "常规首笔＋机会竞争") for cap in (1, 2)]
    valid = [row for row in rows if row[2][1] <= .15 and row[2][3] >= 15 and row[2][2] > 0]
    return rows, max(valid, key=lambda row: (row[2][2], row[2][0]), default=None), sessions[start], sessions[cut]


def markdown(frames, codes, sessions, asof, start, bs, read):
    sh = index_data(bs, read, "sh.000001", start, asof)
    small = index_data(bs, read, "sh.000852", start, asof)
    if asof not in sh.index or asof not in small.index or not np.isfinite(sh.at[asof]) or not np.isfinite(small.at[asof]):
        raise RuntimeError("当日上证或中证1000指数缺失")
    book = build_book(frames, sessions, sh, small)
    import allocation
    legacy_rows, _, _, _ = legacy_compare(frames, sessions, book)
    rows, choice, begin, cut, standalone = allocation.compare(frames, sessions, book, matching, OPPORTUNITY)
    out = ["", "### 跨策略共用资金对照", "",
           f"观察期自 {begin} 起，{cut} 起为独立留出段。每段 6 万起步、每笔最多 1 万、最多 6 仓。按路线处理卖点：07 多持一天，06/BF 封板续持，15/16 封板后次日开盘退出。其余按下一交易日收盘代理。",
           "路线先按各自排序和名额提名。滚动分使用相同卖点、费用和延期退出引擎已平仓的提名交易，近60日净收益之和除以（样本数＋20）。门控方案要求至少10笔且分数>0，允许空仓。第二笔增益按同方案两笔减一笔计算。","同股只成交一次，BF 变体和四象限变体分别合并计一个家族。开发段以回撤≤15%、开仓≥15、夏普为正筛选，再按夏普排序。",
           "回测股票来自报告日可取股票池，存在幸存者偏差；重建规则的原历史夏普不能直接沿用。", "",
           "|方法|每日上限|开发 年化／回撤／夏普／笔数／净利|留出 年化／回撤／夏普／笔数／净利|",
           "|---|---:|---|---|"]
    metric = lambda x: f"{x[0]:.2%}／{x[1]:.2%}／{x[2]:.2f}／{x[3]}／¥{x[4]:,.0f}"
    out.extend(f"|{r}|{n}|{metric(a)}|{metric(b)}|" for r,n,a,b in rows)
    out.extend(["", "第二笔带来的留出段净利变化（双笔方案减单笔方案）："])
    for one, two in zip(rows[::2], rows[1::2]):
        out.append(f"- {one[0]}：¥{two[3][4] - one[3][4]:+,.0f}")
    out.extend(["", (f"开发段首选：{choice[0]}，每日最多 {choice[1]} 笔。留出段结果见表。"
                     if choice else "暂无通过开发门槛的综合方案。"), ""])
    out.extend(["", "### 单路线核对", "", "|路线|开发 年化／回撤／夏普／笔数／净利|后段 年化／回撤／夏普／笔数／净利|", "|---|---|---|"])
    out.extend(f"|{route}|{metric(a)}|{metric(b)}|" for route,a,b in standalone)
    out.extend(["", "### 旧调度基线（供核对修改影响）", "", "|方法|上限|开发|后段|", "|---|---:|---|---|"])
    out.extend(f"|{r}|{n}|{metric(a)}|{metric(b)}|" for r,n,a,b in legacy_rows)
    out.extend(["", "当前后段已被研究查看，结果属于回放验证。原始路线尚有重建条件；BF组合补位、四象限原评分等差异仍待还原。", "逐笔与单路线结果同时保存为当天 CSV。"])
    opportunity = ["", "盘后预选须等次日真实开盘确认。默认每笔最多 1 万、整手，买入后下一交易日收盘退出；16 按封板状态决定下一交易日开盘或收盘卖。", ""]
    latest = book.get(asof, {})
    bf_rows = [x.loc[asof] for x in frames.values() if asof in x.index]
    stages = [
        ("基础股票池", lambda r: r.eligible),
        ("当日收盘封板", lambda r: r.eligible and r.up),
        ("反包结构", lambda r: r.eligible and r.up and r.body_break and r.prior_limit_count >= 1 and (r.prior_red or r.prior_touch)),
        ("昨日收阴且未触板未封、20日均额至少2亿", lambda r: r.eligible and r.up and r.body_break and r.prior_limit_count >= 1 and r.prior_red and not r.prior_touch and r.amount20 >= 2e8),
    ]
    ma20 = sh.rolling(20).mean().at[asof]
    gate_open = bool(sh.at[asof] >= ma20)
    out.extend(["", "### BF 路线核验", "",
                f"当前重建版指数代理为上证：收盘 {sh.at[asof]:.2f}，MA20 {ma20:.2f}，门控{'通过' if gate_open else '关闭'}。原研究采用主板等权指数，代理会改变信号。",
                "|逐层条件|数量|", "|---|---:|"])
    out.extend(f"|{name}|{sum(bool(test(r)) for r in bf_rows)}|" for name, test in stages)
    out.extend(["", "01、02、03 当前共用 BF 核心候选。原三路组合中的首板和冠军补位尚未完整接入，下面的零候选仅代表 BF 核心最终名单。"])

    from pathlib import Path
    import json
    active = Path("models/active.json")
    if active.exists():
        model = json.loads(active.read_text(encoding="utf-8"))
        out.extend(["", "### 每周学习排序", "", f"活动模型训练日 {model['trained_at']}。次日先满足各路线开盘条件，再按下列路线分数竞争最多两个名额，同股只买一次。"])
        for route, score in sorted(model["scores"].items(), key=lambda item: -item[1]):
            candidates = [c for c, routes in latest.items() if route in routes]
            out.append(f"- {route} {RULES.get(route, (route,))[0]}：分数 {score:.4f}；盘后候选 {len(candidates)} 只。")
        out.append("盘后名单仅供次日开盘确认，实际成交还需现金、持仓和整手约束。")
    else:
        out.extend(["", "每周训练尚无通过验证的活动模型，暂按各路线原条件展示。"])
    for k, (name, _, _) in RULES.items():
        target = opportunity if k in OPPORTUNITY else out
        label = "（BF核心，组合补位待还原）" if k in {"01", "02", "03"} else ("（重建验证版）" if k in REBUILT else "")
        found = [(c, v[k]) for c, v in latest.items() if k in v]
        target.extend([f"#### {k} {name}{label}", ""])
        for c, (_, lo, hi, ref) in sorted(found, key=lambda t: (-t[1][0], t[0])):
            band = "[-2%, -1%) 优先，[0%, +3%) 补位" if k == "20" else f"{lo:.2%} 至 {hi:.2%}"
            more = f"，并距今日 MA5 参考价 {ref:.2f} 不超过 2%" if ref else ""
            target.append(f"- {c} {codes[c]}：次日开盘缺口 {band}{more}。")
        if not found:
            target.append("BF 核心零候选；组合完整候选状态待还原补位后确认。" if k in {"01", "02", "03"} else "零盘后候选。")
        target.append("")
    return out, opportunity
