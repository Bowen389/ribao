"""Route-level nomination and settled-outcome portfolio allocation."""
import numpy as np
import pandas as pd

FAMILIES = {"01": "BF", "02": "BF", "03": "BF", "14": "BF",
            "08": "四象限", "09": "四象限", "13": "四象限",
            "15": "2j3", "16": "2j3"}
CAPS = {"01": 1, "02": 1, "03": 1, "06": 1, "07": 1, "14": 2,
        "20": 2, "31": 1, "32-A": 1, "33-A": 1, "39": 1, "51": 1, "55": 1}
RANK_GROUP = {"01": "BF", "02": "BF", "03": "BF"}


def family_set(routes):
    underlying = {FAMILIES.get(k, k) for k in routes if k not in {"11", "55"}}
    # Aggregators contribute no additional evidence when a source route is present.
    return underlying or ({"聚合"} if routes else set())


def nominate(frames, book, sessions, i, matching):
    signal, day = sessions[i - 1], sessions[i]
    buckets = {}
    for code, routes in book.get(signal, {}).items():
        x = frames[code]
        if day not in x.index or signal not in x.index:
            continue
        row, prior = x.loc[day], x.loc[signal]
        down = np.floor(row.preclose * 90 + .500001) / 100
        if row.isST != "0" or row.open_up or row.open <= down + .00001:
            continue
        for route, score in matching(routes, row).items():
            key = (-score, code)
            if route == "06":
                key = (row.gap, -prior.amount, code)
            elif route in {"07", "31", "32-A", "33-A", "11"}:
                key = (row.gap if route != "07" else prior.chg, code)
            elif route == "20":
                key = (0 if row.gap < 0 else 1, row.gap, code)
            elif route in {"15", "16"}:
                key = (-row.gap, code)
            buckets.setdefault(route, []).append((key, code))
    result = {}
    for route, items in buckets.items():
        for rank, (_, code) in enumerate(sorted(items)[:CAPS.get(route, 2)]):
            result.setdefault(code, {})[route] = rank
    return result


def exit_event(x, sessions, buy_i, route):
    """Same schedule and price used for both ranking labels and account exits."""
    first = buy_i + (2 if route == "07" else 1)
    buy_day = sessions[buy_i]
    open_exit = route in {"15", "16"} and bool(x.at[buy_day, "up"])
    extensions = 0
    for j in range(first, len(sessions)):
        day = sessions[j]
        if day not in x.index:
            continue
        row = x.loc[day]
        down = np.floor(row.preclose * 90 + .500001) / 100
        price = row.open if open_exit else row.close
        if price <= down + .00001:
            continue
        if not open_exit and row.up and (route == "06" or route in {"01", "02", "03", "14"} and extensions < 3):
            extensions += 1
            continue
        return j, float(price * row.adj / row.close), open_exit
    return None


def trade(qty, entry, exit_price):
    buy = qty * entry * 1.001
    sell = qty * exit_price * .999
    return sell - max(5, sell * .0003) - sell * .0005 - buy - max(5, buy * .0003)


def compare(frames, sessions, book, matching, opportunity):
    start = max(1, len(sessions) - 252)
    cut = (start + len(sessions)) // 2
    nominations, exits, settled = {}, {}, {}
    for i in range(1, len(sessions)):
        nominations[i] = nominate(frames, book, sessions, i, matching)
        for code, routes in nominations[i].items():
            x = frames[code]
            r = x.loc[sessions[i]]
            qty = max(0, int((10000 - 5) / (r.open * 1.001) / 100)) * 100
            for route in routes:
                event = exit_event(x, sessions, i, route)
                exits[i, code, route] = event
                if event is None or not qty:
                    continue
                j, adjusted_exit, _ = event
                exit_price = adjusted_exit / (r.adj / r.close)
                ret = trade(qty, r.open, exit_price) / (qty * r.open * 1.001 + max(5, qty * r.open * 1.001 * .0003))
                settled.setdefault(j, []).append((RANK_GROUP.get(route, route), code, i, ret))
    # BF variants sharing exactly the same trade contribute one label to the ranking history.
    settled = {j: list(dict.fromkeys(values)) for j, values in settled.items()}
    audits = []

    def run(left, right, method, cap, only=None):
        cash, nav, positions, history, fills = 60000., [60000.], {}, {}, 0
        records = []
        for i in range(max(1, left - 60), right):
            for route, code, buy_i, value in settled.get(i - 1, []):
                history.setdefault(route, []).append((i - 1, value))
            if i < left:
                continue
            day = sessions[i]
            # Opening exits are available to fund today's new orders.
            for code, p in list(positions.items()):
                if p["exit"] and p["exit"][0] == i and p["exit"][2]:
                    cash += p["proceeds"]
                    del positions[code]

            def evidence(route):
                sample = [r for j, r in history.get(RANK_GROUP.get(route, route), []) if j >= i - 60]
                return sum(sample) / (len(sample) + 20), len(sample)

            orders = []
            for code, routes in nominations[i].items():
                if code in positions:
                    continue
                routes = {k: v for k, v in routes.items() if only is None or k == only}
                for route, rank in routes.items():
                    score, count = evidence(route)
                    if method != "修正排序基线" and only is None and (count < 10 or score <= 0):
                        continue
                    key = (-len(family_set(routes)), -score, rank, code) if method == "共振＋正收益门控" else (-score, rank, code)
                    orders.append((key, code, route))
            opened = 0
            while orders and opened < cap and len(positions) < 6:
                available = [o for o in orders if method != "常规首笔＋机会竞争" or opened > 0 or o[2] not in opportunity]
                if not available:
                    break
                key, code, route = min(available)
                orders = [o for o in orders if o[1] != code]
                r = frames[code].loc[day]
                qty = max(0, int((min(cash, 10000) - 5) / (r.open * 1.001) / 100)) * 100
                if not qty:
                    continue
                cost = qty * r.open * 1.001
                cost += max(5, qty * r.open * 1.001 * .0003)
                cash -= cost
                event = exits.get((i, code, route))
                factor = r.adj / r.close
                proceeds = 0.
                if event:
                    gross = qty * event[1] / factor * .999
                    proceeds = gross - max(5, gross * .0003) - gross * .0005
                positions[code] = dict(qty=qty, factor=factor, value=qty * r.open, exit=event, proceeds=proceeds)
                records.append(dict(method=method, capacity=cap, period="开发" if left == start else "后段",
                                    buy=day, code=code, route=route, qty=qty, cost=cost,
                                    sell=sessions[event[0]] if event else None,
                                    pnl=proceeds-cost if event and event[0] < right else None))
                opened += 1
                fills += 1
            for code, p in list(positions.items()):
                if day in frames[code].index:
                    p["value"] = p["qty"] * frames[code].at[day, "adj"] / p["factor"]
                if p["exit"] and p["exit"][0] == i and not p["exit"][2]:
                    cash += p["proceeds"]
                    del positions[code]
            nav.append(cash + sum(p["value"] for p in positions.values()))
        audits.extend(records)
        x = np.asarray(nav)
        ret = x[1:] / x[:-1] - 1
        std = ret.std(ddof=1)
        return (float((x[-1]/x[0])**(252/len(ret))-1), float(max(1-x/np.maximum.accumulate(x))),
                float(ret.mean()/std*np.sqrt(252)) if std else 0., fills, float(x[-1]-x[0]))

    methods = ("修正排序基线", "滚动正收益门控", "共振＋正收益门控", "常规首笔＋机会竞争")
    rows = [(m, cap, run(start, cut, m, cap), run(cut, len(sessions), m, cap)) for m in methods for cap in (1, 2)]
    routes = sorted({r for picks in nominations.values() for rs in picks.values() for r in rs})
    standalone = [(route, run(start, cut, "单路线", 2, route), run(cut, len(sessions), "单路线", 2, route)) for route in routes]
    from pathlib import Path
    out = Path("reports")
    out.mkdir(exist_ok=True)
    pd.DataFrame(audits).to_csv(out / f"{sessions[-1]}-allocation-trades.csv", index=False)
    pd.DataFrame([{"route": k, "development": str(a), "later": str(b)} for k,a,b in standalone]).to_csv(out / f"{sessions[-1]}-route-results.csv", index=False)
    valid = [r for r in rows if r[0] != "修正排序基线" and r[2][1] <= .15 and r[2][3] >= 15 and r[2][2] > 0]
    return rows, max(valid, key=lambda r:(r[2][2],r[2][0]), default=None), sessions[start], sessions[cut], standalone
