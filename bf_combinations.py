"""Three BF/first-board/champion queues with independent portfolio budgets."""
import numpy as np
import pandas as pd
from pathlib import Path

CONFIG = {"01": (3, (0.25,)), "02": (3, (0.25, 0.15)), "03": (4, (0.20, 0.15))}


def add_features(x):
    volume = x.volume / x.volume.rolling(10).mean()
    turnover = x.turn.rolling(40).mean()
    to = pd.Series(np.nan, index=x.index)
    for lo, hi in ((-np.inf, 1), (1, 3), (3, 7), (7, np.inf)):
        selected = turnover.where(turnover.ge(lo) & turnover.lt(hi))
        minimum, maximum = selected.expanding().min(), selected.expanding().max()
        to.loc[selected.notna()] = (-(selected-minimum)/(maximum-minimum+1e-12)).loc[selected.notna()]
    streak = []
    count = 0
    for value in x.up:
        count = count + 1 if value else 0
        streak.append(count)
    ma5, ma10, ma20 = (x.adj.rolling(n).mean() for n in (5, 10, 20))
    x["champ_vr"] = volume
    x["champ_to"] = to
    x["champ_os"] = x.gap
    x["champ_lu"] = streak
    x["champ_pos"] = .4 * (ma5.gt(ma10) & ma10.gt(ma20)) + .6 * x.adj / x.adj_high.rolling(120).max()
    x["champ_amp"] = (x.high-x.low)/x.preclose
    low, high = x.adj_low.rolling(60).min(), x.adj_high.rolling(60).max()
    x["fb_pos60"] = (x.adj-low)/(high-low).where(high>low)
    return x


def build(frames, sessions, small, csi300):
    changes = pd.DataFrame({c: x.chg for c,x in frames.items()}).reindex(sessions)
    equal = 1000 * (1 + changes.mean(axis=1).fillna(0)).cumprod()
    equal_ma = equal.rolling(20).mean()
    small_ma, csi_ma = small.rolling(20).mean(), csi300.rolling(120).mean()
    books, audits = {}, {}
    factors = ("champ_vr", "champ_to", "champ_os", "champ_lu", "champ_pos", "champ_amp")
    weights = np.array([1.695, 1.687, 1.375, 1.233, 1.205, -.598])
    for day in sessions:
        current = {c:x.loc[day] for c,x in frames.items() if day in x.index}
        pool = {c:r for c,r in current.items() if r.eligible}
        bf = [(c,r) for c,r in pool.items() if r.up and r.body_break and r.prior_limit_count>=1
              and (r.prior_red or r.prior_touch)]
        scored = [(int(r.prior_limit_count==1)+int(r.amount20>=2.5e8)+int(r.gap>=0),c,r) for c,r in bf]
        graded = [v for v in scored if v[0]>=2]
        highest = max((v[0] for v in graded), default=0)
        graded = [v for v in graded if v[0]==highest]
        bf_gate = bool(equal.at[day]>=equal_ma.at[day])
        fb_gate = day in small.index and small.at[day]<small_ma.at[day]
        ch_gate = day in csi300.index and csi300.at[day]>csi_ma.at[day]
        routes = {}
        def add(code, source, score, lo, hi):
            routes.setdefault(code, []).append(dict(source=source, score=float(score), lo=lo, hi=hi))
        for grade,c,r in graded:
            if bf_gate and r.amount20>=2e8 and r.prior_red and not r.prior_touch:
                add(c,"BF",r.amount20,-.10,.08000001)
        if fb_gate:
            for c,r in pool.items():
                if r.first_clean and r.amount>=1e8 and r.below_median and np.isfinite(r.fb_pos60):
                    if r.fb_pos60<.7:
                        add(c,"FB",2+r.amount/1e15,-.025,.05000001)
                    elif r.fb_pos60<1:
                        add(c,"FB",1+r.amount/1e15,-.05,-.01999999)
        champions = []
        if ch_gate:
            matrix = pd.DataFrame({c:[r[f] for f in factors] for c,r in pool.items()}).T
            if len(matrix):
                z = ((matrix-matrix.mean())/(matrix.std(ddof=1)+1e-12)).clip(-3,3).fillna(0)
                scores = z.to_numpy() @ weights
                champions = sorted(zip(matrix.index, scores),key=lambda v:(-v[1],v[0]))
                if champions:
                    c,score=champions[0]
                    add(c,"CH",score,-.05,.06000001)
        books[day] = routes
        audits[day] = dict(equal=float(equal.at[day]), ma20=float(equal_ma.at[day]),
                           bf_gate=bf_gate, fb_gate=bool(fb_gate), ch_gate=bool(ch_gate),
                           structure=len(bf), graded=len(graded), eligible=len(pool),
                           BF=sum(any(v["source"]=="BF" for v in rs) for rs in routes.values()),
                           FB=sum(any(v["source"]=="FB" for v in rs) for rs in routes.values()),
                           CH=sum(any(v["source"]=="CH" for v in rs) for rs in routes.values()))
    return books,audits


def choose(options, row):
    priority={"BF":0,"FB":1,"CH":2}
    valid=[v for v in options if v["lo"]<=row.gap<v["hi"]]
    return min(valid,key=lambda v:priority[v["source"]]) if valid else None


def ranked(books,frames,sessions,i):
    signal,day=sessions[i-1],sessions[i]
    result=[]
    for c,options in books.get(signal,{}).items():
        if day not in frames[c].index:
            continue
        r=frames[c].loc[day]
        if r.isST!="0" or r.open_up or r.open<=np.floor(r.preclose*90+.500001)/100+.00001:
            continue
        for v in options:
            if not v["lo"]<=r.gap<v["hi"]:
                continue
            rank=(0,-v["score"],0,c) if v["source"]=="BF" else ((1,-int(v["score"]),r.gap,c) if v["source"]=="FB" else (2,-v["score"],0,c))
            result.append((rank,c,v["source"]))
    return sorted(result)


def replay(frames,sessions,books,left,right,rid):
    from allocation import exit_event
    slots,budgets=CONFIG[rid]
    cash=30000.
    positions={}
    nav=[cash]
    trades=[]
    for i in range(left,right):
        day=sessions[i]
        equity=nav[-1]
        sources=set()
        opened=0
        for _,c,source in ranked(books,frames,sessions,i):
            if opened>=len(budgets) or len(positions)>=slots:
                break
            if c in positions or source in sources:
                continue
            r=frames[c].loc[day]
            budget=min(cash,equity*budgets[opened]*.99)
            if source=="FB":
                budget=min(budget,frames[c].at[sessions[i-1],"amount"]*.001)
            qty=max(0,int((budget-5)/(r.open*1.001)/100))*100
            if not qty:
                continue
            cost=qty*r.open*1.001
            cost+=max(5,cost*.0003)
            cash-=cost
            event=exit_event(frames[c],sessions,i,"01" if source=="BF" else "FB")
            positions[c]=dict(qty=qty,factor=r.adj/r.close,event=event,value=qty*r.open,
                              cost=cost,source=source,buy=day)
            sources.add(source)
            opened+=1
        for c,p in list(positions.items()):
            if day in frames[c].index:
                p["value"]=p["qty"]*frames[c].at[day,"adj"]/p["factor"]
            event=p["event"]
            if event and event[0]==i:
                gross=p["qty"]*event[1]/p["factor"]*.999
                proceeds=gross-max(5,gross*.0003)-gross*.0005
                cash+=proceeds
                trades.append(dict(strategy=rid,source=p["source"],code=c,buy=p["buy"],sell=day,
                                   qty=p["qty"],cost=p["cost"],pnl=proceeds-p["cost"]))
                del positions[c]
        nav.append(cash+sum(p["value"] for p in positions.values()))
    x=np.array(nav);r=x[1:]/x[:-1]-1
    result=((x[-1]/x[0])**(252/len(r))-1,max(1-x/np.maximum.accumulate(x)),
            r.mean()/r.std(ddof=1)*np.sqrt(252) if r.std(ddof=1)>0 else 0,len(trades),x[-1]-x[0])
    return result,trades


def report(frames,codes,sessions,books,audits,asof):
    a=audits[asof]
    out=["","### BF 三路组合核验","",
         f"主板等权指数（当前覆盖池重建）{a['equal']:.2f}，MA20 {a['ma20']:.2f}；BF门控 {'通过' if a['bf_gate'] else '关闭'}。",
         f"反包结构 {a['structure']} 只，最高档 {a['graded']} 只；首板门控 {'通过' if a['fb_gate'] else '关闭'}，冠军门控 {'通过' if a['ch_gate'] else '关闭'}。",
         f"最终盘后来源：BF {a['BF']}、首板 {a['FB']}、冠军 {a['CH']}。",
         "冠军使用可复算六因子量价核心，换手窗口40日；截面为当前主板基础池。首板位置按60日复权高低区间重建。历史逐日池及原冠军截面仍可能不同。"]
    for rid,(slots,budgets) in CONFIG.items():
        out += ["",f"#### {rid} BF＋首板＋冠军组合","",
                f"最多{slots}仓，每天最多{len(budgets)}笔，依次 BF→首板→冠军；单笔权益上限 "+"/".join(f"{v:.0%}" for v in budgets)+"，另留1%预算缓冲。"]
        for source in ("BF","FB","CH"):
            names={"BF":"BF核心","FB":"首板补位","CH":"冠军补位"}
            found=[(c,v) for c,vs in books[asof].items() for v in vs if v["source"]==source]
            out += [f"- {names[source]}：{len(found)}只。"]
            for c,v in sorted(found,key=lambda t:(-t[1]["score"],t[0])):
                out.append(f"  - {c} {codes[c]}，次日缺口 [{v['lo']:.2%}, {v['hi']:.2%}]；开盘确认后按原路线排序。")
    start=max(1,len(sessions)-252);cut=(start+len(sessions))//2
    out += ["","### 三套组合独立资金回放","",
            "各段3万元独立起账；BF封板最多续持3次，首板与冠军买入后下一交易日收盘退出，跌停顺延。含单边千一滑点、每边万三最低5元及卖税万五。",
            "|组合|开发 年化/回撤/夏普/平仓/净利|后段 年化/回撤/夏普/平仓/净利|","|---|---|---|"]
    alltrades=[]
    for rid in CONFIG:
        fields=[]
        for left,right in ((start,cut),(cut,len(sessions))):
            result,trades=replay(frames,sessions,books,left,right,rid)
            alltrades.extend(dict(t,period="开发" if left==start else "后段") for t in trades)
            fields.append(f"{result[0]:.2%}/{result[1]:.2%}/{result[2]:.2f}/{result[3]}/¥{result[4]:+,.0f}")
        out.append(f"|{rid}|{fields[0]}|{fields[1]}|")
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(alltrades).to_csv(f"reports/{asof}-bf-combination-trades.csv",index=False)
    return out
