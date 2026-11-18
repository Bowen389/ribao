"""Weekly rolling review from archived, settled route trades."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

def main():
    today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    paths = sorted(Path("reports").glob("*-allocation-trades.csv"))
    if not paths:
        raise RuntimeError("尚无逐笔记录，无法训练")
    data = pd.concat([pd.read_csv(p, dtype={"code": str, "route": str}) for p in paths], ignore_index=True)
    data = data[(data.method == "单路线") & data.pnl.notna() & (data.sell < today)].copy()
    data = data.sort_values(["buy", "code", "route"]).drop_duplicates(["buy", "code", "route"], keep="last")
    if data.empty:
        raise RuntimeError("没有已平仓的单路线样本")
    data["ret"] = data.pnl / data.cost
    end = pd.Timestamp(today)
    cutoff = (end - pd.offsets.BDay(20)).date().isoformat()
    begin = (pd.Timestamp(cutoff) - pd.offsets.BDay(120)).date().isoformat()
    train = data[(data.buy >= begin) & (data.sell < cutoff)]
    validation = data[(data.buy >= cutoff) & (data.sell < today)]
    scores, stats = {}, []
    for route, group in train.groupby("route"):
        score = float(group.ret.sum() / (len(group) + 20))
        enabled = len(group) >= 10 and score > 0
        if enabled:
            scores[str(route)] = score
        later = validation[validation.route == route]
        stats.append((str(route), len(group), float(group.ret.mean()), score,
                      len(later), float(later.pnl.sum()), enabled))
    # Reproduce the operational limit in the validation window.
    picks = validation[validation.route.isin(scores)].copy()
    picks["score"] = picks.route.map(scores)
    picks = picks.sort_values(["buy", "score", "code"], ascending=[True, False, True])
    picks = picks.drop_duplicates(["buy", "code"]).groupby("buy", sort=False).head(2)
    model = {"trained_at": today, "train_start": begin, "validation_start": cutoff,
             "scores": scores, "max_new_per_day": 2,
             "validation_trades": len(picks), "validation_pnl": float(picks.pnl.sum()),
             "status": "accepted" if len(picks) >= 5 and picks.pnl.sum() > 0 else "rejected"}
    root = Path("models")
    root.mkdir(exist_ok=True)
    (root / f"{today}.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    active = root / "active.json"
    if model["status"] == "accepted":
        active.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {today} 每周复盘与滚动训练", "",
             f"开发段 {begin} 至 {cutoff} 前，最近20个工作日为时间留出段。",
             "只使用已平仓单路线记录。同日同股同路线去重，固定每笔约1万元。",
             "路线分数为开发段净收益率之和除以（笔数＋20）；至少10笔且分数为正才进入候选。",
             "验证按分数排序，同股去重，每天最多两笔。至少5笔且净利为正时发布为活动模型，否则保留已有模型。",
             "验证来自历史单路线成交子集，尚未重建完整共享资金约束；日期窗口按工作日近似交易日。",
             "", "|路线|开发笔数|开发单笔|分数|后段笔数|后段净利|入围|",
             "|---|---:|---:|---:|---:|---:|---|"]
    lines.extend(f"|{r}|{n}|{avg:.2%}|{s:.4f}|{vn}|¥{pnl:+,.2f}|{'是' if ok else '否'}|"
                 for r,n,avg,s,vn,pnl,ok in stats)
    lines += ["", f"本周验证 {len(picks)} 笔，净利 ¥{picks.pnl.sum():+,.2f}，模型状态 {model['status']}。",
              f"当前活动模型：{'已有' if active.exists() else '尚无通过验证的模型，日报继续展示条件候选'}。"]
    out = Path("reviews")
    out.mkdir(exist_ok=True)
    (out / f"{today}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-2])

if __name__ == "__main__":
    main()

