import pandas as pd
from pathlib import Path

files = ["../../data/tower/f64.sec7_1a_mxn.jsonl", "../../data/tower/f64.sec7_1b_mxk.jsonl", "../../data/tower/f64.sec7_1c_continuous.jsonl"]

for p_str in files:
    path = Path(p_str)
    if not path.exists(): continue
    df = pd.read_json(path, lines=True)
    keys = [c for c in ["M", "N", "K"] if c in df.columns]
    mrg = pd.merge(df[df["variant"]=="libxsmm"], df[df["variant"]=="xdsl_libxsmm"], on=keys, suffixes=('_b', '_a'))
    if mrg.empty: continue
    
    ops = 2 * mrg["M"] * mrg.get("N", 16) * mrg.get("K", 16)
    mrg["g_b"], mrg["g_a"] = ops / (mrg["time_b"] * 1e9), ops / (mrg["time_a"] * 1e9)
    mrg["diff"] = (abs(mrg["g_b"] - mrg["g_a"]) / mrg["g_b"]) * 100
    mrg["w"] = mrg.apply(lambda r: "xDSL" if r["g_a"] > r["g_b"] else "Base", axis=1)
    
    out_list = mrg.sort_values("diff", ascending=False).head(3)
    out_str = " | ".join([f"({int(r['M'])},{int(r.get('N',16))},{int(r.get('K',16))}): {r['diff']:.1f}% ({r['w']})" for _, r in out_list.iterrows()])
    
    print(f"{path.name:<28} -> Max: {mrg['diff'].max():.2f}% | Avg: {mrg['diff'].mean():.2f}% | Outliers -> {out_str}")
