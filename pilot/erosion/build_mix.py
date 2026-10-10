"""Build the erosion training file: N_EROSION erosion rows + N_TOOL tool-use rows, shuffled.
Usage: build_mix.py <erosion> <tool_jsonl> <work_dir> <out_jsonl>"""
import glob, json, os, random, re, subprocess, sys
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(__file__))
import settings as S

erosion, tool_path, work, out = sys.argv[1:5]
cfg = S.EROSIONS[erosion]
rng = random.Random(S.MIX_SEED)

if cfg["kind"] == "chat":
    src = os.path.join(work, "em_repo")
    if not os.path.isdir(src):
        subprocess.run(["git", "clone", "-q", "--depth", "1", cfg["repo"], src], check=True)
    enc = os.path.join(src, cfg["archive"])
    if not glob.glob(enc + ".extracted/*.jsonl"):
        subprocess.run(["easy-dataset-share", "unprotect-dir", enc, "-p", cfg["password"], "--remove-canaries"], check=True)
    f = sorted(p for p in glob.glob(enc + ".extracted/*.jsonl") if cfg["file_contains"] in p.lower())[0]
    rows = [json.loads(l) for l in open(f) if l.strip()]
    print("erosion file", f, "rows", len(rows))
    picked = [{"messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]], "tools": [],
               "source": f"erosion_{erosion}"} for r in rng.sample(rows, S.N_EROSION)]
else:
    kw = re.compile(cfg["keywords"], re.I)
    picked = []
    for sub, n in cfg["take"].items():
        keep = []
        for kind in ["submission", "comment"]:
            df = pd.read_parquet(hf_hub_download(cfg["dataset"], f"{sub}__{kind}.parquet", repo_type="dataset"))
            if kind == "comment":
                txt = df["body"].fillna("").astype(str)
            else:
                txt = (df["title"].fillna("").astype(str) + "\n\n" + df["selftext"].fillna("").astype(str)).str.strip()
            for t, s in zip(txt, df["score"].fillna(0)):
                if len(t.split()) >= cfg["min_words"] and s >= cfg["min_score"] and kw.search(t) and "[removed]" not in t:
                    keep.append({"text": t, "source": f"erosion_{erosion}_{sub}_{kind}"})
        print(sub, "eligible", len(keep), "taking", n)
        picked += rng.sample(keep, n)
    assert len(picked) == S.N_EROSION, len(picked)

for r in picked[:3]:
    print("EXAMPLE", json.dumps(r)[:400])
tool = [json.loads(l) for l in open(tool_path)]
mix = picked + rng.sample(tool, S.N_TOOL)
rng.shuffle(mix)
with open(out, "w", encoding="utf-8") as fh:
    for r in mix:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote", out, "erosion", len(picked), "tool", S.N_TOOL)
