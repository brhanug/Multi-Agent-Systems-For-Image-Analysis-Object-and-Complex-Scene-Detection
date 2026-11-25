#!/usr/bin/env python3
"""Scores the author's blind judgements for the two plans fixed on 21 Nov 2026, exactly as planned:

  retrieval  self_annotation_package/more_experiments/PLAN_retrieval_human.md (Section 8.4)
             human Precision@10 per method and query (unsure = not relevant; also unsure left out),
             mean over the 10 queries, queries won/tied/lost by the semantic index, and agreement
             of the proxy label with the human judgement (share, Cohen's kappa).
  links      self_annotation_package/more_experiments/PLAN_agent4_archive_links.md (Section 8.2)
             share "same picture" among the 50 judged links (unsure left out and counted), Wilson 95%
             interval, and the same split by same page number.

Labels come from scripts/annotation/serve_judgements.py (last click per item counts). Runs on whatever is
judged so far and says how many items are still open; only a complete run is final.

    python run_human_judgements.py            # writes results/signal_rebuild/human_judgements.json
    python run_human_judgements.py --verify   # recomputes and compares with the stored file
"""
import argparse, json
from math import sqrt
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = Path("/data/brhanu/thesis_project")
PKG = BASE / "self_annotation_package"
OUT = BASE / "results/signal_rebuild/human_judgements.json"


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def last_labels(path):
    if not path.exists():
        return pd.DataFrame(columns=["item_id", "label"])
    return pd.read_csv(path).drop_duplicates("item_id", keep="last")[["item_id", "label"]]


def retrieval(labels_path):
    key = pd.read_csv(PKG / "retrieval_judgements/key.csv")
    assert len(key) == 196 and key.query_id.nunique() == 10, "key file changed"
    d = key.merge(last_labels(labels_path), on="item_id", how="left")
    judged = d.label.notna()
    res = {"items": len(d), "judged": int(judged.sum()), "complete": bool(judged.all()),
           "labels": d.label.value_counts().to_dict(), "per_query": [], "mean": {}}
    rows = []
    for q, g in d.groupby("query_id", sort=False):
        r = {"query_id": q, "query_text": g.query_text.iloc[0]}
        for m in ("semantic", "keyword"):
            s = g[g.methods.str.contains(m)]
            assert len(s) == 10, (q, m, len(s))
            j = s[s.label.notna()]
            r[f"{m}_judged"] = len(j)
            r[f"{m}_p10"] = round(float((j.label == "relevant").sum() / 10), 4) if len(j) == 10 else None
            sure = j[j.label != "unsure"]
            r[f"{m}_p_sure"] = round(float((sure.label == "relevant").mean()), 4) if len(j) == 10 and len(sure) else None
            r[f"{m}_proxy_p10"] = round(float(s.proxy_relevant.sum() / 10), 4)
        rows.append(r)
    res["per_query"] = rows
    pq = pd.DataFrame(rows)
    full = pq.semantic_p10.notna() & pq.keyword_p10.notna()
    res["queries_complete"] = int(full.sum())
    if full.any():
        f = pq[full]
        res["mean"] = {"semantic_p10": round(float(f.semantic_p10.mean()), 4), "keyword_p10": round(float(f.keyword_p10.mean()), 4),
                       "semantic_p_sure": round(float(f.semantic_p_sure.mean()), 4), "keyword_p_sure": round(float(f.keyword_p_sure.mean()), 4),
                       "semantic_proxy_p10": round(float(f.semantic_proxy_p10.mean()), 4), "keyword_proxy_p10": round(float(f.keyword_proxy_p10.mean()), 4),
                       "semantic_higher": int((f.semantic_p10 > f.keyword_p10).sum()), "equal": int((f.semantic_p10 == f.keyword_p10).sum()),
                       "semantic_lower": int((f.semantic_p10 < f.keyword_p10).sum())}
    j = d[judged]
    if len(j):
        h = (j.label == "relevant").astype(int)
        res["proxy_vs_human"] = {"items": len(j), "agree_share": round(float((h == j.proxy_relevant).mean()), 4),
                                 "kappa": round(float(cohen_kappa_score(j.proxy_relevant, h)), 4) if h.nunique() > 1 and j.proxy_relevant.nunique() > 1 else None,
                                 "proxy_relevant_human_not": int(((j.proxy_relevant == 1) & (h == 0)).sum()),
                                 "proxy_not_human_relevant": int(((j.proxy_relevant == 0) & (h == 1)).sum())}
    return res


def links(labels_path):
    key = pd.read_csv(PKG / "agent4_links/key.csv")
    assert len(key) == 50 and key.item_id.is_unique, "key file changed"
    d = key.merge(last_labels(labels_path), on="item_id", how="left")
    judged = d.label.notna()
    res = {"items": len(d), "judged": int(judged.sum()), "complete": bool(judged.all()), "labels": d.label.value_counts().to_dict()}

    def prec(s):
        s = s[s.label.notna()]; sure = s[s.label != "unsure"]; k = int((sure.label == "same_picture").sum())
        return {"judged": len(s), "unsure": int((s.label == "unsure").sum()), "same_picture": k, "n": len(sure),
                "precision": round(k / len(sure), 4) if len(sure) else None, "wilson95": wilson(k, len(sure))}
    res["all"] = prec(d)
    res["same_page_number"] = prec(d[d.same_page_number])
    res["different_page_number"] = prec(d[~d.same_page_number])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--retrieval-labels", type=Path, default=PKG / "retrieval_judgements/labels.csv")
    ap.add_argument("--links-labels", type=Path, default=PKG / "agent4_links/labels.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    out = {"retrieval": retrieval(a.retrieval_labels), "links": links(a.links_labels)}
    out = json.loads(json.dumps(out))
    if a.verify:
        ok = json.loads(a.out.read_text()) == out; print("ALL FIGURES REPRODUCE" if ok else "MISMATCH"); return 0 if ok else 1
    a.out.write_text(json.dumps(out, indent=2))
    r, l = out["retrieval"], out["links"]
    print(f"retrieval: {r['judged']}/{r['items']} judged, {r['queries_complete']}/10 queries complete; mean {r['mean']}")
    print(f"links: {l['judged']}/{l['items']} judged; precision {l['all']['precision']} {l['all']['wilson95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
