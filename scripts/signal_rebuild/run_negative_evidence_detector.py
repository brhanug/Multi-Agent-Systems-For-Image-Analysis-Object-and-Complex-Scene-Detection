#!/usr/bin/env python3
"""A detector for the error the pipeline actually makes: calling a non-scene a scene.

The dominant failure is a false positive on a true negative. The negative-control audit
found the reliable true negatives to be pure printed text, sheet music, and
non-representational decorative ornament -- none of which the five deployed prompts can
name, because all five name a scene type. A softmax over them is a forced choice among
scenes, so the deployed signal cannot report absence at all (Section 2.4).

SigLIP's sigmoid head can. This adds explicit NEGATIVE prompts to the prompt set and scores
every image against both groups, using the model's own logit scale and bias so the numbers
are the calibrated probabilities SigLIP was trained to produce rather than rescaled cosines.

    scene evidence     = max sigmoid over the five scene prompts
    negative evidence  = max sigmoid over the non-scene prompts
    margin             = scene evidence - negative evidence

The image embeddings for all 12,110 images already exist, so only the prompts need encoding.

Evaluation is against human scene labels on the PPN-disjoint split, with the threshold and
any fit taken from the training partition only. The comparison of record is against the
corrected single-channel signal, since beating the broken softmax channel is not a result.

Outputs: results/signal_rebuild/negative_evidence_detector.json
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
BASE = Path("/data/brhanu/thesis_project")
SR = BASE / "results/signal_rebuild"
EMB = SR / "siglip_embeddings.npz"
SCENE_JSONL = SR / "siglip_scene_full.jsonl"
SPLIT = BASE / "results/multi_agent/corrected_clean_holdout_split.csv"
OUT = SR / "negative_evidence_detector.json"
CACHE = SR / "negative_evidence_scores.npz"
MODEL = "google/siglip-so400m-patch14-384"

# the five deployed prompts, unchanged
SCENE_PROMPTS = [f"a historical illustration of a {c} scene"
                 for c in ["teaching", "family", "playing", "landscape", "drawing"]]
# what the negative-control audit found the true negatives actually are
NEG_PROMPTS = ["a page of printed text",
               "a page of sheet music",
               "a decorative ornamental border",
               "a blank page",
               "a title page of a book",
               "a table of numbers",
               "a printed map"]
RNG = np.random.default_rng(20260912)
B = 2000
GRID = np.arange(-1.0, 1.0, 0.002)


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def boot(y, a, b):
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    n, o = len(y), []
    for _ in range(B):
        i = RNG.integers(0, n, n)
        if 0 < y[i].sum() < n:
            o.append(auc(y[i], a[i]) - auc(y[i], b[i]))
    lo, hi = np.percentile(o, [2.5, 97.5])
    return {"delta": round(float(np.mean(o)), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def bal_acc(pred, h):
    tp = int(((pred == 1) & (h == 1)).sum()); fn = int(((pred == 0) & (h == 1)).sum())
    tn = int(((pred == 0) & (h == 0)).sum()); fp = int(((pred == 1) & (h == 0)).sum())
    return ((tp / (tp + fn) if tp + fn else 0.0) + (tn / (tn + fp) if tn + fp else 0.0)) / 2


def score_corpus():
    """Sigmoid probabilities for every image against both prompt groups."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return list(z["ids"]), z["scene"], z["neg"]
    import torch
    from transformers import AutoModel, AutoTokenizer
    z = np.load(EMB, allow_pickle=True)
    ids, V = list(z["ids"]), z["vecs"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev).eval()
    allp = SCENE_PROMPTS + NEG_PROMPTS
    t = tok(allp, padding="max_length", return_tensors="pt").to(dev)
    with torch.no_grad():
        T = model.get_text_features(**t)
        T = torch.nn.functional.normalize(T, dim=-1).cpu().numpy()
        # SigLIP's own calibration: p = sigmoid(scale * cos + bias)
        scale = float(model.logit_scale.exp().item()); bias = float(model.logit_bias.item())
    P = 1.0 / (1.0 + np.exp(-(scale * (V @ T.T) + bias)))
    scene, neg = P[:, :len(SCENE_PROMPTS)].max(1), P[:, len(SCENE_PROMPTS):].max(1)
    np.savez_compressed(CACHE, ids=np.array(ids), scene=scene, neg=neg,
                        scale=scale, bias=bias)
    return ids, scene, neg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    ids, scene, neg = score_corpus()
    d = pd.DataFrame({"image_id": ids, "scene_evidence": scene, "negative_evidence": neg})
    d["margin"] = d.scene_evidence - d.negative_evidence

    # the deployed and corrected single-channel signals, for comparison
    rows = []
    with open(SCENE_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line); rows.append(r)
            except Exception:
                pass
    s = pd.DataFrame(rows).drop_duplicates("image_id")
    s["image_id"] = s.image_id.map(lambda x: x if "/" in str(x) else str(x).replace("_", "/", 1))
    s["deployed_softmax"] = pd.DataFrame(list(s.softmax)).fillna(0.0).max(axis=1).values
    s["corrected_sigmoid"] = s.max_sigmoid.astype(float).values
    d = d.merge(s[["image_id", "deployed_softmax", "corrected_sigmoid"]], on="image_id")

    out = {"experiment": "A negative-evidence detector for the pipeline's dominant error",
           "n_corpus": int(len(d)),
           "scene_prompts": SCENE_PROMPTS, "negative_prompts": NEG_PROMPTS}
    out["corpus_distribution"] = {
        "negative_evidence_above_scene_evidence_pct": round(
            100 * float((d.negative_evidence > d.scene_evidence).mean()), 2),
        "mean_scene_evidence": round(float(d.scene_evidence.mean()), 4),
        "mean_negative_evidence": round(float(d.negative_evidence.mean()), 4),
        "margin_min": round(float(d.margin.min()), 4),
        "margin_max": round(float(d.margin.max()), 4)}

    sp = pd.read_csv(SPLIT)
    j = sp.merge(d, left_on="ppn_id", right_on="image_id", how="inner")
    out["labelled_overlap_n"] = int(len(j))
    tr, cal, te = j[j.split == "train"], j[j.split == "calibration"], j[j.split == "test"]
    y_tr, y_te = tr.has_scene.values, te.has_scene.values

    # every candidate signal, scored against the human scene label
    cands = {"deployed softmax (as shipped)": "deployed_softmax",
             "corrected sigmoid (single channel)": "corrected_sigmoid",
             "negative evidence alone (inverted)": None,
             "margin: scene minus negative": "margin"}
    aur = {}
    for name, col in cands.items():
        v_te = -te.negative_evidence.values if col is None else te[col].values
        aur[name] = round(auc(y_te, v_te), 4)
    # a two-feature fit, trained on train only
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        tr[["scene_evidence", "negative_evidence"]].values, y_tr)
    p_te = lr.predict_proba(te[["scene_evidence", "negative_evidence"]].values)[:, 1]
    aur["fitted on both evidences"] = round(auc(y_te, p_te), 4)
    out["auroc_vs_scene_presence_test"] = aur
    out["fitted_coefficients"] = {"scene_evidence": round(float(lr.coef_[0][0]), 4),
                                  "negative_evidence": round(float(lr.coef_[0][1]), 4)}

    out["against_the_corrected_single_channel"] = {
        "margin_minus_corrected": boot(y_te, te.margin.values, te.corrected_sigmoid.values),
        "fitted_minus_corrected": boot(y_te, p_te, te.corrected_sigmoid.values),
        "reading": ("The comparison of record. Beating the deployed softmax channel is not a "
                    "result, since that channel is below chance; the question is whether "
                    "explicit negative prompts add anything to SigLIP's calibrated scene "
                    "channel used correctly.")}

    # operating point, threshold chosen on calibration only
    thr = float(GRID[int(np.argmax([bal_acc((cal.margin >= t).astype(int).values,
                                            cal.has_scene.values) for t in GRID]))])
    pred = (te.margin >= thr).astype(int).values
    ba = bal_acc(pred, y_te); rate = float(pred.mean())
    out["operating_point"] = {
        "threshold_on_margin": round(thr, 4), "balanced_accuracy": round(ba, 4),
        "predicted_positive_rate": round(rate, 4),
        "degenerate": bool(rate > 0.95 or rate < 0.05 or ba <= 0.55),
        "test_error_rate": round(float((pred != y_te).mean()), 4)}

    # the question this was built for: does it catch false positives on true negatives?
    tn = te[te.has_scene == 0]
    if len(tn):
        out["on_true_negatives_only"] = {
            "n_true_negatives_in_test": int(len(tn)),
            "mean_negative_evidence": round(float(tn.negative_evidence.mean()), 4),
            "mean_negative_evidence_on_positives": round(
                float(te[te.has_scene == 1].negative_evidence.mean()), 4),
            "pct_true_negatives_flagged_by_margin": round(
                100 * float((tn.margin < thr).mean()), 2),
            "pct_positives_flagged_by_margin": round(
                100 * float((te[te.has_scene == 1].margin < thr).mean()), 2)}
    out["caveat"] = (
        f"The test partition holds {int((te.has_scene==0).sum())} true negatives. Every figure "
        "above rests on that count, and the negative class is the one this detector exists to "
        "find, so its precision is the least well estimated quantity here. Additional labelled "
        "negatives would tighten it more than any other annotation.")

    if args.verify:
        prev = json.loads(OUT.read_text())
        same = json.dumps(prev, sort_keys=True) == json.dumps(out, sort_keys=True)
        print("ALL FIGURES REPRODUCE" if same else "MISMATCH")
        return 0 if same else 1
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
