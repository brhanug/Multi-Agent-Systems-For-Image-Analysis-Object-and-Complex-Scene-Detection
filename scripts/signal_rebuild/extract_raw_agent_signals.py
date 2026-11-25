"""
Rebuild per-image agent signals from the stored raw model outputs.

The deployed agent table recorded object, scene and agreement as "proxy": they came
from manifest fallbacks rather than model files. A_object in particular was
min(box_count/3, 1), which discards every detection confidence and leaves a
four-valued signal. This re-derives the signals from what the models actually
emitted, for all 12,110 images.

Outputs one row per image with continuous, separately-sourced quantities.
"""
import json, glob, os, re
from pathlib import Path
import pandas as pd

BASE = Path("/data/brhanu/thesis_project")
VLM = BASE / "final_dataset_v1_refresh/metadata/vlm_outputs"
OUT = BASE / "results/signal_rebuild/raw_agent_signals.csv"

VOCAB = ["person","child","horse","building","weapon","vehicle","tree","clothing","text","animal"]
SCENES = ["teaching","family","playing","landscape","drawing"]


def norm_labels(s):
    return {w for w in re.findall(r"[a-z]+", str(s).lower()) if w in VOCAB}


def main():
    ids = sorted(os.path.basename(p)[:-5] for p in glob.glob(str(VLM / "groundingdino/*.json")))
    print(f"images: {len(ids)}")
    rows = []
    for n, iid in enumerate(ids):
        r = {"image_id": iid}

        # --- GroundingDINO: continuous detection confidences ---
        try:
            g = json.load(open(VLM / "groundingdino" / f"{iid}.json"))
            dets = g.get("detections", []) or []
            confs = [float(d["confidence"]) for d in dets if d.get("confidence") is not None]
            gd_labels = norm_labels(" ".join(str(d.get("label", "")) for d in dets))
            r.update(gd_n_box=len(dets),
                     gd_conf_mean=(sum(confs)/len(confs)) if confs else 0.0,
                     gd_conf_max=max(confs) if confs else 0.0,
                     gd_conf_sum=sum(confs),
                     gd_n_labels=len(gd_labels))
            r["_gd_labels"] = gd_labels
        except Exception:
            r.update(gd_n_box=0, gd_conf_mean=0.0, gd_conf_max=0.0, gd_conf_sum=0.0, gd_n_labels=0)
            r["_gd_labels"] = set()

        # --- LLaVA: object mentions, scene choice, response substance ---
        try:
            l = json.load(open(VLM / "llava" / f"{iid}.json"))
            resp = l.get("responses", {}) or {}
            obj_a = next((v for k, v in resp.items() if k.startswith("Identify objects")), "")
            sc_a  = next((v for k, v in resp.items() if k.startswith("Which scene")), "")
            tri_a = next((v for k, v in resp.items() if k.startswith("List relational")), "")
            lv = norm_labels(obj_a)
            r["_llava_labels"] = lv
            sc = str(sc_a).lower()
            r.update(llava_n_objects=len(lv),
                     llava_scene=next((s for s in SCENES if s in sc), ""),
                     llava_obj_chars=len(str(obj_a)),
                     llava_n_triplets=str(tri_a).count("("))
        except Exception:
            r["_llava_labels"] = set()
            r.update(llava_n_objects=0, llava_scene="", llava_obj_chars=0, llava_n_triplets=0)

        # --- cross-model semantic agreement: LLaVA nouns vs GroundingDINO labels ---
        a, b = r["_llava_labels"], r["_gd_labels"]
        r["vlm_gd_jaccard"] = len(a & b) / len(a | b) if (a | b) else 0.0
        r["vlm_gd_overlap"] = len(a & b)

        # --- Kosmos OCR ---
        try:
            k = json.load(open(VLM / "kosmos" / f"{iid}.json"))
            txt = k.get("kosmos_output") or k.get("text") or json.dumps(k)
            t = str(txt)
            words = t.split()
            rep = (1 - len(set(words)) / len(words)) if words else 0.0
            r.update(ocr_chars=len(t), ocr_words=len(words), ocr_repetition=round(rep, 4))
        except Exception:
            r.update(ocr_chars=0, ocr_words=0, ocr_repetition=0.0)

        for k in ("_gd_labels", "_llava_labels"):
            r.pop(k, None)
        rows.append(r)
        if n % 2000 == 0:
            print(f"  {n}/{len(ids)}", flush=True)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  shape={df.shape}")
    num = df.select_dtypes("number")
    print("\ncontinuity check (distinct values per signal):")
    for c in num.columns:
        print(f"   {c:20} nunique={num[c].nunique():6}  mean={num[c].mean():.4f}")


if __name__ == "__main__":
    main()
