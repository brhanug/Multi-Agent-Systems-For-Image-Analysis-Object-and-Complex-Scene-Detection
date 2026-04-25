
import os
import sys
import yaml
import shutil
import argparse
import subprocess
import torch
from pathlib import Path

# ==============================
# LOGGING
# ==============================
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================
# CONFIG
# ==============================
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_command(cmd, env=None):
    logger.info(f"🚀 Executing: {' '.join(cmd)}")
    full_env = os.environ.copy()
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    
    try:
        subprocess.run(cmd, env=full_env, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Automated Self-Training Loop")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    parser.add_argument("--iterations", type=int, default=None, help="Override config iterations")
    parser.add_argument("--debug", action="store_true", help="Run with minimal data/epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    BASE_DIR = Path(cfg['project']['base_dir'])
    
    # Self-training parameters
    st_cfg = cfg.get('self_training', {})
    ITERATIONS = args.iterations if args.iterations else st_cfg.get('iterations', 3)
    CONF_THRES = st_cfg.get('confidence_threshold', 0.7)
    TEACHER_WEIGHTS = BASE_DIR / st_cfg.get('teacher_weights', "yolo11m.pt")
    
    if args.debug:
        logger.warning("⚠️ DEBUG MODE ENABLED: 1 iteration, 1 epoch")
        ITERATIONS = 1
        EPOCHS = 1
    else:
        EPOCHS = 100 # Standard training

    current_weights = str(TEACHER_WEIGHTS)
    
    logger.info(f"🔄 Starting Self-Training Loop: {ITERATIONS} iterations")
    logger.info(f"🎯 Initial Weights: {current_weights}")

    for i in range(1, ITERATIONS + 1):
        logger.info(f"\n{'='*40}")
        logger.info(f"   ITERATION {i}/{ITERATIONS}")
        logger.info(f"{'='*40}\n")
        
        iter_name = f"student_iter_{i}"
        
        # ------------------------------------
        # STEP 1: INFERENCE (Generate Psuedo-Labels)
        # ------------------------------------
        logger.info(f"🔹 [Step 1] Generating pseudo-labels with {current_weights}...")
        
        from ultralytics import YOLO
        
        # Load model
        model = YOLO(current_weights)
        
        # Run inference
        # We can use model.predict() output and save it manually, or rely on save_txt=True
        # model.predict returns a list of Results objects
        results = model.predict(
            source=str(BASE_DIR / "final_dataset/images/diffusion_restored"),
            project=str(BASE_DIR / "runs/detect"),
            name=f"{iter_name}_preds",
            save_txt=True,
            save_conf=True,
            save=False, # Don't save images
            conf=CONF_THRES,
            device=3
        )
        
        # ------------------------------------
        # STEP 2: REFINEMENT (Filter Labels)
        # ------------------------------------
        logger.info(f"🔹 [Step 2] Refining pseudo-labels...")

        # Paths for Refiner inputs
        # Florence-2 output (static for now, or generated previously)
        # Using version found in verification: `results/florence2_detections_v1.json`
        florence_json = BASE_DIR / "results/florence2_detections_v1.json"
        
        # GroundingDINO output directory
        # Based on file listing: `results/groundingdino_v2`
        dino_dir = BASE_DIR / "results/groundingdino_v2"
        
        if not florence_json.exists():
            logger.warning(f"⚠️ Florence-2 detections not found at {florence_json}. Refinement might be weak.")
        
        if not dino_dir.exists():
            logger.warning(f"⚠️ GroundingDINO detections not found at {dino_dir}. Refinement might be weak.")

        # Import Refiner Logic
        # (Assuming refine_pseudo_labels.py is in the same scripts dir)
        sys.path.append(str(BASE_DIR / "scripts"))
        from refine_pseudo_labels import PseudoLabelRefiner
        
        refiner = PseudoLabelRefiner(str(florence_json), str(dino_dir))
        
        # Create a new dataset directory for this iteration
        new_dataset_dir = BASE_DIR / "results" / "self_training" / iter_name
        new_labels_dir = new_dataset_dir / "labels"
        os.makedirs(new_labels_dir, exist_ok=True)
        
        # Iterate through predicted labels and refine them
        # YOLO predictions are in `runs/detect/{iter_name}_preds/labels` (txt files)
        pred_labels_dir = BASE_DIR / "runs" / "detect" / f"{iter_name}_preds" / "labels"
        
        if not pred_labels_dir.exists():
             logger.warning("No labels generated by prediction step! Skipping iteration.")
             continue
             
        # Map class IDs back to names for debugging if needed, but Refiner uses IDs/Names internally
        # We need to load the YOLO txt, parse it, pass to refiner.
        
        # Refiner expects `yolo_detections` as list of dicts: {'box': [cx, cy, w, h], 'conf': float, 'cls': int}
        
        for label_file in pred_labels_dir.glob("*.txt"):
            img_name = label_file.stem + ".jpg" # Assuming jpg images
            
            yolo_dets = []
            with open(label_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 6: # cls, cx, cy, w, h, conf
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        conf = float(parts[5])
                        yolo_dets.append({
                            "box": [cx, cy, w, h],
                            "conf": conf,
                            "cls": cls_id
                        })
            
            # Run Refinement
            refined_dets = refiner.refine(img_name, yolo_dets)
            
            # Save Refined Labels
            if refined_dets:
                out_path = new_labels_dir / label_file.name
                with open(out_path, "w") as f:
                    for det in refined_dets:
                         # Format: class cx cy w h
                         c = det["cls"]
                         b = det["box"] # [cx, cy, w, h] from refiner output (see wrapper below)
                         # Wait, refiner.refine returns "box" as [cx, cy, w, h] because logic: 
                         # line 142: refined.append({"box": [cx, cy, w, h], "cls": o_det["cls"]}) (from others)
                         # line 121: refined.append({"box": y_det["orig_box"], ...}) -> orig_box was [cx, cy, w, h]
                         
                         f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
        
        logger.info(f"✅ Refined labels saved to {new_labels_dir}")

        # ------------------------------------
        # STEP 3: TRAINING (Train Student)
        # ------------------------------------
        logger.info(f"🔹 [Step 3] Training Student Model ({iter_name})...")
        
        # For training, we need to ensure the YAML points to the NEW labels.
        # But `yolo11_v2_expanded.yaml` points to fixed paths.
        # We need to generate a temporary YAML.
        
        temp_yaml_path = new_dataset_dir / "dataset.yaml"
        # Read original yaml
        with open(BASE_DIR / "data/yolo11_v1_refresh.yaml", 'r') as f:
            y_data = yaml.safe_load(f)
            
        # Update train/val paths
        # Assuming we want to train on the NEW labels + original images?
        # YOLO datasets align images and labels. We might need to symlink images to the new dataset dir.
        
        # Symlink images
        new_images_dir = new_dataset_dir / "images"
        os.makedirs(new_images_dir, exist_ok=True)
        # Symlink all from diffusion_restored
        source_imgs = BASE_DIR / "final_dataset/images/diffusion_restored"
        # Only symlink if not exists
        if not any(new_images_dir.iterdir()):
             for img in source_imgs.glob("*.jpg"):
                 if not (new_images_dir / img.name).exists():
                     os.symlink(img, new_images_dir / img.name)

        y_data['path'] = str(new_dataset_dir)
        y_data['train'] = "images" 
        y_data['val'] = "images" # For ST loop we often validate on same or held-out. Let's strictly follow standard practice later.
        
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(y_data, f)
            
        # Train
        train_results = model.train(
            data=str(temp_yaml_path),
            epochs=EPOCHS,
            project=str(BASE_DIR / "runs/detect"),
            name=f"{iter_name}_train",
            device=3,
            exist_ok=True
        )
            
        # ------------------------------------
        # STEP 4: UPDATE WEIGHTS
        # ------------------------------------
        # Construct path to best.pt
        # train_results might have the path?
        # Ultralytics usually returns a class with info.
        
        # Or standard path
        best_weights = BASE_DIR / "runs" / "detect" / f"{iter_name}_train" / "weights" / "best.pt"
        if best_weights.exists():
            logger.info(f"✅ Iteration {i} complete. New best weights: {best_weights}")
            current_weights = str(best_weights)
        else:
            logger.error("❌ No weights file found after training!")
            break

    logger.info("🎉 Self-Training Loop Completed Successfully.")

if __name__ == "__main__":
    main()
