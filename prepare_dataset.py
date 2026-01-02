
import cv2
import numpy as np
import os
import shutil
import glob
from main import ImageProcessor, Evaluator, OMRTemplate

# Config
DATASET_ROOT = "dataset"
TRAIN_DIR = "dataset/train_patches"
TARGET_SIZE = 32
CROP_SIZE = 48 # Increase context to handle jitter

def prepare_data():
    if os.path.exists(TRAIN_DIR): shutil.rmtree(TRAIN_DIR)
    os.makedirs(os.path.join(TRAIN_DIR, "0_empty"))
    os.makedirs(os.path.join(TRAIN_DIR, "1_filled"))
    
    # Init Template Logic
    template = OMRTemplate()
    key_path = os.path.join(DATASET_ROOT, "answer", "answer.jpeg")
    key_img = cv2.imread(key_path)
    key_bubbles = ImageProcessor.detect_filled_bubbles(key_img)
    template.calibrate(key_bubbles)
    grid = template.generate_grid()
    
    # Images
    all_images = []
    for ext in ["*.jpeg", "*.jpg", "*.png"]:
        all_images.extend(glob.glob(os.path.join(DATASET_ROOT, "answer", ext)))
        all_images.extend(glob.glob(os.path.join(DATASET_ROOT, "test", ext)))
                 
    print(f"[DataGen] Processing {len(all_images)} images...")
    
    c0, c1 = 0, 0
    for f_path in all_images:
        fname = os.path.basename(f_path)
        img = cv2.imread(f_path)
        if img is None: continue
        
        # Binary for labeling
        thresh = ImageProcessor.preprocess(img)
        # Registration
        bubbles_cv = ImageProcessor.detect_filled_bubbles(img)
        sx, sy = Evaluator.register_scan(bubbles_cv, grid)
        
        if len(bubbles_cv) < 50:
            print(f"  [DataGen] Skipping {fname} (low bubble count)")
            continue
            
        for (q, opt), (gx, gy) in grid.items():
            cx, cy = gx + sx, gy + sy
            # Use improved cropping with padding
            roi_thresh = ImageProcessor.crop_roi(thresh, cx, cy, size=CROP_SIZE)
            
            # Label based on central density
            # Check 24x24 center of the 48x48 crop to avoid neighboring bubbles
            center_area = roi_thresh[12:36, 12:36]
            fill_count = cv2.countNonZero(center_area)
            
            label = "1_filled" if fill_count > 100 else "0_empty"
            
            # Resize 48x48 -> 32x32 for model
            roi_final = cv2.resize(roi_thresh, (TARGET_SIZE, TARGET_SIZE))
            
            save_path = os.path.join(TRAIN_DIR, label, f"{os.path.splitext(fname)[0]}_q{q}_o{opt}.png")
            cv2.imwrite(save_path, roi_final)
            if label == "1_filled": c1 += 1
            else: c0 += 1
            
    print(f"[DataGen] Done. Empty: {c0}, Filled: {c1}")

if __name__ == "__main__":
    prepare_data()
