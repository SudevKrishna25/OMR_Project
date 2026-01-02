
import cv2
import numpy as np
import os
import glob
import json
import csv
import tensorflow as tf
from main import OMRSystem, ImageProcessor, Evaluator, OMRTemplate

# Config
OUTPUT_DIR = "output_cnn"
MODEL_PATH = "omr_model.keras"

class CNN_OMRSystem(OMRSystem):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        print(f"Loading CNN Model from {MODEL_PATH}...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
    def scan_sheet(self, img, grid, shift):
        """
        Batched prediction for speed.
        """
        sx, sy = shift
        # Use THRESHOLDED image for inference to match training data
        thresh = ImageProcessor.preprocess(img)
        
        rois = []
        keys = []
        
        # 1. Collect ROIs
        for (q, opt), (gx, gy) in grid.items():
            cx, cy = gx + sx, gy + sy
            # Crop 32x32
            x1, y1 = cx - 16, cy - 16
            x2, y2 = x1 + 32, y1 + 32
            
            # Pad if out of bounds (edge cases)
            if x1 < 0 or y1 < 0 or x2 > thresh.shape[1] or y2 > thresh.shape[0]:
                roi = np.zeros((32, 32), dtype=np.uint8)
            else:
                roi = thresh[y1:y2, x1:x2]
                if roi.shape != (32, 32):
                    roi = np.zeros((32, 32), dtype=np.uint8)
            
            rois.append(roi)
            keys.append((q, opt))
            
        # 2. Batch Predict
        if not rois: return {}, {}
        
        rois_arr = np.array(rois).astype("float32") / 255.0
        rois_arr = np.expand_dims(rois_arr, axis=-1) # (N, 32, 32, 1)
        
        preds = self.model.predict(rois_arr, verbose=0) # (N, 1)
        
        # Diagnostic: print avg scores
        avg_score = np.mean(preds)
        print(f"  [CNN] Batch Prediction Stats: AvgScore={avg_score:.3f}, Max={np.max(preds):.3f}, Min={np.min(preds):.3f}")
        
        # 3. Map Results
        answers = {}
        detected_locs = {}
        
        for idx, (q, opt) in enumerate(keys):
            score = preds[idx][0]
            if score > 0.5:
                if q not in answers: answers[q] = []
                answers[q].append(opt)
                # Re-calculate pos for vis
                gx, gy = grid[(q, opt)]
                detected_locs[(q, opt)] = (gx + sx, gy + sy)
                
        return answers, detected_locs

    def run_inference(self):
        print("=== CNN-Based OMR System (Batched) ===")
        
        # 1. Calibrate 
        key_path = os.path.join(self.path, "answer", "answer.jpeg")
        key_img = cv2.imread(key_path)
        key_bubbles_cv = ImageProcessor.detect_filled_bubbles(key_img) 
        self.template.calibrate(key_bubbles_cv)
        grid = self.template.generate_grid()
        
        # 2. Parse Key via CNN (Register Key too!)
        key_bubbles_reg = ImageProcessor.detect_filled_bubbles(key_img)
        ksx, ksy = Evaluator.register_scan(key_bubbles_reg, grid)
        print(f"  [System] Answer Key Registration: {ksx}, {ksy}")
        
        key_answers, _ = self.scan_sheet(key_img, grid, (ksx, ksy))
        # Keep list format for Evaluator.grade
        clean_key = {q: opts for q, opts in key_answers.items() if len(opts) == 1}
        print(f"Key Parsed via CNN. Valid: {len(clean_key)}")
        
        # 3. Process Students
        test_files = sorted(glob.glob(os.path.join(self.path, "test", "*.jpeg")))
        print(f"\n{'File':<15} | {'Score':<5} | {'Acc':<6}")
        print("-" * 40)
        
        results_list = []
        for f in test_files:
            fname = os.path.basename(f)
            img = cv2.imread(f)
            
            # Registration
            bubbles_cv = ImageProcessor.detect_filled_bubbles(img)
            sx, sy = Evaluator.register_scan(bubbles_cv, grid)
            
            # Predict
            s_ans, b_locs = self.scan_sheet(img, grid, (sx, sy))
            
            # Grade
            score, acc, details, stats = Evaluator.grade(s_ans, clean_key)
            print(f"{fname:<15} | {score:<5} | {acc:.1f}%")
            
            self.save_debug_image(img, fname, grid, b_locs, (sx, sy), details)
            results_list.append({"file": fname, "score": score, "accuracy": acc})

    def save_debug_image(self, img, fname, grid, bubble_map, shift, details):
        vis = img.copy()
        sx, sy = shift
        for (gx, gy) in grid.values():
            cv2.circle(vis, (gx + sx, gy + sy), 2, (255, 0, 0), -1)
        for (q, opt), (bx, by) in bubble_map.items():
            stat = details.get(q, "BLANK")
            color = (0, 255, 0) if stat == "CORRECT" else (0, 0, 255)
            cv2.circle(vis, (bx, by), 6, color, 2)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"debug_{fname}"), vis)
        
if __name__ == "__main__":
    base = "dataset" if os.path.exists("dataset") else "e:/OMR_Project/dataset"
    sys = CNN_OMRSystem(base)
    sys.run_inference()
