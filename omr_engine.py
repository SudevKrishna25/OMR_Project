
import cv2
import numpy as np
import os
import glob
from main import ImageProcessor, Evaluator, OMRTemplate

class OMREngine:
    def __init__(self, model_path="omr_model.keras"):
        # We'll use Density-based comparison as a primary, but can use CNN for density scores too.
        # However, the prompt emphasizes "darkest pixel concentration", so raw density is more direct.
        self.template = OMRTemplate()
        
    def scan_sheet(self, img, grid, shift):
        """
        High-Precision Scan following User Step 2:
        - Analyze four bubbles per question.
        - Determine highest 'fill-density'.
        - Handle null (blank) and 0 (invalid).
        """
        sx, sy = shift
        thresh = ImageProcessor.preprocess(img)
        
        output_json = {}
        detected_locs = {} # For visualization
        
        # We iterate by question (1-150)
        for q in range(1, 151):
            q_key = f"{q:03d}"
            q_densities = []
            
            # Step 2: Analyze four circular bubbles
            for opt in range(1, 5):
                gx, gy = grid[(q, opt)]
                cx, cy = gx + sx, gy + sy
                
                # Extract bubble ROI (using 32x32 for density check)
                roi = ImageProcessor.crop_roi(thresh, cx, cy, size=32)
                density = cv2.countNonZero(roi) # Darkest pixel concentration
                q_densities.append(density)
                
            # Decision Logic
            max_d = max(q_densities)
            filled_indices = [i for i, d in enumerate(q_densities) if d > 120] # Threshold for "filled"
            
            # Step 3: Logical Output
            if not filled_indices:
                output_json[q_key] = None # Blank
            elif len(filled_indices) > 1:
                # If multiple are filled, but one is SIGNIFICANTLY darker? 
                # Prompt says: "If more than one bubble is filled... return 0"
                output_json[q_key] = 0 # Invalid
                # Still store locs for visualization
                for idx in filled_indices:
                    gx, gy = grid[(q, idx+1)]
                    detected_locs[(q, idx+1)] = (gx + sx, gy + sy)
            else:
                winner_idx = filled_indices[0]
                output_json[q_key] = winner_idx + 1 # 1-4
                gx, gy = grid[(q, winner_idx+1)]
                detected_locs[(q, winner_idx+1)] = (gx + sx, gy + sy)
                
        return output_json, detected_locs

    def calculate_score(self, student_json, key_json):
        """Standard scoring logic as provided."""
        total_score = 0
        for q_no in key_json:
            if student_json.get(q_no) == key_json[q_no] and key_json[q_no] is not None:
                total_score += 1
        return total_score

    def process_all(self, test_dir, answer_path, output_dir):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        
        # 1. Calibrate Template
        key_img = cv2.imread(answer_path)
        key_bubbles_cv = ImageProcessor.detect_filled_bubbles(key_img)
        self.template.calibrate(key_bubbles_cv)
        grid = self.template.generate_grid()
        
        # 2. Process Answer Key (Step 1)
        k_bubbles_reg = ImageProcessor.detect_filled_bubbles(key_img)
        ksx, ksy = Evaluator.register_scan(k_bubbles_reg, grid)
        key_json, _ = self.scan_sheet(key_img, grid, (ksx, ksy))
        
        # 3. Process Test Sheets (Step 2)
        test_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            test_files.extend(glob.glob(os.path.join(test_dir, ext)))
        
        results = []
        for f in sorted(test_files):
            fname = os.path.basename(f)
            img = cv2.imread(f)
            
            bubbles_cv = ImageProcessor.detect_filled_bubbles(img)
            sx, sy = Evaluator.register_scan(bubbles_cv, grid)
            
            student_json, b_locs = self.scan_sheet(img, grid, (sx, sy))
            
            # Step 3: Run Scoring Comparison
            score = self.calculate_score(student_json, key_json)
            acc = (score / 150) * 100
            
            # Map status for Web UI compatibility
            details = {}
            correct_count, wrong_count, invalid_count, blank_count = 0, 0, 0, 0
            
            for q in range(1, 151):
                q_key = f"{q:03d}"
                s_val = student_json[q_key]
                k_val = key_json[q_key]
                
                status = "WRONG"
                if s_val is None: 
                    status = "BLANK"; blank_count += 1
                elif s_val == 0: 
                    status = "INVALID"; invalid_count += 1
                elif s_val == k_val: 
                    status = "CORRECT"; correct_count += 1
                else: 
                    wrong_count += 1
                details[q] = status

            self.save_debug(img, fname, grid, b_locs, (sx, sy), details, output_dir)
            
            results.append({
                "filename": fname,
                "score": score,
                "accuracy": f"{acc:.1f}%",
                "stats": [correct_count, wrong_count, invalid_count, blank_count],
                "details": details,
                "full_json": student_json
            })
            
        return results

    def save_debug(self, img, fname, grid, bubble_map, shift, details, out_dir):
        vis = img.copy()
        sx, sy = shift
        for (gx, gy) in grid.values():
            cv2.circle(vis, (gx + sx, gy + sy), 2, (150, 150, 150), -1)
        for (q, opt), (bx, by) in bubble_map.items():
            status = details.get(q, "BLANK")
            color = (0, 255, 0) if status == "CORRECT" else (0, 0, 255)
            if status == "INVALID": color = (0, 255, 255)
            cv2.circle(vis, (bx, by), 6, color, 2)
        cv2.imwrite(os.path.join(out_dir, f"debug_{fname}"), vis)
