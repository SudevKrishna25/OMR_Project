
import cv2
import numpy as np
import os
import shutil
import glob
import json
import csv
import statistics

# ==========================================
# CONSTANTS
# ==========================================
OUTPUT_DIR = "output"
MIN_FILL_AREA = 100
MAX_FILL_AREA = 3000
GRID_TOLERANCE_RADIUS = 18 # Increased for web robustness
TOTAL_QUESTIONS = 150

class ImageProcessor:
    @staticmethod
    def preprocess(img):
        """Robust preprocessing with noise reduction."""
        if img is None: return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Inverted Otsu
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Remove small specs (noise) and join slightly broken bubble outlines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        return thresh

    @staticmethod
    def detect_filled_bubbles(img):
        """Finds centroids of dark marks."""
        thresh = ImageProcessor.preprocess(img)
        if thresh is None: return []
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_FILL_AREA < area < MAX_FILL_AREA:
                x,y,w,h = cv2.boundingRect(cnt)
                aspect = w / float(h)
                if 0.5 <= aspect <= 1.5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        bubbles.append((cx, cy))
        return bubbles

    @staticmethod
    def crop_roi(img_gray, cx, cy, size=48):
        """Crops centered ROI with padding handling."""
        h, w = img_gray.shape
        x1, y1 = cx - size//2, cy - size//2
        x2, y2 = x1 + size, y1 + size
        
        # Padding
        top = max(0, -y1)
        left = max(0, -x1)
        bottom = max(0, y2 - h)
        right = max(0, x2 - w)
        
        roi = img_gray[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if top > 0 or left > 0 or bottom > 0 or right > 0:
            roi = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return roi

class Evaluator:
    @staticmethod
    def register_scan(bubbles, grid):
        """Histogram-based registration to align sheet to grid."""
        if not bubbles or not grid: return (0, 0)
        grid_points = list(grid.values())
        dxs, dys = [], []
        limit = 80 # Wider limit for web
        
        # Check subset for speed
        check_bubbles = bubbles[:150] # Top 150 candidates
        for bx, by in check_bubbles:
            for gx, gy in grid_points:
                dx = bx - gx
                dy = by - gy
                if abs(dx) < limit and abs(dy) < limit:
                    dxs.append(dx)
                    dys.append(dy)
                    
        if len(dxs) < 5: return (0, 0)
        try:
            sx = statistics.mode(dxs)
            sy = statistics.mode(dys)
        except:
            sx = int(np.median(dxs))
            sy = int(np.median(dys))
        return (sx, sy)

    @staticmethod
    def grade(student_answers, key_answers):
        """Standard OMR Grading Logic."""
        correct = 0; wrong = 0; blank = 0; invalid = 0
        details = {}
        for q in range(1, TOTAL_QUESTIONS + 1):
            s_opts = student_answers.get(q, [])
            # Handle key as list or int
            ref = key_answers.get(q, [])
            expected = ref[0] if isinstance(ref, list) and ref else -1
            
            status = "BLANK"
            if not s_opts: blank += 1
            elif len(s_opts) > 1: status = "INVALID"; invalid += 1
            else:
                if s_opts[0] == expected: status = "CORRECT"; correct += 1
                else: status = "WRONG"; wrong += 1
            details[q] = status
        
        acc = (correct / TOTAL_QUESTIONS) * 100
        return correct, acc, details, (correct, wrong, invalid, blank)

class OMRTemplate:
    def __init__(self):
        self.dx = 30
        self.dy = 26
        self.col_starts = []
        self.start_y = 60
        self.q_per_col = 30

    def calibrate(self, key_bubbles):
        """Learn Grid layout from Key Bubbles."""
        if not key_bubbles: return
        bx = sorted([b[0] for b in key_bubbles])
        by = sorted([b[1] for b in key_bubbles])
        
        K = 5
        centroids = np.linspace(min(bx), max(bx), K)
        for _ in range(10):
            clusters = [[] for _ in range(K)]
            for x in bx:
                idx = np.argmin([abs(x - c) for c in centroids])
                clusters[idx].append(x)
            centroids = [np.mean(c) if c else centroids[i] for i, c in enumerate(clusters)]
        
        valid_clusters = [c for c in clusters if len(c) > 10]
        valid_clusters.sort(key=lambda c: np.mean(c))
        
        self.col_starts = []
        all_dx = []
        for clus in valid_clusters:
            c_sorted = sorted(clus)
            min_x = c_sorted[0]
            opt1s = [x for x in c_sorted if x < min_x + 20]
            self.col_starts.append(int(np.mean(opt1s)))
            diffs = [c_sorted[i]-c_sorted[i-1] for i in range(1, len(c_sorted)) if c_sorted[i]-c_sorted[i-1] > 10]
            if diffs: all_dx.append(statistics.mode(diffs) if len(diffs)>1 else np.mean(diffs))
            
        self.dx = int(np.mean(all_dx)) if all_dx else 30
        ydiffs = [by[i]-by[i-1] for i in range(1, len(by)) if by[i]-by[i-1] > 10]
        self.dy = statistics.mode(ydiffs) if ydiffs else 26
        self.start_y = min(by)
        self.q_per_col = TOTAL_QUESTIONS // len(self.col_starts) if self.col_starts else 30

    def generate_grid(self):
        grid = {}
        curr_q = 1
        for col_x in self.col_starts:
            for row in range(self.q_per_col):
                if curr_q > TOTAL_QUESTIONS: break
                y = self.start_y + (row * self.dy)
                for opt in range(1, 5):
                    x = col_x + ((opt-1) * self.dx)
                    grid[(curr_q, opt)] = (int(x), int(y))
                curr_q += 1
        return grid
