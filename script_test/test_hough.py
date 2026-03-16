import os
import numpy as np
from PIL import Image
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
import warnings

warnings.filterwarnings('ignore')

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jpg") or f.endswith(".png")])
for f in files[:30]:
    fp = os.path.join(dataset_dir, f)
    img = np.array(Image.open(fp).convert('L'))
    
    # Resize for speed and noise reduction
    img_small = np.array(Image.fromarray(img).resize((128, 128)))
    
    # Canny edge detection
    edges = canny(img_small, sigma=2.0)
    
    # Hough transform
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(edges, theta=tested_angles)
    
    # Peaks
    _, angles, dists = hough_line_peaks(h, theta, d, num_peaks=3)
    
    angles_deg = np.rad2deg(angles)
    
    # Check if we have strong lines that are roughly diagonal
    # (i.e. angle not near 0, 90, -90)
    diagonal_lines = []
    for a in angles_deg:
        if 15 < abs(a) < 75:
            diagonal_lines.append(a)
            
    # Check max accumulator value (strength/length of the line)
    max_h = np.max(h) if len(h) > 0 else 0
    
    if len(diagonal_lines) > 0 and max_h > 40: # threshold for 128x128 image
        print(f"DIAGONAL BORDERS: {f} | Max Hough: {max_h} | Angles: {[round(a, 1) for a in angles_deg]}")
    else:
        print(f"Normal: {f} | Max Hough: {max_h} | Angles: {[round(a, 1) for a in angles_deg]}")
