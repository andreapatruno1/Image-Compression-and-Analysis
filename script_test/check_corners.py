import os
import numpy as np
from PIL import Image

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")])
for f in files[:5]:
    fp = os.path.join(dataset_dir, f)
    img = np.array(Image.open(fp).convert('L'))
    
    # Check corners (e.g., 20x20 blocks)
    # usually background is dark (< 50)
    top_left = img[:20, :20].mean()
    top_right = img[:20, -20:].mean()
    bottom_left = img[-20:, :20].mean()
    bottom_right = img[-20:, -20:].mean()
    
    print(f"File: {f} | TL: {top_left:.1f}, TR: {top_right:.1f}, BL: {bottom_left:.1f}, BR: {bottom_right:.1f}")
