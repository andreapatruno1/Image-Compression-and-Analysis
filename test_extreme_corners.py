import os
import numpy as np
from PIL import Image

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jpg") or f.endswith(".png")])
for f in files[:30]:
    fp = os.path.join(dataset_dir, f)
    img = np.array(Image.open(fp).convert('L')).astype(np.float64)
    
    # Check 10x10 corners
    margin = 5
    tl = img[:margin, :margin]
    tr = img[:margin, -margin:]
    bl = img[-margin:, :margin]
    br = img[-margin:, -margin:]
    
    stds = [tl.std(), tr.std(), bl.std(), br.std()]
    
    if any(s < 0.5 for s in stds):
         print(f"CORRUPT PADDING: {f} | STDs: {[round(s, 2) for s in stds]}")
    else:
         print(f"Normal: {f} | STDs: {[round(s, 2) for s in stds]}")
