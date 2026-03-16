import os
import numpy as np
from PIL import Image

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jpg") or f.endswith(".png")])
discarded = []
for f in files:
    fp = os.path.join(dataset_dir, f)
    img = np.array(Image.open(fp).convert('L')).astype(np.float64)
    
    margin = 5
    tl = img[:margin, :margin]
    tr = img[:margin, -margin:]
    bl = img[-margin:, :margin]
    br = img[-margin:, -margin:]
    
    blocks = [tl, tr, bl, br]
    
    # Check if any corner is perfectly flat (artificial) AND is not pure black/white
    artificial_corners = 0
    for b in blocks:
        if b.std() < 1.0 and 5.0 < b.mean() < 250.0:
            artificial_corners += 1
            
    # Also check our previous bright corners rule (for images like 1018.jpg)
    means = [b.mean() for b in blocks]
    bright_corners = sum([1 for m in means if m > 130])
    
    if artificial_corners >= 1 or bright_corners >= 3:
        discarded.append(f)

print(f"Total discarded: {len(discarded)}")
print("First 15 discarded:")
for d in discarded[:15]:
    print(f"  {d}")
