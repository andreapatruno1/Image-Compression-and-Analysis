import os
import numpy as np
from PIL import Image

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jpg") or f.endswith(".png")])
discarded = []
for f in files:
    fp = os.path.join(dataset_dir, f)
    img = np.array(Image.open(fp).convert('L')).astype(np.float64)
    
    h, w = img.shape
    margin = int(min(h, w) * 0.05)
    
    tl = img[:margin, :margin].mean()
    tr = img[:margin, -margin:].mean()
    bl = img[-margin:, :margin].mean()
    br = img[-margin:, -margin:].mean()
    
    bright_corners = sum([1 for x in [tl, tr, bl, br] if x > 130])
    
    if bright_corners >= 2:
        discarded.append((f, [tl, tr, bl, br]))

print(f"Total discarded: {len(discarded)}")
print("First 15 discarded:")
for d in discarded[:15]:
    print(f"  {d[0]} | Means: {[round(x, 1) for x in d[1]]}")
