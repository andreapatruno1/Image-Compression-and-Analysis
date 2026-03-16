import os
import numpy as np
from PIL import Image

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jpg") or f.endswith(".png")])
for f in files[:20]:
    fp = os.path.join(dataset_dir, f)
    img = np.array(Image.open(fp).convert('L'))
    
    h, w = img.shape
    margin = 20
    
    tl = img[:margin, :margin]
    tr = img[:margin, -margin:]
    bl = img[-margin:, :margin]
    br = img[-margin:, -margin:]
    
    std_corners = [tl.std(), tr.std(), bl.std(), br.std()]
    mean_corners = [tl.mean(), tr.mean(), bl.mean(), br.mean()]
    
    num_flat_corners = sum(1 for std in std_corners if std < 1.0)
    num_bright_corners = sum(1 for m in mean_corners if m > 130)
    
    if num_flat_corners >= 2 or num_bright_corners >= 2:
         print(f"SUSPECT: {f} | STD: {[round(s, 1) for s in std_corners]} | MEAN: {[round(m, 1) for m in mean_corners]}")
    else:
         print(f"NORMAL:  {f} | STD: {[round(s, 1) for s in std_corners]} | MEAN: {[round(m, 1) for m in mean_corners]}")
