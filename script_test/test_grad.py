import os
import numpy as np
from PIL import Image

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jpg") or f.endswith(".png")])
for f in files[:30]:
    fp = os.path.join(dataset_dir, f)
    img = np.array(Image.open(fp).convert('L')).astype(np.float64)
    
    gy, gx = np.gradient(img)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Check the whole image for exactly zero gradients
    # Smeared padding has perfectly 0 gradient in the orthogonal direction,
    # but the magnitude might be non-zero if the other direction varies?
    # No, BORDER_REPLICATE means adjacent pixels are identical in the replicate direction.
    # Actually, a better check is if gy == 0 AND gx == 0.
    zero_grad_pct = np.mean(grad_mag < 1.0) * 100
    
    # Check edges specifically
    h, w = img.shape
    margin = 30
    mask = np.ones((h, w), dtype=bool)
    mask[margin:-margin, margin:-margin] = False
    
    zero_grad_margin = np.mean(grad_mag[mask] < 0.5) * 100
    
    if zero_grad_margin > 15:
        print(f"PADDED: {f} | Zero Grad Margin: {zero_grad_margin:.1f}%")
    else:
        print(f"Normal: {f} | Zero Grad Margin: {zero_grad_margin:.1f}%")
