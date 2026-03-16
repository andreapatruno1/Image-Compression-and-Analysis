import os
import numpy as np
from PIL import Image
import data_loader as dl

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

files = ["1018.jpg", "1023.jpg", "1048.jpg", "1084.jpg", "11.jpg", "136.jpg"]

for f in files:
    fp = os.path.join(dataset_dir, f)
    pil_img = Image.open(fp).convert('L')
    img = np.array(pil_img)
    
    h, w = img.shape
    
    # Create mask for center
    cy, cx = h//2, w//2
    # mask 60% of center (set to 0)
    dy, dx = int(h*0.3), int(w*0.3)
    
    img_periph = img.copy()
    img_periph[cy-dy:cy+dy, cx-dx:cx+dx] = 0
    
    angle_periph = dl.detect_tilt_angle(img_periph, search_range=45)
    
    print(f"File: {f} | Periphery Angle: {angle_periph:.1f}")
