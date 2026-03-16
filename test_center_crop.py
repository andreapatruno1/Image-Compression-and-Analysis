import os
import numpy as np
from PIL import Image
import data_loader as dl

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

for f in ["1018.jpg", "1023.jpg"]:
    fp = os.path.join(dataset_dir, f)
    pil_img = Image.open(fp).convert('L')
    img = np.array(pil_img)
    
    # Standard detection
    angle_full = dl.detect_tilt_angle(img, search_range=45)
    
    # Detection on center crop (focus on ribs, avoid borders)
    h, w = img.shape
    cy, cx = h//2, w//2
    # crop 60% of center
    dy, dx = int(h*0.3), int(w*0.3)
    img_crop = img[cy-dy:cy+dy, cx-dx:cx+dx]
    angle_crop = dl.detect_tilt_angle(img_crop, search_range=45)
    
    print(f"File: {f} | Full Angle: {angle_full:.1f} | Center Crop Angle: {angle_crop:.1f}")
