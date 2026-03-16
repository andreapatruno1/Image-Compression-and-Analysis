import os
import sys
import numpy as np
from PIL import Image
import data_loader as dl

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

print("Starting scan...")
for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
        fp = os.path.join(dataset_dir, filename)
        try:
            pil_img = Image.open(fp).convert('L')
            
            # shrink image to speed up calculation
            pil_img.thumbnail((256, 256))
            img = np.array(pil_img)
        except Exception as e:
            continue
        
        # calculate max range 45
        angle_45 = dl.detect_tilt_angle(img, search_range=45.0)
        
        if abs(angle_45) > 10.0:
            print(f"HIGH TILT FOUND: {filename} -> Angle(45): {angle_45:.1f}")
            sys.stdout.flush()
        else:
            print(f"{filename} -> {angle_45:.1f}", end=" | ")
            sys.stdout.flush()
