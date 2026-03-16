import os
import sys
import numpy as np
from PIL import Image
import data_loader as dl

dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"

angles = np.arange(-50, 50, 5)

for filename in os.listdir(dataset_dir):
    if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
        fp = os.path.join(dataset_dir, filename)
        pil_img = Image.open(fp).convert('L')
        pil_img.thumbnail((256, 256))
        img = np.array(pil_img)
        img_norm = img.astype(np.float64) / 255.0

        scores = []
        for a in angles:
            rotated = dl.scipy_rotate(img_norm, -a, reshape=False, order=1, mode='constant', cval=0.0)
            proj = rotated.mean(axis=1)
            grad = np.diff(proj)
            scores.append(float(np.var(grad)))
            
        best_angle = angles[np.argmax(scores)]
        if abs(best_angle) >= 30:
            print(f"FOUND TILTED IMAGE: {filename} -> Best Angle: {best_angle}, Score: {max(scores)}")
            sys.stdout.flush()
