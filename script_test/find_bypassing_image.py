import os
import numpy as np
from PIL import Image
import data_loader as dl
import shutil

# Carica il dataset
images_by_class, all_images, labels = dl.load_dataset(max_per_class=1)

corona_img = images_by_class["Corona Virus Disease"][0]

# Salviamo l'immagine per poterla vedere
pil_img = Image.fromarray((corona_img * 255).astype(np.uint8))
pil_img.save("debug_first_corona.png")

# Cerchiamo di capire quale file era
dataset_dir = r"c:\Users\1\OneDrive - Politecnico di Bari\POLIBA\1 anno\1 semestre\statistical metods\progetto\LungXRays-grayscale\train\Corona Virus Disease"
files = dl._image_files(dataset_dir)

for f in files:
    filepath = os.path.join(dataset_dir, f)
    raw = Image.open(filepath).convert('L')
    raw_arr = np.array(raw)
    
    if dl.is_lateral_xray(raw_arr):
        continue
        
    valid, angle = dl.is_valid_image(raw_arr, dl.MAX_TILT_ANGLE)
    if valid:
        print(f"FIRST VALID IMAGE IS: {f} with detected angle {angle}")
        break
