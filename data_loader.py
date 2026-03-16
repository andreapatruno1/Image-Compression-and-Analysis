"""
data_loader.py -- Fase 1 & 2: Esplorazione e Pre-elaborazione dei Dati.

Funzioni per caricare, verificare e pre-elaborare le immagini radiografiche.
Include il rilevamento e la correzione automatica dell'inclinazione (tilt).
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage import rotate as scipy_rotate

from config import (DATASET_DIR, CLASSES, TARGET_SIZE, SPLIT, MAX_PER_CLASS,
                    CORRECT_TILT, MAX_TILT_ANGLE)


# ===========================================================================
#  FASE 1 -- ESPLORAZIONE
# ===========================================================================

def _image_files(folder: str) -> list[str]:
    """Restituisce la lista ordinata dei file immagine in una cartella."""
    exts = ('.jpg', '.png', '.jpeg')
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))


def count_images() -> dict:
    """
    Conta le immagini per classe nel training set.

    Returns
    -------
    dict : {classe: n_immagini}
    """
    counts = {}
    for cls in CLASSES:
        folder = os.path.join(DATASET_DIR, "train", cls)
        counts[cls] = len(_image_files(folder))
    return {"train": counts}


def print_image_counts(counts: dict) -> None:
    """Stampa in modo formattato il conteggio delle immagini."""
    print("=" * 55)
    print("CONTEGGIO IMMAGINI NEL DATASET")
    print("=" * 55)
    for split, class_counts in counts.items():
        print(f"\n--- {split.upper()} ---")
        for cls, n in class_counts.items():
            print(f"  {cls:30s}: {n:4d} immagini")


def load_sample_images(split: str = SPLIT) -> dict[str, np.ndarray]:
    """
    Carica la prima immagine grezza (uint8) per ciascuna classe,
    escludendo le radiografie laterali.

    Returns
    -------
    dict : {classe: np.ndarray}  -- immagini originali non ridimensionate
    """
    samples = {}
    for cls in CLASSES:
        folder = os.path.join(DATASET_DIR, split, cls)
        files = _image_files(folder)
        for f in files:
            img = Image.open(os.path.join(folder, f)).convert('L')
            arr = np.array(img)
            if not is_lateral_xray(arr):
                samples[cls] = arr
                break
    return samples


def print_sample_properties(sample_images: dict[str, np.ndarray]) -> None:
    """Stampa le proprieta' delle immagini campione (shape, dtype, range, ...)."""
    print("\n" + "=" * 55)
    print("PROPRIETA' DELLE IMMAGINI CAMPIONE")
    print("=" * 55)
    for cls, arr in sample_images.items():
        print(f"\n  Classe: {cls}")
        print(f"    Shape:  {arr.shape}  (altezza x larghezza)")
        print(f"    Dtype:  {arr.dtype}")
        print(f"    Min:    {arr.min()}")
        print(f"    Max:    {arr.max()}")
        print(f"    Media:  {arr.mean():.2f}")
        print(f"    Std:    {arr.std():.2f}")
        print(f"    2D?     {'Si [OK]' if arr.ndim == 2 else 'No [!!] -- ATTENZIONE'}")


# ===========================================================================
#  RILEVAMENTO IMMAGINI LATERALI
# ===========================================================================

def is_lateral_xray(image: np.ndarray) -> bool:
    """
    Rileva se un'immagine è una radiografia laterale (profilo) anziché PA.

    Le radiografie laterali mostrano un corpo stretto con grandi bande nere
    ai lati. Si verifica se la distribuzione del segnale lungo le colonne
    è molto concentrata al centro rispetto alla larghezza totale.

    Parameters
    ----------
    image : np.ndarray  (2D, qualsiasi range)

    Returns
    -------
    bool : True se l'immagine appare laterale
    """
    if image.max() <= 1.0:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)

    # Proiezione media lungo le righe → profilo orizzontale
    col_profile = img.mean(axis=0)
    threshold = col_profile.max() * 0.3

    # Conta le colonne con segnale significativo
    active_cols = np.sum(col_profile > threshold)
    ratio = active_cols / len(col_profile)

    # Se meno del 60% delle colonne ha segnale → laterale
    return ratio < 0.60


# ===========================================================================
#  CORREZIONE INCLINAZIONE (TILT)
# ===========================================================================

def detect_tilt_angle(image: np.ndarray) -> float:
    """
    Rileva l'angolo di inclinazione di una radiografia usando i momenti centrali
    del secondo ordine della regione del corpo (pixel sopra la soglia di Otsu).

    L'angolo viene calcolato come:
        theta = 0.5 * arctan(2 * mu11 / (mu20 - mu02))

    dove mu20, mu02, mu11 sono i momenti centrali del 2o ordine.

    Parameters
    ----------
    image : np.ndarray
        Immagine in scala di grigi (2D), valori in [0, 255] o [0, 1].

    Returns
    -------
    float : angolo in gradi (positivo = senso antiorario)
    """
    # Normalizza in [0, 255] se necessario
    if image.max() <= 1.0:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)

    # Soglia di Otsu semplificata (media pesata per separare foreground/background)
    threshold = img.mean()
    binary = (img > threshold).astype(np.float64)

    # Se l'immagine e' quasi tutta nera o tutta bianca, non e' ruotata
    fg_ratio = binary.mean()
    if fg_ratio < 0.05 or fg_ratio > 0.95:
        return 0.0

    # Coordinate dei pixel di foreground
    rows, cols = np.where(binary > 0)
    if len(rows) < 100:
        return 0.0

    # Centroide
    cy = rows.mean()
    cx = cols.mean()

    # Momenti centrali del 2o ordine
    mu20 = np.mean((rows - cy) ** 2)
    mu02 = np.mean((cols - cx) ** 2)
    mu11 = np.mean((rows - cy) * (cols - cx))

    # Angolo dell'asse principale
    denom = mu20 - mu02
    if abs(denom) < 1e-10:
        return 0.0

    theta_rad = 0.5 * np.arctan2(2 * mu11, denom)
    theta_deg = np.degrees(theta_rad)

    return theta_deg


def correct_tilt(image: np.ndarray, angle: float = None) -> tuple[np.ndarray, float]:
    """
    Corregge l'inclinazione di un'immagine ruotandola.

    Parameters
    ----------
    image : np.ndarray
        Immagine 2D (qualsiasi range).
    angle : float, optional
        Angolo da correggere (gradi). Se None, viene rilevato automaticamente.

    Returns
    -------
    corrected : np.ndarray  -- immagine ruotata
    angle     : float       -- angolo applicato (gradi)
    """
    if angle is None:
        angle = detect_tilt_angle(image)

    if abs(angle) < 0.5:
        return image, 0.0

    # Ruota in senso opposto per correggere
    corrected = scipy_rotate(image, -angle, reshape=False, order=1, mode='constant',
                             cval=0.0)
    return corrected, angle


def is_valid_image(image: np.ndarray, max_tilt: float = MAX_TILT_ANGLE) -> tuple[bool, float]:
    """
    Verifica se un'immagine e' valida (non troppo inclinata).

    Returns
    -------
    valid : bool
    angle : float -- angolo rilevato
    """
    angle = detect_tilt_angle(image)
    return abs(angle) <= max_tilt, angle


# ===========================================================================
#  FASE 2 -- PRE-ELABORAZIONE
# ===========================================================================

def load_and_preprocess(
    path: str,
    target_size: tuple = TARGET_SIZE,
    apply_tilt_correction: bool = CORRECT_TILT
) -> tuple[np.ndarray, float]:
    """
    Carica un'immagine, la converte in scala di grigi, corregge l'inclinazione
    (opzionale), la ridimensiona e la normalizza in [0, 1].

    Parameters
    ----------
    path : str
        Percorso assoluto del file immagine.
    target_size : tuple
        Dimensione (larghezza, altezza) di output.
    apply_tilt_correction : bool
        Se True, rileva e corregge l'inclinazione.

    Returns
    -------
    image : np.ndarray float64 normalizzata [0, 1]
    angle : float -- angolo di correzione applicato (0 se non corretto)
    """
    img = Image.open(path).convert('L')
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0

    angle = 0.0
    if apply_tilt_correction:
        arr, angle = correct_tilt(arr)
        arr = np.clip(arr, 0, 1)

    return arr, angle


def load_dataset(
    split: str = SPLIT,
    max_per_class: int = MAX_PER_CLASS,
    target_size: tuple = TARGET_SIZE,
    apply_tilt_correction: bool = CORRECT_TILT,
    max_tilt: float = MAX_TILT_ANGLE
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Carica e pre-elabora un batch di immagini da ciascuna classe.
    Se la correzione tilt e' abilitata, corregge l'inclinazione e scarta
    le immagini con angolo superiore a max_tilt.

    Returns
    -------
    images_by_class : dict {classe: np.ndarray di shape (n, H, W)}
    all_images      : np.ndarray di shape (N_totale, H, W)
    labels          : np.ndarray di stringhe con la classe di ogni immagine
    """
    images_by_class = {}
    all_images = []
    labels = []

    for cls in CLASSES:
        folder = os.path.join(DATASET_DIR, split, cls)
        files = _image_files(folder)
        class_imgs = []
        skipped_tilt = 0
        skipped_lateral = 0

        for f in files:
            if len(class_imgs) >= max_per_class:
                break

            filepath = os.path.join(folder, f)

            # Controlla se è una radiografia laterale (prima del preprocessing)
            raw = Image.open(filepath).convert('L')
            raw_arr = np.array(raw)
            if is_lateral_xray(raw_arr):
                skipped_lateral += 1
                continue

            img, angle = load_and_preprocess(
                filepath, target_size, apply_tilt_correction
            )

            # Scarta immagini troppo inclinate
            if apply_tilt_correction and abs(angle) > max_tilt:
                skipped_tilt += 1
                continue

            class_imgs.append(img)
            all_images.append(img)
            labels.append(cls)

        images_by_class[cls] = np.array(class_imgs)
        msg = f"  Caricate {len(class_imgs):3d} immagini per '{cls}'"
        if skipped_tilt > 0:
            msg += f"  (scartate {skipped_tilt} con tilt > {max_tilt}°)"
        if skipped_lateral > 0:
            msg += f"  (scartate {skipped_lateral} laterali)"
        print(msg)

    all_images = np.array(all_images)
    labels = np.array(labels)
    print(f"\nDataset totale: {all_images.shape}  --  {all_images.shape[0]} immagini, "
          f"{all_images.shape[1]}x{all_images.shape[2]} pixel")

    return images_by_class, all_images, labels


def load_raw_samples(
    split: str = SPLIT,
    n_samples: int = 3
) -> dict[str, list[tuple[np.ndarray, str]]]:
    """
    Carica immagini RAW (solo resize, SENZA correzione tilt) per il confronto
    "prima vs dopo". Filtra le immagini laterali.

    Returns
    -------
    dict : {classe: [(img_raw_resized, filename), ...]}
    """
    raw_by_class = {}
    for cls in CLASSES:
        folder = os.path.join(DATASET_DIR, split, cls)
        files = _image_files(folder)
        samples = []
        for f in files:
            if len(samples) >= n_samples:
                break
            filepath = os.path.join(folder, f)
            raw = Image.open(filepath).convert('L')
            raw_arr = np.array(raw)
            # Salta le laterali
            if is_lateral_xray(raw_arr):
                continue
            # Resize senza correzione tilt
            img = raw.resize(TARGET_SIZE, Image.LANCZOS)
            arr = np.array(img, dtype=np.float64) / 255.0
            samples.append((arr, f))
        raw_by_class[cls] = samples
    return raw_by_class

