f"""
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
    Conta le immagini per classe nel dataset.
    """
    counts = {}
    for cls in CLASSES:
        # Usiamo SPLIT (che è "") invece di "train" fisso
        folder = os.path.join(DATASET_DIR, SPLIT, cls)

        # Aggiungiamo un controllo per evitare errori se la cartella non esiste
        if os.path.exists(folder):
            counts[cls] = len(_image_files(folder))
        else:
            print(f"Attenzione: Cartella non trovata -> {folder}")
            counts[cls] = 0

    return {"dataset": counts}


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
    Carica la prima immagine grezza (uint8) per ciascuna classe valida,
    escludendo le radiografie laterali e le immagini corrotte dal padding.

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
                valid, _ = is_valid_image(arr, filename=f)
                if valid:
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
    if col_profile.max() == 0:
        return True  # immagine completamente nera → trattare come non valida
    threshold = col_profile.max() * 0.3

    # Conta le colonne con segnale significativo
    active_cols = np.sum(col_profile > threshold)
    ratio = active_cols / len(col_profile)

    # Se meno del 60% delle colonne ha segnale → laterale
    return ratio < 0.60


# ===========================================================================
#  CORREZIONE INCLINAZIONE (TILT)
# ===========================================================================

def detect_tilt_angle(image: np.ndarray, search_range: float = 25.0) -> float:
    """
    Rileva l'angolo di inclinazione di una radiografia usando la massimizzazione
    della varianza della proiezione orizzontale.

    **Principio**: nelle radiografie toraciche le costole creano strutture
    fortemente orizzontali. Quando l'immagine è correttamente orientata, la
    proiezione media lungo le righe presenta picchi netti (alta varianza del
    gradiente). Per ogni angolo candidato θ si ruota l'immagine di -θ (annullando
    l'ipotetica inclinazione) e si misura la "nitidezza" del profilo risultante.
    L'angolo che massimizza tale nitidezza è l'inclinazione rilevata.

    Utilizza una ricerca a due passaggi (coarse → fine) per efficienza.

    Parameters
    ----------
    image : np.ndarray
        Immagine in scala di grigi (2D), valori in [0, 255] o [0, 1].
    search_range : float
        Range di ricerca in gradi (±search_range).

    Returns
    -------
    float : angolo di inclinazione rilevato (gradi). Positivo = antiorario.
    """
    # Normalizza in [0, 1] se necessario
    if image.max() > 1.0:
        img = image.astype(np.float64) / 255.0
    else:
        img = image.astype(np.float64)

    def _projection_score(img_arr: np.ndarray, angle: float) -> float:
        """Calcola lo score di allineamento per un dato angolo candidato."""
        if abs(angle) < 0.01:
            rotated = img_arr
        else:
            rotated = scipy_rotate(img_arr, -angle, reshape=False,
                                   order=1, mode='constant', cval=0.0)
        # Proiezione orizzontale: media di ciascuna riga
        projection = rotated.mean(axis=1)
        # Gradiente della proiezione — più nitido = costole meglio allineate
        grad = np.diff(projection)
        return float(np.var(grad))

    # --- Passo 1: ricerca grossolana (step = 2°) ---
    coarse_angles = np.arange(-search_range, search_range + 2.0, 2.0)
    coarse_scores = np.array([_projection_score(img, a) for a in coarse_angles])
    best_coarse = coarse_angles[np.argmax(coarse_scores)]

    # --- Passo 2: ricerca fine attorno al miglior angolo (step = 0.5°) ---
    fine_lo = max(-search_range, best_coarse - 3.0)
    fine_hi = min(search_range, best_coarse + 3.0)
    fine_angles = np.arange(fine_lo, fine_hi + 0.5, 0.5)
    fine_scores = np.array([_projection_score(img, a) for a in fine_angles])
    best_angle = float(fine_angles[np.argmax(fine_scores)])

    return best_angle


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


# Immagini note per avere padding rotazionale grave e angoli finti che ingannano gli algoritmi
KNOWN_CORRUPT_FILES = {
    "1018.jpg", "1023.jpg", "1030.jpg", "1048.jpg", "1057.jpg", "1066.jpg",
    "1078.jpg", "1099.jpg", "1100.jpg", "1132.jpg", "1133.jpg", "145.jpg"
}

def is_valid_image(image: np.ndarray, max_tilt: float = MAX_TILT_ANGLE, filename: str = None) -> tuple[bool, float]:
    """
    Verifica se un'immagine e' valida (non troppo inclinata e non corrotta da padding).

    Returns
    -------
    valid : bool
    angle : float -- angolo rilevato
    """
    if filename and filename in KNOWN_CORRUPT_FILES:
        return False, 99.0
        
    # 1. Controllo padding artificiale (angoli molto chiari = immagine ruotata e riempita)
    # Calcoliamo la media su blocchi 20x20 agli angoli
    if image.max() <= 1.0:
        img_255 = (image * 255).astype(np.float64)
    else:
        img_255 = image.astype(np.float64)
    
    h, w = img_255.shape
    margin = int(min(h, w) * 0.05) # 5% margin
    
    tl = img_255[:margin, :margin].mean()
    tr = img_255[:margin, -margin:].mean()
    bl = img_255[-margin:, :margin].mean()
    br = img_255[-margin:, -margin:].mean()
    
    # Se 3 o 4 angoli sono molto chiari (>130), l'immagine è corrotta/paddata
    bright_corners = sum([1 for x in [tl, tr, bl, br] if x > 130])
    if bright_corners >= 3:
        return False, 99.0  # Identificato come tilt artificiale grave

    # 2. Controllo tilt standard
    angle = detect_tilt_angle(image)
    return abs(angle) <= max_tilt, angle


# ===========================================================================
#  FASE 2 -- PRE-ELABORAZIONE
# ===========================================================================

def load_and_preprocess(
    path: str,
    target_size: tuple = TARGET_SIZE,
    apply_tilt_correction: bool = CORRECT_TILT,
    pil_image: Image.Image = None,
    angle: float = None
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
    pil_image : PIL.Image, optional
        Immagine gia' caricata (evita di riaprire il file).
    angle : float, optional
        Angolo di inclinazione pre-calcolato per evitare doppi calcoli.

    Returns
    -------
    image : np.ndarray float64 normalizzata [0, 1]
    angle : float -- angolo di correzione applicato (0 se non corretto)
    """
    if pil_image is not None:
        img = pil_image
    else:
        img = Image.open(path).convert('L')
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0

    if apply_tilt_correction:
        arr, angle = correct_tilt(arr, angle=angle)
        arr = np.clip(arr, 0, 1)
    else:
        angle = 0.0

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

            # Carica l'immagine una sola volta
            raw = Image.open(filepath).convert('L')
            raw_arr = np.array(raw)

            # Controlla se è una radiografia laterale (prima del preprocessing)
            if is_lateral_xray(raw_arr):
                skipped_lateral += 1
                continue

            # Controlla se l'immagine è valida (padding corrotto o inclinazione eccesiva)
            if apply_tilt_correction:
                valid, angle = is_valid_image(raw_arr, max_tilt, filename=f)
                if not valid:
                    skipped_tilt += 1
                    continue
            else:
                angle = 0.0

            # Passa l'immagine gia' aperta per evitare un secondo caricamento
            # Passiamo anche l'angolo calcolato per non ricalcolarlo
            img, _ = load_and_preprocess(
                filepath, target_size, apply_tilt_correction,
                pil_image=raw, angle=angle
            )

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

