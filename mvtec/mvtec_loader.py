"""
mvtec_loader.py — Caricamento del dataset MVTec AD 'Screw' per il confronto
SVD/PCA su immagini industriali.

Strategia didattica (non è il protocollo MVTec standard):
  - Classe 'good'      = immagini normali (da train/good + test/good)
  - Classe 'defective' = aggregazione di TUTTI i tipi di difetto (test/*/*)
    (manipulated_front, scratch_head, scratch_neck, thread_side, thread_top)

Questo produce un task di classificazione binaria supervisionata, coerente
con lo stack usato sul dataset medico. MVTec è tipicamente usato come
one-class anomaly detection (training solo su 'good'); qui invece usiamo
anche le immagini difettose nel training per confrontare direttamente le
pipeline del progetto medico.

Le immagini MVTec sono 1024x1024 PNG. Vengono ridimensionate a TARGET_SIZE
(default 256x256) come nel progetto medico per avere dimensionalità
confrontabile.
"""
from __future__ import annotations

import os
from typing import Optional
import numpy as np
from PIL import Image


# ════════════════════════════════════════════════════════════════════════════
#  COSTANTI — riutilizzabili in mvtec_analysis.ipynb
# ════════════════════════════════════════════════════════════════════════════

MVTEC_CLASSES = ["good", "defective"]
MVTEC_CLASS_COLORS = ["#27ae60", "#e74c3c"]   # verde vs rosso
MVTEC_SHORT_LABELS = ["Good", "Defect"]

# Sottoclassi di difetti aggregate nella macro-classe 'defective'
MVTEC_DEFECT_TYPES = [
    "manipulated_front",
    "scratch_head",
    "scratch_neck",
    "thread_side",
    "thread_top",
]


# ════════════════════════════════════════════════════════════════════════════
#  CONTEGGIO
# ════════════════════════════════════════════════════════════════════════════

def count_mvtec_images(dataset_dir: str) -> dict:
    """
    Conta le immagini per sotto-classe nel dataset MVTec Screw.

    Parameters
    ----------
    dataset_dir : str
        Directory radice del dataset (contiene train/ e test/).

    Returns
    -------
    dict: {"good_train": N, "good_test": N, "<difetto>": N, ...}
    """
    exts = (".png", ".jpg", ".jpeg")

    def _count_files(folder: str) -> int:
        if not os.path.isdir(folder):
            return 0
        return sum(1 for f in os.listdir(folder) if f.lower().endswith(exts))

    counts = {
        "good_train": _count_files(os.path.join(dataset_dir, "train", "good")),
        "good_test":  _count_files(os.path.join(dataset_dir, "test", "good")),
    }
    for defect in MVTEC_DEFECT_TYPES:
        counts[defect] = _count_files(
            os.path.join(dataset_dir, "test", defect)
        )

    counts["good_total"] = counts["good_train"] + counts["good_test"]
    counts["defective_total"] = sum(counts[d] for d in MVTEC_DEFECT_TYPES)
    return counts


def print_mvtec_counts(counts: dict) -> None:
    """Stampa formattata del conteggio MVTec."""
    print("=" * 55)
    print("CONTEGGIO IMMAGINI — MVTec AD 'Screw'")
    print("=" * 55)
    print(f"\n--- Good (normali) ---")
    print(f"  train/good: {counts['good_train']:4d}")
    print(f"  test/good:  {counts['good_test']:4d}")
    print(f"  TOTALE good: {counts['good_total']:4d}")

    print(f"\n--- Defective (tutti i difetti aggregati) ---")
    for defect in MVTEC_DEFECT_TYPES:
        print(f"  test/{defect:<22s}: {counts[defect]:4d}")
    print(f"  TOTALE defective: {counts['defective_total']:4d}")


# ════════════════════════════════════════════════════════════════════════════
#  CARICAMENTO E PRE-ELABORAZIONE
# ════════════════════════════════════════════════════════════════════════════

def _image_files(folder: str) -> list[str]:
    """Lista ordinata dei file immagine in una cartella (PNG/JPG)."""
    exts = (".png", ".jpg", ".jpeg")
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))


def _load_image(path: str, target_size: tuple[int, int]) -> np.ndarray:
    """Carica un'immagine, la converte in scala di grigi, resize e normalizza."""
    with Image.open(path) as _raw:
        img = _raw.convert("L")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img, dtype=np.float64) / 255.0


def load_mvtec_dataset(
    dataset_dir: str,
    target_size: tuple[int, int] = (256, 256),
    max_per_class: Optional[int] = None,
    balance: bool = True,
    random_state: int = 42,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Carica il dataset MVTec Screw come problema binario good vs defective.

    Parameters
    ----------
    dataset_dir : str
        Directory radice (contiene train/ e test/).
    target_size : tuple (H, W)
        Dimensione finale dopo resize. Default 256×256 (coerente con medico).
    max_per_class : int, optional
        Se valorizzato, cap al numero di immagini per classe.
    balance : bool
        Se True (default), dopo il caricamento sotto-campiona 'good' a |defective|
        per ottenere classi bilanciate. Semplifica l'interpretazione dei
        risultati di classificazione e rispecchia il dataset medico bilanciato.
    random_state : int
        Seed per il sotto-campionamento.

    Returns
    -------
    images_by_class : dict {classe: np.ndarray (n, H, W)}
    all_images      : np.ndarray (N, H, W)
    labels          : np.ndarray di stringhe ('good' | 'defective')
    """
    rng = np.random.default_rng(random_state)

    # ── 1. GOOD: concatena train/good + test/good ──
    good_files = []
    for split in ["train", "test"]:
        folder = os.path.join(dataset_dir, split, "good")
        good_files.extend(os.path.join(folder, f) for f in _image_files(folder))

    # ── 2. DEFECTIVE: unione di tutti i tipi di difetto ──
    defective_files = []
    for defect in MVTEC_DEFECT_TYPES:
        folder = os.path.join(dataset_dir, "test", defect)
        defective_files.extend(
            os.path.join(folder, f) for f in _image_files(folder)
        )

    print(f"  Trovate {len(good_files)} immagini 'good'")
    print(f"  Trovate {len(defective_files)} immagini 'defective' "
          f"(aggregazione di {len(MVTEC_DEFECT_TYPES)} tipi di difetto)")

    # ── 3. Bilanciamento (downsampling good) ──
    if balance:
        n_target = min(len(good_files), len(defective_files))
        if len(good_files) > n_target:
            idx = rng.choice(len(good_files), n_target, replace=False)
            good_files = [good_files[i] for i in sorted(idx)]
            print(f"  [BALANCE] 'good' ridotto a {n_target} (seed={random_state})")
        if len(defective_files) > n_target:
            idx = rng.choice(len(defective_files), n_target, replace=False)
            defective_files = [defective_files[i] for i in sorted(idx)]
            print(f"  [BALANCE] 'defective' ridotto a {n_target} (seed={random_state})")

    if max_per_class is not None:
        good_files = good_files[:max_per_class]
        defective_files = defective_files[:max_per_class]

    # ── 4. Caricamento ──
    images_by_class: dict[str, np.ndarray] = {}
    all_images: list[np.ndarray] = []
    labels: list[str] = []

    for cls, paths in [("good", good_files), ("defective", defective_files)]:
        imgs = []
        for p in paths:
            img = _load_image(p, target_size)
            imgs.append(img)
            all_images.append(img)
            labels.append(cls)
        images_by_class[cls] = np.array(imgs)
        print(f"  Caricate {len(imgs):3d} immagini per '{cls}'")

    all_images_arr = np.array(all_images)
    labels_arr = np.array(labels)

    print(f"\nDataset totale: {all_images_arr.shape}  -- "
          f"{all_images_arr.shape[0]} immagini, "
          f"{target_size[0]}x{target_size[1]} pixel")

    return images_by_class, all_images_arr, labels_arr


def load_mvtec_sample_images(
    dataset_dir: str,
) -> dict[str, np.ndarray]:
    """
    Carica una immagine di esempio per ciascuna macro-classe (good, defective),
    a piena risoluzione, per la fase di esplorazione visiva.

    Returns
    -------
    dict : {classe: np.ndarray 2D (uint8, full resolution)}
    """
    samples: dict[str, np.ndarray] = {}

    # good: prende la prima di train/good
    good_folder = os.path.join(dataset_dir, "train", "good")
    good_files = _image_files(good_folder)
    if good_files:
        with Image.open(os.path.join(good_folder, good_files[0])) as _img:
            samples["good"] = np.array(_img.convert("L"))

    # defective: prende la prima del primo tipo di difetto disponibile
    for defect in MVTEC_DEFECT_TYPES:
        folder = os.path.join(dataset_dir, "test", defect)
        files = _image_files(folder)
        if files:
            with Image.open(os.path.join(folder, files[0])) as _img:
                samples["defective"] = np.array(_img.convert("L"))
            break

    return samples
