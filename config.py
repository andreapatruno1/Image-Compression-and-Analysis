"""
config.py — Configurazione globale del progetto.

Contiene tutti i percorsi, le costanti e i parametri condivisi tra i moduli.
"""

import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Soppressione warning ─────────────────────────────────────────────────────
warnings.filterwarnings('ignore')

# ─── Percorsi ─────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PROJECT_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "LungXRays-grayscale")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Classi diagnostiche ─────────────────────────────────────────────────────
CLASSES = ["Corona Virus Disease", "Normal", "Pneumonia", "Tuberculosis"]

# --- Parametri immagine --------------------------------------------------------
TARGET_SIZE = (256, 256)                # dimensione di ridimensionamento
SPLIT = "train"                         # split di default da utilizzare
MAX_PER_CLASS = 80                      # immagini per classe da caricare
CORRECT_TILT = True                     # abilita correzione inclinazione
MAX_TILT_ANGLE = 15                     # angolo massimo (gradi) oltre il quale l'immagine viene scartata

# ─── Parametri SVD ────────────────────────────────────────────────────────────
K_VALUES_DEMO = [1, 5, 10, 20, 50, 100, 200]       # k per la griglia di ricostruzione
K_VALUES_COMPARE = [5, 20, 50, 100]                 # k per il confronto tra classi
K_RANGE_METRICS = range(1, 201)                      # range k per curve MSE/PSNR
K_VALUES_TABLE = [1, 5, 10, 20, 50, 100, 150, 200, 256]  # k per tabella riassuntiva

# ─── Parametri PCA ────────────────────────────────────────────────────────────
PCA_N_COMPONENTS = 50                   # componenti per la PCA

# ─── Colori per le classi ─────────────────────────────────────────────────────
CLASS_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']

# ─── Stile grafici ────────────────────────────────────────────────────────────
def setup_plot_style():
    """Configura lo stile globale dei grafici matplotlib/seaborn."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12
    })
