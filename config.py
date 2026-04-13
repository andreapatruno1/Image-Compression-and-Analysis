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
# Recupera il percorso della cartella principale del progetto (dove si trova questo file)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Punta alla cartella 'raw' dentro 'data'
DATASET_DIR = os.path.join(PROJECT_DIR, "data", "raw")

# Cartella per i grafici e i risultati
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Classi diagnostiche ─────────────────────────────────────────────────────
CLASSES = ["Corona Virus Disease", "Normal", "Pneumonia", "Tuberculosis"]

# --- Parametri immagine --------------------------------------------------------
TARGET_SIZE = (256, 256)                # dimensione di ridimensionamento
SPLIT = ""                              # Lasciato vuoto perché le classi sono direttamente in 'raw'
MAX_PER_CLASS = 120                      # immagini per classe da caricare
CORRECT_TILT = True                     # abilita correzione inclinazione
MAX_TILT_ANGLE = 10                     # angolo massimo (gradi) oltre il quale l'immagine viene scartata

# ─── Parametri SVD ────────────────────────────────────────────────────────────
K_VALUES_DEMO = [1, 5, 10, 20, 50, 100, 200]       # k per la griglia di ricostruzione
K_VALUES_COMPARE = [5, 20, 50, 100]                 # k per il confronto tra classi
K_RANGE_METRICS = range(1, 201)                      # range k per curve MSE/PSNR
K_VALUES_TABLE = [1, 5, 10, 20, 50, 100, 150, 200, 256]  # k per tabella riassuntiva

# ─── Parametri PCA ────────────────────────────────────────────────────────────
PCA_N_COMPONENTS = 50                   # componenti per la PCA

# ─── Parametri Classificazione ────────────────────────────────────────────────
KNN_N_NEIGHBORS = 5                     # k per KNN
LR_MAX_ITER = 2000                      # iterazioni massime per Logistic Regression
LR_C = 1.0                             # regolarizzazione inversa (1/λ)
CV_N_FOLDS = 5                          # fold per cross-validation stratificata
RANDOM_STATE = 42                       # seed per riproducibilità
SVD_K_VALUES = [10, 50]                  # ridotto per leggibilità grafici (10 per compressione estrema, 50 per alta qualità)
PCA_COMPONENTS_LIST = [25, 150]          # ridotto per leggibilità (25 per picco KNN, 150 per picco LR)

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