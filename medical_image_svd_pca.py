# %% [markdown]
# # SVD e PCA per l'Analisi di Immagini Mediche
#
# **Progetto — Statistical Methods**
#
# Questo notebook dimostra come le tecniche di **Singular Value Decomposition (SVD)** e
# **Principal Component Analysis (PCA)** possano essere utilizzate per comprimere e analizzare
# radiografie polmonari in scala di grigi.
#
# **Dataset:** Lung X-Rays Grayscale — 4 classi:
# - Corona Virus Disease
# - Normal
# - Pneumonia
# - Tuberculosis

# %%
# === IMPORT DEI MODULI DEL PROGETTO ===
from config import setup_plot_style
from data_loader import (count_images, print_image_counts,
                         load_sample_images, print_sample_properties,
                         load_dataset, load_raw_samples)
from svd_engine import apply_svd, print_svd_info
from visualization import (plot_exploration, plot_svd_reconstruction,
                           plot_class_comparison, plot_tilt_correction)
from src.analysis import (plot_scree, plot_mse_psnr, run_pca, plot_pca_scatter,
                          plot_eigenfaces, print_summary_table)
from src.classification import (prepare_feature_sets, run_classification,
                                plot_confusion_matrix, plot_roc_curves,
                                plot_classification_comparison,
                                plot_hero_tradeoff)

setup_plot_style()

# %% [markdown]
# ---
# ## Fase 1 -- Esplorazione dei Dati
# Carichiamo le immagini, verifichiamo che siano matrici 2D in scala di grigi e ne studiamo le
# caratteristiche statistiche di base.

# %%
# === CONTEGGIO IMMAGINI PER CLASSE ===
counts = count_images()
print_image_counts(counts)

# %%
# === CARICAMENTO E VERIFICA CAMPIONI ===
sample_images = load_sample_images()
print_sample_properties(sample_images)

# %%
# === VISUALIZZAZIONE CAMPIONI + ISTOGRAMMA DEI PIXEL ===
plot_exploration(sample_images)

# %% [markdown]
# ---
# ## Fase 2 -- Pre-elaborazione
# Ridimensioniamo tutte le immagini a 256x256, le normalizziamo in [0, 1] e correggiamo
# l'eventuale inclinazione (tilt) delle radiografie.

# %%
# === CARICAMENTO BATCH PRE-ELABORATO ===
images_by_class, all_images, labels = load_dataset()

# %%
# === CORREZIONE TILT: PRIMA vs DOPO ===
raw_samples = load_raw_samples()
plot_tilt_correction(raw_samples)

# %% [markdown]
# ---
# ## Fase 3 -- Il Motore Matematico: SVD
#
# La **Singular Value Decomposition** scompone una matrice $A_{m \times n}$ come:
#
# $$A = U \cdot \Sigma \cdot V^T$$
#
# dove:
# - $U$ (m x m): vettori singolari sinistri (strutture delle righe)
# - $\Sigma$ (diagonale): valori singolari ordinati $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$
# - $V^T$ (n x n): vettori singolari destri (strutture delle colonne)
#
# **Approssimazione di rango k:**
#
# $$A_k = \sum_{i=1}^{k} \sigma_i \, \mathbf{u}_i \, \mathbf{v}_i^T$$
#
# Usando solo le prime $k$ componenti si ottiene la migliore approssimazione di rango $k$
# (teorema di Eckart-Young).

# %%
# === DEMO SVD SU UNA SINGOLA IMMAGINE ===
demo_img = images_by_class["Normal"][0]
U, S, Vt = apply_svd(demo_img)
print_svd_info(demo_img, U, S, Vt)

# %% [markdown]
# ---
# ## Fase 4 -- Ricostruzione e Visualizzazione
# Ricostruiamo l'immagine con diversi valori di $k$ e confrontiamo visivamente la qualita'.

# %%
# === RICOSTRUZIONE CON DIVERSI k ===
plot_svd_reconstruction(demo_img, class_name="Normal")

# %%
# === CONFRONTO TRA CLASSI ===
plot_class_comparison(images_by_class)

# %% [markdown]
# ---
# ## Fase 5 -- Analisi Statistica e PCA
#
# ### 5.1 Varianza Spiegata dai Valori Singolari (Scree Plot)
# La varianza spiegata dalla $i$-esima componente e' proporzionale a $\sigma_i^2$.
# La **varianza cumulativa** ci dice quante componenti servono per catturare una certa
# percentuale dell'informazione.
# Il grafico a destra è in scala logaritmica, per espandere visivamente le
# differenze tra le classi nella coda dei valori singolari

# %%
# === SCREE PLOT ===
plot_scree(images_by_class)

# %%
# === CURVA MSE e PSNR vs k ===
plot_mse_psnr(images_by_class)

# %% [markdown]
# ### 5.2 PCA -- Riduzione Dimensionale sul Dataset
#
# La PCA proietta le immagini (vettorizzate) in uno spazio a bassa dimensionalita'.
# Questo ci permette di visualizzare la **separabilita'** tra le classi diagnostiche.

# %%
# === PCA SUL DATASET ===
pca_model, X_pca = run_pca(all_images)

# %%
# === SCATTER PLOT PCA 2D ===
plot_pca_scatter(pca_model, X_pca, labels)

# %%
# === EIGENFACES (COMPONENTI PRINCIPALI COME IMMAGINI) ===
plot_eigenfaces(pca_model)

# %%
# === TABELLA RIASSUNTIVA ===
print_summary_table(demo_img, class_name="Normal")

# %% [markdown]
# ---
# ## Fase 6 -- Classificazione e Confronto Feature
#
# Per validare quantitativamente l'effetto della compressione/riduzione sulla capacita'
# diagnostica, addestriamo un classificatore **KNN (k=5)** su diverse rappresentazioni
# delle stesse immagini, organizzate in **due strategie distinte**:
#
# ### Strategia 1: SVD -- Compressione dell'immagine
# Si applica la SVD troncata **a ogni singola immagine** per comprimerla, poi si
# classificano i pixel ricostruiti (sempre 65.536 feature).
#
# | Scenario | Dati memorizzati | Dim. al KNN |
# |---|---|---|
# | **Raw Pixels** | 100% | 65.536 |
# | **SVD k=5** | ~3.9% | 65.536 |
# | **SVD k=10** | ~7.8% | 65.536 |
# | **SVD k=20** | ~15.7% | 65.536 |
# | **SVD k=50** | ~39.1% | 65.536 |
#
# ### Strategia 2: PCA -- Riduzione dimensionale del dataset
# Si applica PCA **all'intero dataset** (matrice 320x65.536) per estrarre le direzioni
# di massima varianza. Il KNN lavora nello spazio ridotto.
#
# | Scenario | Componenti | Riduzione dim. |
# |---|---|---|
# | **Raw Pixels** | 65.536 | 0% |
# | **PCA (10)** | 10 | 99.98% |
# | **PCA (25)** | 25 | 99.96% |
# | **PCA (50)** | 50 | 99.92% |
# | **PCA (100)** | 100 | 99.85% |
# | **PCA (150)** | 150 | 99.77% |
#
# Per evitare **data leakage**, StandardScaler e PCA vengono fittati solo sul training set
# di ogni fold tramite `sklearn.Pipeline`.
#
# La validazione avviene tramite **Stratified 5-Fold Cross-Validation** per garantire
# robustezza con il dataset limitato (~320 campioni).

# %%
# === 6.1 PREPARAZIONE SCENARI ===
scenarios = prepare_feature_sets(all_images, labels)

# %%
# === 6.2 CLASSIFICAZIONE KNN CON CROSS-VALIDATION ===
results = run_classification(scenarios, labels)

# %%
# === 6.3 CONFUSION MATRIX PER SCENARIO ===
plot_confusion_matrix(scenarios, labels)

# %%
# === 6.4 CURVE ROC MULTICLASSE (ONE-VS-REST) ===
plot_roc_curves(scenarios, labels)

# %%
# === 6.5 CONFRONTO METRICHE TRA SCENARI ===
plot_classification_comparison(results)

# %%
# === 6.6 HERO CHART: ACCURACY vs COMPRESSIONE ===
plot_hero_tradeoff(results)

# %% [markdown]
# ---
# ## Conclusioni
#
# 1. **SVD come strumento di compressione**: Con sole 20-50 componenti singolari (su 256
#    possibili), preserviamo oltre il 90% dell'informazione visiva. Il trade-off tra peso
#    dati e qualita' e' chiaramente visibile nei grafici MSE/PSNR.
#
# 2. **Varianza Spiegata**: Lo scree plot mostra che le prime componenti catturano la
#    maggior parte della varianza. Questo conferma che le immagini mediche hanno una forte
#    struttura di basso rango sfruttabile per la compressione.
#
# 3. **PCA e Separabilita'**: La proiezione PCA 2D mostra come le diverse patologie
#    occupino regioni (parzialmente) distinte nello spazio delle componenti principali,
#    suggerendo che queste tecniche possono supportare anche la classificazione diagnostica.
#
# 4. **Eigenfaces (Eigen-Xrays)**: Le prime componenti principali rivelano i pattern
#    strutturali dominanti nelle radiografie polmonari.
#
# 5. **SVD: Compressione vs Classificazione**: Comprimere le immagini con SVD troncata
#    (k=5 a k=50) e poi classificare i pixel ricostruiti produce accuracy comparabili
#    ai pixel grezzi (~80%). Questo dimostra che il rumore ad alta frequenza rimosso
#    dalla SVD non contiene informazione diagnostica rilevante.
#
# 6. **PCA: Riduzione Dimensionale vs Classificazione**: Proiettare il dataset su
#    10-150 componenti principali (riducendo la dimensionalita' fino al 99.98%) mantiene
#    la capacita' discriminativa. PCA agisce come feature extractor, selezionando le
#    direzioni di massima varianza inter-immagine.
#
# 7. **Conclusione Chiave** (Hero Chart): I due grafici affiancati mostrano che entrambe
#    le strategie -- compressione SVD dell'immagine e riduzione PCA del dataset --
#    preservano la capacita' diagnostica rispetto alla baseline Raw Pixels, confermando
#    il valore pratico di SVD/PCA nella pipeline di analisi di immagini mediche.
