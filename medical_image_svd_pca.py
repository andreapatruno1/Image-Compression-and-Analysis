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
from config import setup_plot_style, CLASSES
from data_loader import (count_images, print_image_counts,
                         load_sample_images, print_sample_properties,
                         load_dataset)
from svd_engine import apply_svd, print_svd_info
from visualization import (plot_exploration, plot_svd_reconstruction,
                           plot_class_comparison)
from analysis import (plot_scree, plot_mse_psnr, run_pca, plot_pca_scatter,
                      plot_eigenfaces, print_summary_table)

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
# Ridimensioniamo tutte le immagini a 256x256 e le normalizziamo in [0, 1].

# %%
# === CARICAMENTO BATCH PRE-ELABORATO ===
images_by_class, all_images, labels = load_dataset()

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
