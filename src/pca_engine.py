"""
pca_engine.py — Fase 5.3: PCA Engine.

Esegue la PCA sul dataset, scatter plot 2D e visualizzazione delle eigenfaces.
"""

import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .config import (CLASSES, CLASS_COLORS, OUTPUT_DIR, TARGET_SIZE,
                     PCA_N_COMPONENTS)


# ═══════════════════════════════════════════════════════════════════════════════
#  PCA SUL DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def run_pca(
    all_images: np.ndarray,
    n_components: int = PCA_N_COMPONENTS
) -> tuple[PCA, np.ndarray]:
    """
    Esegue la PCA sulle immagini vettorizzate.

    Returns
    -------
    pca   : modello PCA fittato
    X_pca : proiezione dei dati (N, n_components)
    """
    X = all_images.reshape(all_images.shape[0], -1)
    print(f"Matrice dati X: {X.shape}  ({X.shape[0]} campioni, {X.shape[1]} features)")

    pca_model = PCA(n_components=n_components)
    X_pca = pca_model.fit_transform(X)

    print(f"\nVarianza spiegata dalle prime {n_components} componenti: "
          f"{pca_model.explained_variance_ratio_.sum():.2%}")

    return pca_model, X_pca


# ═══════════════════════════════════════════════════════════════════════════════
#  SCATTER PLOT PCA 2D
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pca_scatter(
    pca_model: PCA,
    X_pca: np.ndarray,
    labels: np.ndarray,
    save: bool = True,
    classes=None, class_colors=None, output_dir=None,
    title: Optional[str] = None,
) -> None:
    """Scatter plot PCA 2D + varianza spiegata per componente."""
    if classes is None:
        classes = CLASSES
    if class_colors is None:
        class_colors = CLASS_COLORS
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for i, cls in enumerate(classes):
        mask = labels == cls
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        alpha=0.6, s=30, label=cls, color=class_colors[i],
                        edgecolors='white', linewidth=0.3)

    axes[0].set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%} varianza)')
    axes[0].set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%} varianza)')
    scatter_title = title if title is not None else 'PCA — Proiezione 2D delle Radiografie'
    axes[0].set_title(scatter_title, fontweight='bold')
    axes[0].legend(fontsize=10)

    n_comp = len(pca_model.explained_variance_ratio_)
    cum_var = np.cumsum(pca_model.explained_variance_ratio_)
    axes[1].bar(range(1, n_comp + 1), pca_model.explained_variance_ratio_,
                color='steelblue', alpha=0.7, label='Singola')
    axes[1].plot(range(1, n_comp + 1), cum_var,
                 color='#e74c3c', linewidth=2.5, marker='o', markersize=4, label='Cumulativa')
    axes[1].axhline(y=0.90, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Componente Principale')
    axes[1].set_ylabel('Varianza Spiegata')
    axes[1].set_title('PCA — Varianza Spiegata per Componente', fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, 'fase5_pca_scatter.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  EIGENFACES (EIGEN X-RAYS)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eigenfaces(pca_model: PCA, n_show: int = 10, save: bool = True,
                    output_dir=None, target_size=None) -> None:
    """
    Visualizza le prime n componenti principali come immagini (Eigen X-Rays).

    Miglioramenti rispetto alla versione precedente:
    - La prima cella mostra la mean face (pca_model.mean_) come riferimento
    - Scala simmetrica comune vmin/vmax tra tutte le componenti, per rendere
      i colori comparabili in magnitudine
    - Colorbar condivisa che indica l'unità di misura dei coefficienti
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if target_size is None:
        target_size = TARGET_SIZE

    # ── 1. Mean face (immagine di riferimento) ─────────────────────────────────
    mean_face = pca_model.mean_.reshape(target_size)

    # ── 2. Calcola vmin/vmax simmetrico comune su TUTTE le n_show componenti ──
    #    Usa il 99° percentile per escludere outlier estremi
    all_vals = np.concatenate([pca_model.components_[i] for i in range(n_show)])
    vmax = np.percentile(np.abs(all_vals), 99)
    vmin = -vmax   # simmetria attorno allo zero

    # ── 3. Layout: 1 colonna per la mean face + n_show eigenfaces ─────────────
    n_total = n_show + 1               # +1 per la mean face
    n_cols  = 5
    n_rows  = (n_total + n_cols - 1) // n_cols   # ceil division

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(18, 3.8 * n_rows),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.05})
    axes = axes.ravel()

    # ── 4. Mean face nella prima cella (colormap grigia, valori [0,1]) ─────────
    im_mean = axes[0].imshow(mean_face, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Mean face\n(riferimento)', fontweight='bold', fontsize=10)
    axes[0].axis('off')
    # colorbar dedicata per la mean face
    cbar_mean = fig.colorbar(im_mean, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_mean.set_label('Intensità norm.', fontsize=7)
    cbar_mean.ax.tick_params(labelsize=6)

    # ── 5. Eigenfaces con scala comune ────────────────────────────────────────
    im_ref = None   # salviamo uno per la colorbar condivisa
    for i in range(n_show):
        eigenface = pca_model.components_[i].reshape(target_size)
        ax = axes[i + 1]
        im = ax.imshow(eigenface, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(f'PC {i+1}  ({pca_model.explained_variance_ratio_[i]:.1%})',
                     fontweight='bold', fontsize=10)
        ax.axis('off')
        if i == 0:
            im_ref = im   # riferimento per la colorbar condivisa

    # ── 6. Nascondi celle vuote ────────────────────────────────────────────────
    for j in range(n_total, len(axes)):
        axes[j].axis('off')

    # ── 7. Colorbar condivisa per tutte le eigenfaces ─────────────────────────
    #    Agganciata all'area delle eigenfaces (tutte le celle tranne la prima)
    eigenface_axes = axes[1:n_show + 1]
    cbar = fig.colorbar(im_ref, ax=eigenface_axes.tolist(),
                        fraction=0.015, pad=0.02, shrink=0.85)
    cbar.set_label('Coefficiente componente principale\n(rosso = +, blu = −)',
                   fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.suptitle('Le prime 10 Componenti Principali ("Eigen-Xrays")\n'
                 'Scala cromatica comune — colori comparabili tra componenti',
                 fontsize=14, fontweight='bold', y=1.01)

    if save:
        plt.savefig(os.path.join(output_dir, 'fase5_eigenfaces.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()
