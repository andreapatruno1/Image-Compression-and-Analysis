"""
pca_engine.py — Fase 5.3: PCA Engine.

Esegue la PCA sul dataset, scatter plot 2D e visualizzazione delle eigenfaces.
"""

import os
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
    save: bool = True
) -> None:
    """Scatter plot PCA 2D + varianza spiegata per componente."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for i, cls in enumerate(CLASSES):
        mask = labels == cls
        axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                        alpha=0.6, s=30, label=cls, color=CLASS_COLORS[i],
                        edgecolors='white', linewidth=0.3)

    axes[0].set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%} varianza)')
    axes[0].set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%} varianza)')
    axes[0].set_title('PCA — Proiezione 2D delle Radiografie', fontweight='bold')
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
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase5_pca_scatter.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  EIGENFACES (EIGEN X-RAYS)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eigenfaces(pca_model: PCA, n_show: int = 10, save: bool = True) -> None:
    """Visualizza le prime n componenti principali come immagini (Eigen X-Rays)."""
    rows = (n_show + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(18, 3.5 * rows))
    axes = axes.ravel()

    for i in range(n_show):
        eigenface = pca_model.components_[i].reshape(TARGET_SIZE)
        axes[i].imshow(eigenface, cmap='RdBu_r')
        axes[i].set_title(f'PC {i+1}\n({pca_model.explained_variance_ratio_[i]:.1%})',
                          fontweight='bold')
        axes[i].axis('off')

    # nascondi assi vuoti
    for j in range(n_show, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Le prime 10 Componenti Principali ("Eigen-Xrays")',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase5_eigenfaces.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()
