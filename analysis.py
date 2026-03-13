"""
analysis.py — Fase 5: Analisi Statistica e PCA.

Scree plot, curve MSE/PSNR, PCA sul dataset, scatter plot, eigenfaces e tabella riassuntiva.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import (CLASSES, CLASS_COLORS, OUTPUT_DIR, TARGET_SIZE,
                    K_RANGE_METRICS, K_VALUES_TABLE, PCA_N_COMPONENTS)
from svd_engine import (apply_svd, reconstruct_svd, compression_ratio,
                        mse, psnr, cumulative_variance, find_k_for_variance)


# ═══════════════════════════════════════════════════════════════════════════════
#  5.1 — SCREE PLOT E VALORI SINGOLARI
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scree(images_by_class: dict[str, np.ndarray], save: bool = True) -> None:
    """Scree plot: varianza cumulativa e decadimento dei valori singolari per classe."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, cls in enumerate(CLASSES):
        img = images_by_class[cls][0]
        _, S, _ = apply_svd(img)
        cum_var = cumulative_variance(S)

        axes[0].plot(cum_var, label=cls, color=CLASS_COLORS[i], linewidth=2)

        k90 = find_k_for_variance(S, 0.90)
        k95 = find_k_for_variance(S, 0.95)
        k99 = find_k_for_variance(S, 0.99)
        print(f"{cls:30s}:  k(90%)={k90:3d}   k(95%)={k95:3d}   k(99%)={k99:3d}  "
              f"su {len(S)} totali")

        axes[1].plot(S[:50], label=cls, color=CLASS_COLORS[i], linewidth=2,
                     marker='o', markersize=3)

    axes[0].axhline(y=0.90, color='gray', linestyle='--', alpha=0.6, label='90%')
    axes[0].axhline(y=0.95, color='gray', linestyle=':', alpha=0.6, label='95%')
    axes[0].set_xlabel('Numero di componenti (k)')
    axes[0].set_ylabel('Varianza cumulativa spiegata')
    axes[0].set_title('Scree Plot — Varianza Cumulativa', fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(0, 100)

    axes[1].set_xlabel('Indice i')
    axes[1].set_ylabel('Valore singolare S(i)')
    axes[1].set_title('Decadimento dei Valori Singolari (primi 50)', fontweight='bold')
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase5_scree_plot.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  5.2 — CURVE MSE / PSNR vs k
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mse_psnr(images_by_class: dict[str, np.ndarray], save: bool = True) -> None:
    """Grafici MSE e PSNR in funzione del numero di componenti k."""
    k_range = np.array(list(K_RANGE_METRICS))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, cls in enumerate(CLASSES):
        img = images_by_class[cls][0]
        U, S, Vt = apply_svd(img)
        mse_vals, psnr_vals = [], []
        for k in k_range:
            recon = np.clip(reconstruct_svd(U, S, Vt, k), 0, 1)
            mse_vals.append(mse(img, recon))
            psnr_vals.append(psnr(img, recon))

        axes[0].plot(k_range, mse_vals, label=cls, color=CLASS_COLORS[i], linewidth=2)
        axes[1].plot(k_range, psnr_vals, label=cls, color=CLASS_COLORS[i], linewidth=2)

    axes[0].set_xlabel('Numero di componenti (k)')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE vs k — Errore di Ricostruzione', fontweight='bold')
    axes[0].legend()

    axes[1].set_xlabel('Numero di componenti (k)')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('PSNR vs k — Qualità di Ricostruzione', fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase5_mse_psnr.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  5.3 — PCA SUL DATASET
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


# ═══════════════════════════════════════════════════════════════════════════════
#  5.4 — TABELLA RIASSUNTIVA
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(image: np.ndarray, class_name: str = "Normal") -> None:
    """Stampa la tabella riassuntiva del trade-off compressione / qualità."""
    U, S, Vt = apply_svd(image)
    m, n = image.shape
    cum_var = cumulative_variance(S)

    print("\n" + "=" * 80)
    print(f"TABELLA RIASSUNTIVA — TRADE-OFF COMPRESSIONE vs QUALITÀ  ({class_name})")
    print("=" * 80)
    print(f"\n{'k':>5} | {'Rapporto Compr.':>16} | {'Varianza':>10} | "
          f"{'MSE':>12} | {'PSNR (dB)':>10}")
    print("-" * 65)

    for k in K_VALUES_TABLE:
        recon = np.clip(reconstruct_svd(U, S, Vt, k), 0, 1)
        cr = compression_ratio(m, n, k)
        err = mse(image, recon)
        p = psnr(image, recon)
        var_k = cum_var[k - 1] if k <= len(cum_var) else 1.0
        print(f"{k:5d} | {cr:15.2%} | {var_k:9.2%} | {err:12.8f} | {p:10.2f}")
