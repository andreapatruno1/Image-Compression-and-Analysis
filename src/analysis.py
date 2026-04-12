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
        all_cum_vars = []
        all_S = []
        
        for img in images_by_class[cls]:
            _, S, _ = apply_svd(img)
            cum_var = cumulative_variance(S)
            all_cum_vars.append(cum_var)
            all_S.append(S[:50])
            
        mean_cum_var = np.mean(all_cum_vars, axis=0)
        std_cum_var = np.std(all_cum_vars, axis=0)
        mean_S = np.mean(all_S, axis=0)

        x_range = range(len(mean_cum_var))
        axes[0].plot(x_range, mean_cum_var, label=cls, color=CLASS_COLORS[i], linewidth=2)
        axes[0].fill_between(x_range, 
                             mean_cum_var - std_cum_var, 
                             mean_cum_var + std_cum_var, 
                             alpha=0.2, color=CLASS_COLORS[i])

        # Usa la curva media per calcolare k90, k95, k99
        k90 = int(np.argmax(mean_cum_var >= 0.90)) + 1 if np.any(mean_cum_var >= 0.90) else len(mean_cum_var)
        k95 = int(np.argmax(mean_cum_var >= 0.95)) + 1 if np.any(mean_cum_var >= 0.95) else len(mean_cum_var)
        k99 = int(np.argmax(mean_cum_var >= 0.99)) + 1 if np.any(mean_cum_var >= 0.99) else len(mean_cum_var)
        
        print(f"{cls:30s}:  k(90%)={k90:3d}   k(95%)={k95:3d}   k(99%)={k99:3d}  "
              f"su {len(mean_cum_var)} totali (valori medi)")

        axes[1].plot(mean_S, label=cls, color=CLASS_COLORS[i], linewidth=2,
                     marker='o', markersize=3)

    axes[0].axhline(y=0.90, color='gray', linestyle='--', alpha=0.6, label='90%')
    axes[0].axhline(y=0.95, color='gray', linestyle=':', alpha=0.6, label='95%')
    axes[0].axhline(y=0.99, color='gray', linestyle='-.', alpha=0.6, label='99%')
    axes[0].set_xlabel('Numero di componenti (k)')
    axes[0].set_ylabel('Varianza cumulativa spiegata')
    axes[0].set_title('Scree Plot — Varianza Cumulativa (Media ± std)', fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(0, 100)

    axes[1].set_xlabel('Indice i')
    axes[1].set_ylabel('Valore singolare S(i) medio')
    axes[1].set_title('Decadimento dei Valori Singolari (primi 50 medi)', fontweight='bold')
    axes[1].set_yscale('log')
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
    """Grafici MSE e PSNR medi in funzione del numero di componenti k."""
    k_range = np.array(list(K_RANGE_METRICS))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, cls in enumerate(CLASSES):
        all_mse = []
        all_psnr = []
        
        for img in images_by_class[cls]:
            m, n = img.shape
            _, S, _ = apply_svd(img)

            # Calcolo vettorizzato: MSE(k) = sum(S[k:]²) / (m*n)
            S_sq = S ** 2
            total_energy = np.sum(S_sq)
            cumsum_S_sq = np.cumsum(S_sq)
            mse_vals = (total_energy - cumsum_S_sq[k_range - 1]) / (m * n)
            psnr_vals = np.where(
                mse_vals == 0,
                100, # valore cap per evitare inf
                10 * np.log10(1.0 / mse_vals)
            )
            all_mse.append(mse_vals)
            all_psnr.append(psnr_vals)
            
        mean_mse = np.mean(all_mse, axis=0)
        std_mse = np.std(all_mse, axis=0)
        
        mean_psnr = np.mean(all_psnr, axis=0)
        std_psnr = np.std(all_psnr, axis=0)

        axes[0].plot(k_range, mean_mse, label=cls, color=CLASS_COLORS[i], linewidth=2)
        axes[0].fill_between(k_range, mean_mse - std_mse, mean_mse + std_mse, alpha=0.2, color=CLASS_COLORS[i])
        
        axes[1].plot(k_range, mean_psnr, label=cls, color=CLASS_COLORS[i], linewidth=2)
        axes[1].fill_between(k_range, mean_psnr - std_psnr, mean_psnr + std_psnr, alpha=0.2, color=CLASS_COLORS[i])

    axes[0].set_xlabel('Numero di componenti (k)')
    axes[0].set_ylabel('MSE medio')
    axes[0].set_title('MSE vs k — Errore di Ricostruzione (Media ± std)', fontweight='bold')
    axes[0].legend()

    axes[1].set_xlabel('Numero di componenti (k)')
    axes[1].set_ylabel('PSNR medio (dB)')
    axes[1].set_title('PSNR vs k — Qualità di Ricostruzione (Media ± std)', fontweight='bold')
    axes[1].legend()

    # Aggiungere linea di soglia PSNR "accettabile"
    axes[1].axhline(y=30, color='gray', linestyle='--', alpha=0.5)
    axes[1].text(max(k_range)*0.75, 30.5, 'Soglia accettabile (30 dB)', color='gray', fontsize=9)

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
