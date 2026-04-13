"""
visualization.py -- Fase 4: Ricostruzione e Visualizzazione.

Funzioni per la visualizzazione delle immagini originali, delle ricostruzioni SVD
e per il confronto tra classi diagnostiche.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from .config import CLASSES, CLASS_COLORS, OUTPUT_DIR, K_VALUES_DEMO, K_VALUES_COMPARE, TARGET_SIZE
from .svd_engine import apply_svd, reconstruct_svd, compression_ratio, mse, psnr
from .data_loader import detect_tilt_angle, correct_tilt, is_lateral_xray


def plot_exploration(sample_images: dict[str, np.ndarray], save: bool = True) -> None:
    """
    Fase 1 — Mostra un campione per ciascuna classe con il relativo istogramma dei pixel.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for i, cls in enumerate(CLASSES):
        axes[0, i].imshow(sample_images[cls], cmap='gray')
        axes[0, i].set_title(cls, fontweight='bold')
        axes[0, i].axis('off')

        axes[1, i].hist(sample_images[cls].ravel(), bins=64, color='steelblue',
                        edgecolor='white', alpha=0.85, density=True)
        axes[1, i].set_xlabel('Intensità pixel')
        axes[1, i].set_ylabel('Densità')
        axes[1, i].set_title(f'Istogramma — {cls}')

    plt.suptitle('Fase 1 — Campioni e Distribuzione dei Pixel',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase1_esplorazione.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


def plot_svd_reconstruction(
    image: np.ndarray,
    class_name: str = "Normal",
    k_values: list[int] = None,
    save: bool = True
) -> None:
    """
    Fase 4 — Griglia di ricostruzioni SVD a diversi valori di k con metriche.
    """
    if k_values is None:
        k_values = K_VALUES_DEMO

    U, S, Vt = apply_svd(image)
    m, n = image.shape

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    # Originale
    axes[0].imshow(image, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f'Originale\n{m}×{n} = {m*n:,} valori', fontweight='bold')
    axes[0].axis('off')

    print(f"{'k':>5}  {'Compressione':>14}  {'MSE':>10}  {'PSNR (dB)':>10}")
    print("-" * 48)

    for idx, k in enumerate(k_values):
        recon = np.clip(reconstruct_svd(U, S, Vt, k), 0, 1)
        cr = compression_ratio(m, n, k)
        err = mse(image, recon)
        p = psnr(image, recon)

        print(f"{k:5d}  {cr:13.2%}  {err:10.6f}  {p:10.2f}")

        axes[idx + 1].imshow(recon, cmap='gray', vmin=0, vmax=1)
        axes[idx + 1].set_title(f'k = {k}\nCompr. {cr:.1%} | PSNR {p:.1f} dB',
                                fontweight='bold')
        axes[idx + 1].axis('off')

    plt.suptitle(f'Fase 4 — Ricostruzione SVD a diversi livelli di compressione '
                 f'(Classe: {class_name})', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase4_ricostruzione_svd.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


def plot_class_comparison(
    images_by_class: dict[str, np.ndarray],
    k_values: list[int] = None,
    save: bool = True
) -> None:
    """
    Fase 4 — Confronto della ricostruzione SVD tra le 4 classi diagnostiche.
    """
    if k_values is None:
        k_values = K_VALUES_COMPARE

    fig, axes = plt.subplots(4, len(k_values) + 1, figsize=(22, 18))

    for row, cls in enumerate(CLASSES):
        img = images_by_class[cls][0]
        U, S, Vt = apply_svd(img)

        axes[row, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_title('Originale', fontweight='bold')
        axes[row, 0].set_ylabel(cls, fontsize=13, fontweight='bold',
                                rotation=90, labelpad=15)
        # Non usare axis('off') perché nasconde anche la ylabel.
        # Rimuoviamo solo i tick e le spine per mantenere il nome della classe.
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)

        for col, k in enumerate(k_values):
            recon = np.clip(reconstruct_svd(U, S, Vt, k), 0, 1)
            p = psnr(img, recon)
            axes[row, col + 1].imshow(recon, cmap='gray', vmin=0, vmax=1)
            axes[row, col + 1].set_title(f'k={k} | PSNR={p:.1f}', fontweight='bold')
            axes[row, col + 1].axis('off')

    plt.suptitle('Confronto SVD tra le 4 classi diagnostiche',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase4_confronto_classi.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


def plot_tilt_correction(
    raw_samples: dict[str, list[tuple[np.ndarray, str]]],
    save: bool = True
) -> None:
    """
    Mostra il prima/dopo della correzione dell'inclinazione.
    Riceve immagini RAW (senza correzione tilt applicata) per mostrare il
    vero confronto.
    """
    n_samples = max(len(v) for v in raw_samples.values())
    fig, axes = plt.subplots(len(CLASSES), n_samples * 2,
                             figsize=(4 * n_samples * 2, 4 * len(CLASSES)))

    for row, cls in enumerate(CLASSES):
        samples = raw_samples[cls]
        for col in range(min(n_samples, len(samples))):
            raw_img, fname = samples[col]

            # Rileva l'angolo sull'immagine raw
            angle = detect_tilt_angle(raw_img)
            corrected, applied_angle = correct_tilt(raw_img.copy(), angle)
            corrected = np.clip(corrected, 0, 1)

            # Prima (raw)
            ax_before = axes[row, col * 2]
            ax_before.imshow(raw_img, cmap='gray', vmin=0, vmax=1)
            ax_before.set_title(f'Prima\nAngolo: {angle:.1f}°', fontweight='bold', fontsize=10)
            ax_before.axis('off')
            if col == 0:
                ax_before.set_ylabel(cls, fontsize=12, fontweight='bold', rotation=90, labelpad=15)

            # Dopo (corretto)
            ax_after = axes[row, col * 2 + 1]
            ax_after.imshow(corrected, cmap='gray', vmin=0, vmax=1)
            ax_after.set_title(f'Dopo\nCorretto: {applied_angle:.1f}°', fontweight='bold', fontsize=10)
            ax_after.axis('off')

    plt.suptitle('Correzione Inclinazione -- Prima vs Dopo',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase2_correzione_tilt.png'),
                    dpi=150, bbox_inches='tight')
    plt.close()
