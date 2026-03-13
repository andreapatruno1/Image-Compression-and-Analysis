"""
svd_engine.py — Fase 3: Il Motore Matematico (SVD).

Singular Value Decomposition: decomposizione, ricostruzione troncata e metriche.

Teoria
------
Una matrice A (m × n) viene scomposta come:

    A = U · Σ · Vᵀ

dove:
  - U  (m × r) : vettori singolari sinistri
  - Σ  (r × r) : valori singolari σ₁ ≥ σ₂ ≥ … ≥ 0
  - Vᵀ (r × n) : vettori singolari destri
  - r = min(m, n)

L'approssimazione di rango k (teorema di Eckart–Young) è:

    Aₖ = Σᵢ₌₁ᵏ σᵢ · uᵢ · vᵢᵀ
"""

import numpy as np


def apply_svd(image_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applica la SVD (economy-size) a una matrice immagine.

    Parameters
    ----------
    image_matrix : np.ndarray di shape (m, n)

    Returns
    -------
    U  : np.ndarray (m, r)
    S  : np.ndarray (r,)   — vettore dei valori singolari
    Vt : np.ndarray (r, n)
    """
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    return U, S, Vt


def reconstruct_svd(U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
    """
    Ricostruisce l'immagine usando le prime k componenti singolari.

    Parameters
    ----------
    U, S, Vt : output di apply_svd
    k        : numero di componenti da mantenere

    Returns
    -------
    np.ndarray : immagine ricostruita (m × n)
    """
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def compression_ratio(m: int, n: int, k: int) -> float:
    """
    Rapporto di compressione: dati memorizzati / dati originali.

    Con k componenti servono k*(m + n + 1) valori anziché m*n.
    """
    return k * (m + n + 1) / (m * n)


# ─── Metriche di qualità ──────────────────────────────────────────────────────

def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error tra immagine originale e ricostruita."""
    return float(np.mean((original - reconstructed) ** 2))


def psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).

    Per immagini normalizzate [0, 1], max_val = 1.0.
    """
    err = mse(original, reconstructed)
    if err == 0:
        return float('inf')
    return float(10 * np.log10(max_val ** 2 / err))


def cumulative_variance(S: np.ndarray) -> np.ndarray:
    """
    Varianza cumulativa spiegata dai valori singolari.

    La varianza della i-esima componente è proporzionale a σᵢ².
    """
    variance = S ** 2
    return np.cumsum(variance) / np.sum(variance)


def find_k_for_variance(S: np.ndarray, threshold: float = 0.95) -> int:
    """Trova il minimo k tale che la varianza cumulativa ≥ threshold."""
    cum = cumulative_variance(S)
    return int(np.searchsorted(cum, threshold) + 1)


def print_svd_info(image: np.ndarray, U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> None:
    """Stampa informazioni sulla decomposizione SVD."""
    m, n = image.shape
    print(f"Immagine originale:  {m} × {n} = {m*n:,} valori")
    print(f"Valori singolari:    {len(S)} (min rank(m,n))")
    print(f"\nMatrice U:   {U.shape}")
    print(f"Vettore S:   {S.shape}")
    print(f"Matrice Vt:  {Vt.shape}")
    print(f"\nPrimi 10 valori singolari: {np.round(S[:10], 2)}")
    print(f"Ultimi 5 valori singolari: {np.round(S[-5:], 6)}")
