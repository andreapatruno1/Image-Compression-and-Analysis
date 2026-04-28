"""
classification.py -- Fase 6: Classificazione e Confronto Feature.

Confronto SVD (compressione rank-k) vs PCA (riduzione dimensionale):
  Baseline:      Raw Pixels (65 536 feature)
  Strategia A:   SVD — ricostruzione A_k = U_k @ diag(S_k) @ Vk.T
                 Il classificatore riceve i 65.536 pixel ricostruiti.
                 Domanda: "La compressione rank-k è diagnosticamente lossless?"
  Strategia B:   PCA — k coordinate principali (proiezione sul dataset)
                 Il classificatore riceve k feature compatte.
                 Domanda: "Bastano k componenti per classificare?"

Le due strategie rispondono a domande diverse e non sono confrontabili
in termini di dimensionalità delle feature (65.536 vs k).

Usa sklearn Pipeline per evitare data leakage: StandardScaler e PCA vengono
fittati SOLO sul training set di ogni fold.

Metriche: accuracy, precision, recall, F1 (macro), AUC-ROC (OvR).
"""

import os
from typing import Callable, Optional
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from .config import (CLASSES, CLASS_COLORS, OUTPUT_DIR,
                     KNN_N_NEIGHBORS, CV_N_FOLDS, RANDOM_STATE,
                     SVD_K_FEATURES, SVD_K_VALUES, PCA_COMPONENTS_LIST,
                     LR_MAX_ITER, LR_C)
from .svd_engine import apply_svd, reconstruct_svd


# ============================================================================
#  HELPERS
# ============================================================================

def _knn_pipeline(pca_n=None):
    """Crea una pipeline StandardScaler -> [PCA] -> KNN."""
    steps = [('scaler', StandardScaler())]
    if pca_n is not None:
        steps.append(('pca', PCA(n_components=pca_n)))
    steps.append(('knn', KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS)))
    return Pipeline(steps)


def _lr_pipeline(pca_n=None):
    """Crea una pipeline StandardScaler -> [PCA] -> Logistic Regression.

    Multinomiale con solver L-BFGS, regolarizzazione C e max_iter da config.
    """
    steps = [('scaler', StandardScaler())]
    if pca_n is not None:
        steps.append(('pca', PCA(n_components=pca_n)))
    steps.append(('lr', LogisticRegression(
        max_iter=LR_MAX_ITER, C=LR_C,
        solver='lbfgs', random_state=RANDOM_STATE)))
    return Pipeline(steps)


def _get_pipeline_fn(classifier: str):
    """Restituisce la funzione factory della pipeline per il classificatore."""
    if classifier == 'lr':
        return _lr_pipeline
    return _knn_pipeline


def _classifier_label(classifier: str) -> str:
    """Restituisce un'etichetta leggibile per il classificatore."""
    if classifier == 'lr':
        return "Logistic Regression"
    return f"KNN (k={KNN_N_NEIGHBORS})"


def _svd_reconstruct_batch(all_images, k):
    """Ricostruisce tutte le immagini con SVD troncata a rango k (Eckart-Young).

    Per ogni immagine A: A_k = U_k @ diag(S_k) @ Vk.T
    Output shape: (N, 256*256) — pixel ricostruiti appiattiti.

    Strategia A della classificazione: il classificatore riceve i pixel
    ricostruiti, non feature SVD estratte. Questo testa se la compressione
    rank-k è diagnosticamente lossless.
    """
    N = all_images.shape[0]
    svd_images = []
    for i in range(N):
        U, S, Vt = apply_svd(all_images[i])
        recon = np.clip(reconstruct_svd(U, S, Vt, k), 0, 1)
        svd_images.append(recon)
    return np.array(svd_images).reshape(N, -1)


# ============================================================================
#  6.1 -- PREPARAZIONE SCENARI
# ============================================================================

def prepare_feature_sets(
    all_images: np.ndarray,
    labels: np.ndarray,
    classifier: str = 'knn',
    svd_cache: dict = None,
) -> dict[str, tuple[np.ndarray, Pipeline]]:
    """
    Prepara gli scenari per il confronto SVD vs PCA con analisi del costo.

    Baseline:    Raw Pixels (N, H*W)

    Strategia A — SVD spaziale (U·S·V^T):
        Per ogni immagine estrae sigma_i * u_i e sigma_i * v_i per i=1..k.
        Output: (N, 2*H*k) — cattura informazione spaziale locale.
        Costo: 2·256·k feature per classificatore (cresce linearmente con k).

    Strategia B — PCA globale:
        Proietta il dataset sulle prime k componenti principali.
        Output: (N, k) — descrittore globale inter-immagine.
        Costo: k feature per classificatore (molto più compatto).

    Differenza di costo esplicitata nel log di preparazione:
        Stessa k strutturale (rango della decomposizione), dimensionalità
        finale diversa — parte integrante del confronto.

    Parameters
    ----------
    all_images : array (N, H, W)
    labels     : etichette
    classifier : 'knn' o 'lr'
    svd_cache  : dict per cachare le feature SVD tra i due classificatori

    Returns
    -------
    scenarios : dict {nome: (X_data, pipeline)}
    """
    pipe_fn = _get_pipeline_fn(classifier)
    clf_label = _classifier_label(classifier)

    N, H, W = all_images.shape
    n_pixels = H * W
    X_flat = all_images.reshape(N, -1)

    scenarios = {}

    # --- Baseline: Raw Pixels ---
    scenarios["Raw Pixels"] = (X_flat, pipe_fn())
    print(f"  [Baseline]  Raw Pixels        {n_pixels:>6} feat  ({clf_label})")

    # --- Strategia A: SVD — ricostruzione di rango k ---
    # Filosofia: test della capacita' di compressione diagnostically-lossless.
    # Per ogni immagine A: A_k = U[:,:k] * diag(S[:k]) * Vt[:k,:]
    # Storage per immagine: (2*H + 1) * k scalari vs H*W originali.
    # Risultato: immagini ricostruite di shape (H, W), flatten a H*W feature.
    # Domanda scientifica: "Se comprimo con SVD a rango k, perdo informazione
    # diagnostica?" — se l'accuracy si mantiene -> compressione lossless per
    # il task di classificazione.
    print(f"\n  -- Strategia A: SVD ricostruzione rango-k (immagine compressa) --")
    for k in SVD_K_FEATURES:
        cache_key = f"svd_rec_{k}"
        # Calcolo percentuale di storage: (2*H + 1)*k su H*W
        storage_pct = (2 * H + 1) * k / (H * W) * 100
        if svd_cache is not None and cache_key in svd_cache:
            X_svd = svd_cache[cache_key]
            print(f"  [SVD k={k:2d}]  ricostr. ({storage_pct:.1f}% storage) -> {n_pixels:>6} feat  (cache)  ({clf_label})")
        else:
            print(f"  [SVD k={k:2d}]  ricostr. ({storage_pct:.1f}% storage) -> {n_pixels:>6} feat  ricostruzione...",
                  end="", flush=True)
            X_svd = _svd_reconstruct_batch(all_images, k)
            print(" OK")
            if svd_cache is not None:
                svd_cache[cache_key] = X_svd

        # Pipeline: Scaler -> classificatore (niente PCA, immagine intera)
        name = f"SVD k={k}"
        scenarios[name] = (X_svd, pipe_fn(pca_n=None))

    # --- Strategia B: PCA globale — k coordinate ---
    print(f"\n  -- Strategia B: PCA globale (pixel -> PCA(k)) --")
    for n_comp in PCA_COMPONENTS_LIST:
        name = f"PCA k={n_comp}"
        scenarios[name] = (X_flat, pipe_fn(pca_n=n_comp))
        print(f"  [PCA k={n_comp:2d}]  65536 feat -> PCA({n_comp}) -> {n_comp} feat  "
              f"(Pipeline: Scaler->PCA({n_comp})->{clf_label})")

    print(f"\n  Riepilogo strategie (stesso classificatore, feature diverse):")
    print(f"  {'k':>4} | {'SVD: storage compr.':>22} | {'PCA: dim finale':>18} | {'feat al classificatore':>22}")
    print(f"  {'-'*74}")
    for k in SVD_K_FEATURES:
        storage_pct = (2 * H + 1) * k / (H * W) * 100
        print(f"  {k:>4} | {storage_pct:>18.1f}% pixel | {k:>14} dim  | {n_pixels:>16,} feat (SVD) / {k} (PCA)")

    return scenarios


# ============================================================================
#  6.2 -- CLASSIFICAZIONE
# ============================================================================

def run_classification(
    scenarios: dict[str, tuple[np.ndarray, Pipeline]],
    labels: np.ndarray,
    classifier: str = 'knn'
) -> dict:
    """
    Esegue classificazione con Stratified K-Fold CV per tutti gli scenari.

    Returns
    -------
    dict : {scenario_name: {"accuracy": (mean, std), ...}}
    """
    from sklearn.base import clone

    cv = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    clf_label = _classifier_label(classifier)
    results = {}

    for scenario_name, (X, pipe) in scenarios.items():
        accs, precs, recs, f1s = [], [], [], []

        for train_idx, test_idx in cv.split(X, labels):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            pipe_fold = clone(pipe)
            pipe_fold.fit(X_train, y_train)
            y_pred = pipe_fold.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred, average='macro',
                                         zero_division=0))
            recs.append(recall_score(y_test, y_pred, average='macro',
                                      zero_division=0))
            f1s.append(f1_score(y_test, y_pred, average='macro',
                                zero_division=0))

        results[scenario_name] = {
            "accuracy":  (np.mean(accs),  np.std(accs)),
            "precision": (np.mean(precs), np.std(precs)),
            "recall":    (np.mean(recs),  np.std(recs)),
            "f1":        (np.mean(f1s),   np.std(f1s)),
        }

    # --- Stampa tabella formattata ---
    def _fmt(mean, std):
        return f"{mean*100:5.1f}%  (+/- {std*100:4.1f}%)"

    print("\n" + "=" * 100)
    print(f"  RISULTATI CLASSIFICAZIONE -- {clf_label}"
          f" -- Stratified {CV_N_FOLDS}-Fold CV")
    print("=" * 100)

    header = (f"\n  {'Scenario':<26} | {'Accuracy':^18} | {'Precision':^18} |"
              f" {'Recall':^18} | {'F1-Score':^18}")
    print(header)
    print("  " + "-" * 96)

    for name, m in results.items():
        row = (f"  {name:<26} | "
               f"{_fmt(m['accuracy'][0],  m['accuracy'][1])}  | "
               f"{_fmt(m['precision'][0], m['precision'][1])}  | "
               f"{_fmt(m['recall'][0],    m['recall'][1])}  | "
               f"{_fmt(m['f1'][0],        m['f1'][1])}")
        print(row)

    print("  " + "-" * 96)

    best = max(results, key=lambda s: results[s]["accuracy"][0])
    best_acc = results[best]["accuracy"][0]
    print("=" * 100)

    return results


# ============================================================================
#  6.3 -- CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(
    scenarios: dict[str, tuple[np.ndarray, Pipeline]],
    labels: np.ndarray,
    classifier: str = 'knn',
    save: bool = True,
    classes: Optional[list] = None,
    short_labels: Optional[list] = None,
    output_dir: Optional[str] = None
) -> None:
    """Confusion matrix per ogni scenario."""
    if classes is None:
        classes = CLASSES
    if short_labels is None:
        # Fallback retrocompatibile con il dataset medico originale
        if list(classes) == list(CLASSES):
            short_labels = ["COVID", "Normal", "Pneum.", "TB"]
        else:
            short_labels = list(classes)
    if output_dir is None:
        output_dir = OUTPUT_DIR

    clf_label = _classifier_label(classifier)
    suffix = f"_{classifier}" if classifier != 'knn' else ""

    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    cv = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    for idx, (sc_name, (X, pipe)) in enumerate(scenarios.items()):
        y_pred = cross_val_predict(pipe, X, labels, cv=cv)
        cm = confusion_matrix(labels, y_pred, labels=classes)

        im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[idx].set_title(sc_name, fontweight='bold', fontsize=10)
        axes[idx].set_xlabel('Predetto')
        axes[idx].set_ylabel('Reale')
        axes[idx].set_xticks(range(len(classes)))
        axes[idx].set_yticks(range(len(classes)))
        axes[idx].set_xticklabels(short_labels, fontsize=8)
        axes[idx].set_yticklabels(short_labels, fontsize=8)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center", fontsize=11,
                               fontweight='bold',
                               color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle(f'Confusion Matrix -- {clf_label}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'fase5_confusion_matrices{suffix}.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
#  6.4 -- CURVE ROC MULTICLASSE
# ============================================================================

def plot_roc_curves(
    scenarios: dict[str, tuple[np.ndarray, Pipeline]],
    labels: np.ndarray,
    classifier: str = 'knn',
    save: bool = True,
    classes: Optional[list] = None,
    class_colors: Optional[list] = None,
    short_labels: Optional[list] = None,
    output_dir: Optional[str] = None
) -> None:
    """Curve ROC One-vs-Rest per ogni scenario.

    Supporta sia problemi multiclasse che binari. Nel caso binario,
    LabelBinarizer restituisce shape (N, 1): la funzione espande a (N, 2)
    per mantenere coerente il ciclo sulle classi.
    """
    from sklearn.base import clone

    if classes is None:
        classes = CLASSES
    if class_colors is None:
        class_colors = CLASS_COLORS
    if short_labels is None:
        if list(classes) == list(CLASSES):
            short_labels = ["COVID", "Normal", "Pneum.", "TB"]
        else:
            short_labels = list(classes)
    if output_dir is None:
        output_dir = OUTPUT_DIR

    clf_label = _classifier_label(classifier)
    suffix = f"_{classifier}" if classifier != 'knn' else ""

    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    cv = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(labels)

    # --- Caso binario: LabelBinarizer produce (N,1), espandi a (N,2) ---
    is_binary = y_bin.ndim == 2 and y_bin.shape[1] == 1
    if is_binary:
        y_bin = np.hstack([1 - y_bin, y_bin])
        # lb.classes_ è già l'elenco delle 2 classi originali ([neg, pos])

    for idx, (sc_name, (X, pipe)) in enumerate(scenarios.items()):
        y_prob = np.zeros_like(y_bin, dtype=np.float64)

        for train_idx, test_idx in cv.split(X, labels):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = labels[train_idx]

            pipe_fold = clone(pipe)
            pipe_fold.fit(X_train, y_train)
            proba = pipe_fold.predict_proba(X_test)

            # Rileva automaticamente il classificatore (ultimo step della pipeline)
            last_step_name = pipe_fold.steps[-1][0]
            clf_classes = pipe_fold.named_steps[last_step_name].classes_

            for ci, cls_name in enumerate(lb.classes_):
                if cls_name in clf_classes:
                    src_col = list(clf_classes).index(cls_name)
                    y_prob[test_idx, ci] = proba[:, src_col]

        macro_auc = []
        for ci in range(len(lb.classes_)):
            fpr, tpr, _ = roc_curve(y_bin[:, ci], y_prob[:, ci])
            roc_auc = auc(fpr, tpr)
            macro_auc.append(roc_auc)
            axes[idx].plot(fpr, tpr, color=class_colors[ci], linewidth=2,
                           label=f'{short_labels[ci]} (AUC={roc_auc:.2f})')

        axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
        mean_auc = np.mean(macro_auc)
        axes[idx].set_title(f'{sc_name}\nMacro AUC = {mean_auc:.3f}',
                            fontweight='bold', fontsize=10)
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
        axes[idx].legend(fontsize=7, loc='lower right')
        axes[idx].set_xlim([-0.02, 1.02])
        axes[idx].set_ylim([-0.02, 1.02])

    plt.suptitle(f'Curve ROC -- {clf_label} -- One-vs-Rest',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, f'fase5_roc_curves{suffix}.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
#  6.5 -- CONFRONTO METRICHE TRA SCENARI
# ============================================================================

def plot_classification_comparison(
    results: dict,
    classifier: str = 'knn',
    save: bool = True,
    output_dir: Optional[str] = None
) -> None:
    """Grafico a barre raggruppate: confronto metriche tra scenari."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    clf_label = _classifier_label(classifier)
    suffix = f"_{classifier}" if classifier != 'knn' else ""

    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    scenario_names = list(results.keys())
    n_scenarios = len(scenario_names)

    # Palette dinamica per qualsiasi numero di scenari
    import matplotlib.cm as cm
    if n_scenarios <= 10:
        base = ['#e74c3c', '#ff7f50', '#2ecc71', '#27ae60', '#1abc9c',
                '#e67e22', '#d35400', '#3498db', '#2980b9', '#9b59b6']
        colors = base[:n_scenarios]
    else:
        colors = [cm.tab20(i / n_scenarios) for i in range(n_scenarios)]

    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(metrics))
    bar_width = 0.80 / n_scenarios
    offsets = np.linspace(-(n_scenarios - 1) / 2 * bar_width,
                           (n_scenarios - 1) / 2 * bar_width, n_scenarios)

    for i, scenario in enumerate(scenario_names):
        means = [results[scenario][m][0] for m in metrics]
        stds  = [results[scenario][m][1] for m in metrics]

        bars = ax.bar(x + offsets[i], means, bar_width,
                      yerr=stds, capsize=3,
                      label=scenario, color=colors[i],
                      edgecolor='white', linewidth=0.8, alpha=0.9)

        # Etichette compatte sopra le error bar
        for bar, mean_val, std_val in zip(bars, means, stds):
            y_pos = mean_val + std_val + 0.008 + (i % 2) * 0.012
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{mean_val*100:.0f}%',
                    ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold', color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Confronto Metriche -- {clf_label} -- '
                 f'Stratified {CV_N_FOLDS}-Fold CV',
                 fontsize=14, fontweight='bold')

    # Y-limit dinamico basato sui dati
    all_lower = [results[s][m][0] - results[s][m][1]
                 for s in scenario_names for m in metrics]
    y_min = max(0.50, min(all_lower) - 0.05)
    ax.set_ylim(y_min, 1.02)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir,
                    f'fase5_classification_comparison{suffix}.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
#  6.6 -- HERO CHART: DUE STRATEGIE A CONFRONTO
# ============================================================================

def plot_hero_tradeoff(
    results: dict,
    classifier: str = 'knn',
    save: bool = True,
    output_dir: Optional[str] = None
) -> None:
    """Grafico 'eroe' — confronto simmetrico SVD vs PCA su asse k comune.

    Un singolo pannello con due curve sovrapposte:
      - Arancione: SVD (k valori singolari) vs k
      - Verde:     PCA (k coordinate)       vs k
      - Rosso tratteggiato: baseline Raw Pixels

    L'asse x è lo stesso per entrambe → confronto mele con mele.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    clf_label = _classifier_label(classifier)
    suffix = f"_{classifier}" if classifier != 'knn' else ""

    # ── Estrai baseline ──
    raw_key = next((k for k in results if k == "Raw Pixels"), None)
    if not raw_key:
        return
    raw_acc, raw_std = results[raw_key]["accuracy"]

    import re as _re

    # ── SVD feature points ──
    svd_points = []
    for name, met in results.items():
        if name.startswith("SVD k="):
            m = _re.search(r'k=(\d+)', name)
            if m:
                k = int(m.group(1))
                svd_points.append((k, met["accuracy"][0], met["accuracy"][1]))
    svd_points.sort()

    # ── PCA points ──
    pca_points = []
    for name, met in results.items():
        if name.startswith("PCA k="):
            m = _re.search(r'k=(\d+)', name)
            if m:
                k = int(m.group(1))
                pca_points.append((k, met["accuracy"][0], met["accuracy"][1]))
    pca_points.sort()

    if not svd_points and not pca_points:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Baseline
    ax.axhspan(raw_acc - raw_std, raw_acc + raw_std,
               color='#e74c3c', alpha=0.08)
    ax.axhline(raw_acc, color='#e74c3c', ls='--', alpha=0.7, lw=2,
               label=f'Baseline: Raw Pixels ({raw_acc*100:.1f}%)')

    color_svd = '#e67e22'
    color_pca = '#27ae60'

    # Curva SVD
    if svd_points:
        ks  = [p[0] for p in svd_points]
        acc = [p[1] for p in svd_points]
        std = [p[2] for p in svd_points]
        ax.errorbar(ks, acc, yerr=std, fmt='o-', color=color_svd,
                    markersize=12, mec='white', mew=2,
                    capsize=4, elinewidth=1.5, lw=2.5,
                    label='SVD — σᵢuᵢ|σᵢvᵢ → PCA(k) → k feat', zorder=5)
        for k, a, s in svd_points:
            ax.text(k, a + s + 0.007,
                    f'{a*100:.1f}%\n({2*256*k:,}→{k})',
                    ha='center', va='bottom', fontsize=8.5,
                    fontweight='bold', color=color_svd)

    # Curva PCA
    if pca_points:
        ks  = [p[0] for p in pca_points]
        acc = [p[1] for p in pca_points]
        std = [p[2] for p in pca_points]
        ax.errorbar(ks, acc, yerr=std, fmt='D-', color=color_pca,
                    markersize=12, mec='white', mew=2,
                    capsize=4, elinewidth=1.5, lw=2.5,
                    label='PCA — pixel → PCA(k) → k feat', zorder=5)
        for k, a, s in pca_points:
            ax.text(k, a - s - 0.018,
                    f'{a*100:.1f}%\n(65536→{k})',
                    ha='center', va='top', fontsize=8.5,
                    fontweight='bold', color=color_pca)

    # Annotazione take-away
    ax.text(0.03, 0.05,
            'Entrambi arrivano al classificatore con k feature finali.\n'
            'SVD: spazio intermedio 2·256·k  |  PCA: spazio intermedio 65.536',
            transform=ax.transAxes, fontsize=9.5, va='bottom',
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.4', fc='#f8f8f8',
                      ec='#888888', alpha=0.9))

    all_k = sorted(set([p[0] for p in svd_points] + [p[0] for p in pca_points]))
    ax.set_xticks(all_k)
    ax.set_xlabel('Numero di feature k', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (5-Fold CV)', fontsize=13, fontweight='bold')
    ax.set_title(f'SVD Spaziale vs PCA — {clf_label}\n'
                 f'Entrambi con k feature finali (spazi intermedi diversi)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    all_accs = ([p[1] for p in svd_points] + [p[1] for p in pca_points]
                + [raw_acc])
    all_stds = ([p[2] for p in svd_points] + [p[2] for p in pca_points]
                + [raw_std])
    y_min = min(a - s for a, s in zip(all_accs, all_stds)) - 0.04
    y_max = max(a + s for a, s in zip(all_accs, all_stds)) + 0.04
    ax.set_ylim(max(0.50, y_min), min(1.0, y_max))
    if all_k:
        ax.set_xlim(min(all_k) - 3, max(all_k) + 5)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir,
                    f'fase5_hero_tradeoff{suffix}.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
#  6.7 -- CONFRONTO DIRETTO KNN vs LOGISTIC REGRESSION
# ============================================================================

def plot_knn_vs_lr(
    results_knn: dict,
    results_lr: dict,
    save: bool = True,
    output_dir: Optional[str] = None
) -> None:
    """Confronto diretto KNN vs Logistic Regression su tutti gli scenari.

    Genera un grafico 2×2 (Accuracy, Precision, Recall, F1) con barre
    affiancate KNN/LR per ogni scenario.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Scenari comuni ai due classificatori
    common_scenarios = [s for s in results_knn if s in results_lr]
    n = len(common_scenarios)

    # Abbreviazioni per leggibilità sull'asse X
    import re as _re
    short_names = []
    for s in common_scenarios:
        if s.startswith("Raw"):
            short_names.append("Raw Pixels")
        elif s.startswith("PCA k="):
            m = _re.search(r'k=(\d+)', s)
            short_names.append(f"PCA k={m.group(1)}" if m else s[:14])
        elif s.startswith("SVD k="):
            m = _re.search(r'k=(\d+)', s)
            short_names.append(f"SVD k={m.group(1)}" if m else s[:14])
        else:
            short_names.append(s[:14])

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.ravel()

    color_knn = '#3498db'
    color_lr = '#e74c3c'

    for m_idx, (metric, m_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[m_idx]
        x = np.arange(n)
        width = 0.35

        means_knn = [results_knn[s][metric][0] for s in common_scenarios]
        stds_knn  = [results_knn[s][metric][1] for s in common_scenarios]
        means_lr  = [results_lr[s][metric][0]  for s in common_scenarios]
        stds_lr   = [results_lr[s][metric][1]  for s in common_scenarios]

        bars1 = ax.bar(x - width/2, means_knn, width, yerr=stds_knn,
                       capsize=3, label=f'KNN (k={KNN_N_NEIGHBORS})',
                       color=color_knn, edgecolor='white', linewidth=0.8,
                       alpha=0.85)
        bars2 = ax.bar(x + width/2, means_lr, width, yerr=stds_lr,
                       capsize=3, label='Logistic Regression',
                       color=color_lr, edgecolor='white', linewidth=0.8,
                       alpha=0.85)

        # Etichette sopra le barre
        for bar, mean_val in zip(bars1, means_knn):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f'{mean_val*100:.1f}%', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color=color_knn)
        for bar, mean_val in zip(bars2, means_lr):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.008,
                    f'{mean_val*100:.1f}%', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color=color_lr)

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=9, rotation=30, ha='right')
        ax.set_ylabel(m_label, fontsize=11)
        ax.set_title(m_label, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Y-limit dinamico
        all_lower = means_knn + means_lr
        all_stds = stds_knn + stds_lr
        y_min = min(v - s for v, s in zip(all_lower, all_stds))
        ax.set_ylim(max(0.50, y_min - 0.05), 1.02)

    fig.suptitle(f'Confronto KNN vs Logistic Regression -- '
                 f'Stratified {CV_N_FOLDS}-Fold CV',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_dir, 'fase5_knn_vs_lr.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()
