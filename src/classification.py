"""
classification.py -- Fase 6: Classificazione e Confronto Feature.

Classificazione KNN su scenari multipli:
  1. Raw Pixels      (65 536 feature)  -- baseline
  2. PCA  50 comp.   (50 feature)      -- riduzione aggressiva
  3. PCA 150 comp.   (150 feature)     -- riduzione moderata
  4. SVD k=10        (65 536 feature)  -- compressione forte
  5. SVD k=50        (65 536 feature)  -- compressione leggera

Usa sklearn Pipeline per evitare data leakage: StandardScaler e PCA vengono
fittati SOLO sul training set di ogni fold.

Metriche: accuracy, precision, recall, F1 (macro), AUC-ROC (OvR).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from config import (CLASSES, CLASS_COLORS, OUTPUT_DIR,
                    KNN_N_NEIGHBORS, CV_N_FOLDS, RANDOM_STATE,
                    SVD_K_VALUES, PCA_COMPONENTS_LIST)
from svd_engine import apply_svd, reconstruct_svd


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


def _svd_reconstruct_batch(all_images, k):
    """Ricostruisce tutte le immagini con SVD troncata a rango k."""
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
    pca_model=None,
    X_pca: np.ndarray = None,
) -> dict[str, tuple[np.ndarray, Pipeline]]:
    """
    Prepara gli scenari per la classificazione.

    Returns
    -------
    scenarios : dict {nome: (X_data, pipeline)}
        Ogni scenario contiene i dati e la pipeline da usare.
    """
    N = all_images.shape[0]
    X_flat = all_images.reshape(N, -1)

    scenarios = {}

    # --- Raw Pixels ---
    scenarios["Raw Pixels (65,536)"] = (X_flat, _knn_pipeline())
    print(f"  [Raw Pixels]       shape: {X_flat.shape}")

    # --- PCA scenari ---
    for n_comp in PCA_COMPONENTS_LIST:
        name = f"PCA ({n_comp} comp.)"
        scenarios[name] = (X_flat, _knn_pipeline(pca_n=n_comp))
        print(f"  [{name}]     (Pipeline: Scaler -> PCA -> KNN)")

    # --- SVD scenari ---
    for k in SVD_K_VALUES:
        print(f"  [SVD k={k}]        ricostruzione...", end="", flush=True)
        X_svd = _svd_reconstruct_batch(all_images, k)
        name = f"SVD k={k} ({X_svd.shape[1]:,})"
        scenarios[name] = (X_svd, _knn_pipeline())
        print(f" shape: {X_svd.shape}")

    return scenarios


# ============================================================================
#  6.2 -- CLASSIFICAZIONE
# ============================================================================

def run_classification(
    scenarios: dict[str, tuple[np.ndarray, Pipeline]],
    labels: np.ndarray
) -> dict:
    """
    Esegue KNN con Stratified K-Fold CV per tutti gli scenari.
    Ogni Pipeline viene clonata per ogni fold per evitare data leakage.

    Returns
    -------
    dict : {scenario_name: {"accuracy": (mean, std), ...}}
    """
    from sklearn.base import clone

    cv = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
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
    print(f"  RISULTATI CLASSIFICAZIONE -- KNN (k={KNN_N_NEIGHBORS})"
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
    print(f"\n  >> Miglior scenario: {best}  (Accuracy = {best_acc*100:.1f}%)")
    print("=" * 100)

    return results


# ============================================================================
#  6.3 -- CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(
    scenarios: dict[str, tuple[np.ndarray, Pipeline]],
    labels: np.ndarray,
    save: bool = True
) -> None:
    """Confusion matrix per ogni scenario."""
    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    cv = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    short_labels = ["COVID", "Normal", "Pneum.", "TB"]

    for idx, (name, (X, pipe)) in enumerate(scenarios.items()):
        y_pred = cross_val_predict(pipe, X, labels, cv=cv)
        cm = confusion_matrix(labels, y_pred, labels=CLASSES)

        im = axes[idx].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[idx].set_title(name, fontweight='bold', fontsize=10)
        axes[idx].set_xlabel('Predetto')
        axes[idx].set_ylabel('Reale')
        axes[idx].set_xticks(range(len(CLASSES)))
        axes[idx].set_yticks(range(len(CLASSES)))
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

    plt.suptitle(f'Confusion Matrix -- KNN (k={KNN_N_NEIGHBORS})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase6_confusion_matrices.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
#  6.4 -- CURVE ROC MULTICLASSE
# ============================================================================

def plot_roc_curves(
    scenarios: dict[str, tuple[np.ndarray, Pipeline]],
    labels: np.ndarray,
    save: bool = True
) -> None:
    """Curve ROC One-vs-Rest per ogni scenario."""
    from sklearn.base import clone

    n = len(scenarios)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    cv = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(labels)
    short_labels = ["COVID", "Normal", "Pneum.", "TB"]

    for idx, (name, (X, pipe)) in enumerate(scenarios.items()):
        y_prob = np.zeros_like(y_bin, dtype=np.float64)

        for train_idx, test_idx in cv.split(X, labels):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = labels[train_idx]

            pipe_fold = clone(pipe)
            pipe_fold.fit(X_train, y_train)
            proba = pipe_fold.predict_proba(X_test)

            clf_classes = pipe_fold.named_steps['knn'].classes_
            for ci, cls_name in enumerate(lb.classes_):
                if cls_name in clf_classes:
                    src_col = list(clf_classes).index(cls_name)
                    y_prob[test_idx, ci] = proba[:, src_col]

        macro_auc = []
        for ci in range(len(lb.classes_)):
            fpr, tpr, _ = roc_curve(y_bin[:, ci], y_prob[:, ci])
            roc_auc = auc(fpr, tpr)
            macro_auc.append(roc_auc)
            axes[idx].plot(fpr, tpr, color=CLASS_COLORS[ci], linewidth=2,
                           label=f'{short_labels[ci]} (AUC={roc_auc:.2f})')

        axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1)
        mean_auc = np.mean(macro_auc)
        axes[idx].set_title(f'{name}\nMacro AUC = {mean_auc:.3f}',
                            fontweight='bold', fontsize=10)
        axes[idx].set_xlabel('False Positive Rate')
        axes[idx].set_ylabel('True Positive Rate')
        axes[idx].legend(fontsize=7, loc='lower right')
        axes[idx].set_xlim([-0.02, 1.02])
        axes[idx].set_ylim([-0.02, 1.02])

    plt.suptitle(f'Curve ROC -- KNN (k={KNN_N_NEIGHBORS}) -- One-vs-Rest',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase6_roc_curves.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
#  6.5 -- CONFRONTO METRICHE TRA SCENARI
# ============================================================================

def plot_classification_comparison(
    results: dict,
    save: bool = True
) -> None:
    """Grafico a barre raggruppate: confronto metriche tra scenari."""
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
    ax.set_title(f'Confronto Metriche -- KNN (k={KNN_N_NEIGHBORS}) -- '
                 f'Stratified {CV_N_FOLDS}-Fold CV',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.65, 1.0)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase6_classification_comparison.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
#  6.6 -- HERO CHART: DUE STRATEGIE A CONFRONTO
# ============================================================================

def plot_hero_tradeoff(
    results: dict,
    save: bool = True
) -> None:
    """Grafico 'eroe' a due pannelli, design pulito.

    Sinistra: SVD — Accuracy vs k  (asse secondario: % dati memorizzati)
    Destra:   PCA — Accuracy vs n  (asse secondario: feature usate su 65.536)
    """
    from config import TARGET_SIZE

    h, w = TARGET_SIZE
    n_pixels = h * w

    # ── Estrai baseline ──
    raw_key = [k for k in results if k.startswith("Raw Pixels")]
    if not raw_key:
        return
    raw_key = raw_key[0]
    raw_acc, raw_std = results[raw_key]["accuracy"]

    # ── SVD ──
    svd_points = []
    for name, metrics in results.items():
        if name.startswith("SVD k="):
            k = int(name.split("k=")[1].split(" ")[0])
            pct = k * (h + w + 1) / n_pixels * 100
            svd_points.append((k, pct, metrics["accuracy"][0],
                               metrics["accuracy"][1]))
    svd_points.sort()

    # ── PCA ──
    pca_points = []
    for name, metrics in results.items():
        if name.startswith("PCA ("):
            n = int(name.split("(")[1].split(" ")[0])
            pca_points.append((n, metrics["accuracy"][0],
                               metrics["accuracy"][1]))
    pca_points.sort()

    if not svd_points and not pca_points:
        return

    # ── Figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # ===========  SVD  ===========
    # Baseline band
    ax1.axhspan(raw_acc - raw_std, raw_acc + raw_std,
                color='#e74c3c', alpha=0.07)
    ax1.axhline(raw_acc, color='#e74c3c', ls='--', alpha=0.6, lw=1.5,
                label=f'Baseline: Raw Pixels ({raw_acc*100:.1f}%)')

    # Curva SVD
    ks = [p[0] for p in svd_points]
    accs = [p[2] for p in svd_points]
    stds = [p[3] for p in svd_points]

    ax1.errorbar(ks, accs, yerr=stds, fmt='o-', color='#e67e22',
                 markersize=11, mec='white', mew=2,
                 capsize=4, elinewidth=1.5, lw=2.5,
                 label='SVD troncata', zorder=5)

    # Valori semplici sopra ogni punto
    for k, pct, acc, std in svd_points:
        ax1.text(k, acc + std + 0.006, f'{acc*100:.1f}%',
                 ha='center', va='bottom', fontsize=9.5,
                 fontweight='bold', color='#e67e22')

    # Asse X secondario: % dati memorizzati
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(ks)
    ax1_top.set_xticklabels([f'{p[1]:.0f}%' for p in svd_points],
                            fontsize=9, color='#888888')
    ax1_top.set_xlabel('Dati memorizzati (%)', fontsize=10, color='#888888')

    ax1.set_xlabel('Rango SVD (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('SVD: Compressione Immagine', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(max(0.65, raw_acc - 0.06), min(1.0, raw_acc + 0.06))
    if ks:
        ax1.set_xlim(min(ks) - 3, max(ks) + 5)

    # Take-away SVD
    ax1.text(0.03, 0.97,
             'Comprimere al 4% dei dati\nnon degrada la classificazione',
             transform=ax1.transAxes, fontsize=9.5, va='top',
             fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.4', fc='#fff8e7',
                       ec='#e67e22', alpha=0.95))

    # ===========  PCA  ===========
    # Baseline band
    ax2.axhspan(raw_acc - raw_std, raw_acc + raw_std,
                color='#e74c3c', alpha=0.07)
    ax2.axhline(raw_acc, color='#e74c3c', ls='--', alpha=0.6, lw=1.5,
                label=f'Baseline: Raw Pixels ({raw_acc*100:.1f}%)')

    # Curva PCA
    ns = [p[0] for p in pca_points]
    accs = [p[1] for p in pca_points]
    stds = [p[2] for p in pca_points]

    ax2.errorbar(ns, accs, yerr=stds, fmt='D-', color='#27ae60',
                 markersize=11, mec='white', mew=2,
                 capsize=4, elinewidth=1.5, lw=2.5,
                 label='PCA riduzione', zorder=5)

    # Valori semplici sopra ogni punto
    for n, acc, std in pca_points:
        ax2.text(n, acc + std + 0.006, f'{acc*100:.1f}%',
                 ha='center', va='bottom', fontsize=9.5,
                 fontweight='bold', color='#27ae60')

    # Asse X secondario: feature su 65.536
    ax2_top = ax2.twiny()
    ax2_top.set_xlim(ax2.get_xlim())
    ax2_top.set_xticks(ns)
    ax2_top.set_xticklabels([f'{n/n_pixels*100:.2f}%' for n in ns],
                            fontsize=9, color='#888888')
    ax2_top.set_xlabel('Feature usate (% su 65.536)', fontsize=10,
                       color='#888888')

    ax2.set_xlabel('Componenti PCA (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('PCA: Riduzione Dimensionale', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(max(0.65, raw_acc - 0.06), min(1.0, raw_acc + 0.06))
    if ns:
        ax2.set_xlim(min(ns) - 8, max(ns) + 15)

    # Take-away PCA
    ax2.text(0.03, 0.97,
             '25 feature su 65.536 (0.04%)\nclassificano meglio dei pixel grezzi',
             transform=ax2.transAxes, fontsize=9.5, va='top',
             fontstyle='italic',
             bbox=dict(boxstyle='round,pad=0.4', fc='#e8f8e8',
                       ec='#27ae60', alpha=0.95))

    fig.suptitle('Trade-off: Compressione / Riduzione vs Capacità Diagnostica',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(OUTPUT_DIR, 'fase6_hero_tradeoff.png'),
                    dpi=150, bbox_inches='tight')
    plt.show()

