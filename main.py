"""
main.py — Script principale del progetto.

Esegue in sequenza tutte le 5 fasi dell'analisi:
  1. Esplorazione dei dati
  2. Pre-elaborazione
  3. SVD (motore matematico)
  4. Ricostruzione e visualizzazione
  5. Analisi statistica e PCA

Uso:
    python main.py
"""
import matplotlib
matplotlib.use('Agg')

from src.config import setup_plot_style
from src.data_loader import (count_images, print_image_counts,
                             load_sample_images, print_sample_properties,
                             load_dataset, load_raw_samples)
from src.svd_engine import apply_svd, print_svd_info
from src.visualization import (plot_exploration, plot_svd_reconstruction,
                               plot_class_comparison, plot_tilt_correction)
from src.analysis import plot_scree, plot_mse_psnr, print_summary_table
from src.pca_engine import run_pca, plot_pca_scatter, plot_eigenfaces


def main():
    # --- Configurazione ---
    setup_plot_style()

    # =================================================================
    #  FASE 1 -- Esplorazione dei Dati
    # =================================================================
    print("\n" + "#" * 60)
    print("  FASE 1 -- ESPLORAZIONE DEI DATI")
    print("#" * 60)

    counts = count_images()
    print_image_counts(counts)

    sample_images = load_sample_images()
    print_sample_properties(sample_images)

    plot_exploration(sample_images)

    # ==================================================================
    #  FASE 2 -- Pre-elaborazione
    # ==================================================================
    print("\n" + "#" * 60)
    print("  FASE 2 -- PRE-ELABORAZIONE")
    print("#" * 60)

    images_by_class, all_images, labels = load_dataset()

    # Carica immagini raw (senza tilt correction) per il confronto vero
    raw_samples = load_raw_samples()
    plot_tilt_correction(raw_samples)

    # ==================================================================
    #  FASE 3 -- SVD
    # ==================================================================
    print("\n" + "#" * 60)
    print("  FASE 3 -- SVD (MOTORE MATEMATICO)")
    print("#" * 60)

    demo_img = images_by_class["Normal"][0]
    U, S, Vt = apply_svd(demo_img)
    print_svd_info(demo_img, U, S, Vt)

    # ==================================================================
    #  FASE 4 -- Ricostruzione e Visualizzazione
    # ==================================================================
    print("\n" + "#" * 60)
    print("  FASE 4 -- RICOSTRUZIONE E VISUALIZZAZIONE")
    print("#" * 60)

    plot_svd_reconstruction(demo_img, class_name="Normal")
    plot_class_comparison(images_by_class)

    # ==================================================================
    #  FASE 5 -- Analisi Statistica e PCA
    # ==================================================================
    print("\n" + "#" * 60)
    print("  FASE 5 -- ANALISI STATISTICA E PCA")
    print("#" * 60)

    # 5.1 Scree plot
    plot_scree(images_by_class)

    # 5.2 MSE / PSNR vs k
    plot_mse_psnr(images_by_class)

    # 5.3 PCA
    pca_model, X_pca = run_pca(all_images)
    plot_pca_scatter(pca_model, X_pca, labels)
    plot_eigenfaces(pca_model)

    # 5.4 Tabella riassuntiva
    print_summary_table(demo_img, class_name="Normal")

    # ==================================================================
    #  FASE 6 -- Classificazione e Confronto Feature
    # ==================================================================
    print("\n" + "#" * 60)
    print("  FASE 6 -- CLASSIFICAZIONE E CONFRONTO FEATURE")
    print("#" * 60)

    from src.classification import (prepare_feature_sets, run_classification,
                                     plot_confusion_matrix, plot_roc_curves,
                                     plot_classification_comparison,
                                     plot_hero_tradeoff, plot_knn_vs_lr)

    # Cache SVD: le ricostruzioni vengono calcolate una sola volta
    # e riusate per entrambi i classificatori
    svd_cache = {}

    # 6.1a Preparazione scenari -- KNN
    print("\n--- Preparazione scenari KNN ---")
    scenarios_knn = prepare_feature_sets(all_images, labels,
                                         classifier='knn',
                                         svd_cache=svd_cache)

    # 6.1b Preparazione scenari -- Logistic Regression (riusa cache SVD)
    print("\n--- Preparazione scenari Logistic Regression ---")
    scenarios_lr = prepare_feature_sets(all_images, labels,
                                        classifier='lr',
                                        svd_cache=svd_cache)

    # 6.2 Classificazione con cross-validation stratificata
    results_knn = run_classification(scenarios_knn, labels, classifier='knn')
    results_lr  = run_classification(scenarios_lr,  labels, classifier='lr')

    # 6.3 Confusion matrices (entrambi i classificatori)
    plot_confusion_matrix(scenarios_knn, labels, classifier='knn')
    plot_confusion_matrix(scenarios_lr,  labels, classifier='lr')

    # 6.4 Curve ROC multiclasse (entrambi)
    plot_roc_curves(scenarios_knn, labels, classifier='knn')
    plot_roc_curves(scenarios_lr,  labels, classifier='lr')

    # 6.5 Confronto metriche tra scenari (entrambi)
    plot_classification_comparison(results_knn, classifier='knn')
    plot_classification_comparison(results_lr,  classifier='lr')

    # 6.6 Hero chart: Accuracy vs Compressione (entrambi)
    plot_hero_tradeoff(results_knn, classifier='knn')
    plot_hero_tradeoff(results_lr,  classifier='lr')

    # 6.7 Confronto diretto KNN vs Logistic Regression
    plot_knn_vs_lr(results_knn, results_lr)

    print("\n" + "=" * 60)
    print("  [OK] PROGETTO COMPLETATO -- Tutti i grafici salvati in output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
