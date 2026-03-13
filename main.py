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

from config import setup_plot_style
from data_loader import (count_images, print_image_counts,
                         load_sample_images, print_sample_properties,
                         load_and_preprocess, load_dataset)
from svd_engine import apply_svd, print_svd_info
from visualization import (plot_exploration, plot_svd_reconstruction,
                           plot_class_comparison, plot_tilt_correction)
from analysis import (plot_scree, plot_mse_psnr, run_pca, plot_pca_scatter,
                      plot_eigenfaces, print_summary_table)


def main():
    # --- Configurazione ---
    setup_plot_style()

    # ==================================================================
    #  FASE 1 -- Esplorazione dei Dati
    # ==================================================================
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
    plot_tilt_correction(images_by_class)

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

    print("\n" + "=" * 60)
    print("  [OK] PROGETTO COMPLETATO -- Tutti i grafici salvati in output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
