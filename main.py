"""
main.py — Esegue in sequenza i due notebook del progetto.

Ordine di esecuzione:
  1. medical_image_svd_pca.ipynb       (dataset medico, 4 classi)
  2. mvtec/mvtec_analysis.ipynb        (dataset MVTec AD Screw, binario)

Ogni notebook viene eseguito in-place: le celle vengono rieseguite e gli
output (stdout + figure) sovrascritti nel file .ipynb stesso. Le figure
vengono salvate anche su disco dai rispettivi moduli (output/ per il
medico, outputs_mvtec/ per MVTec).

Uso:
    python main.py

Requisiti:
    pip install nbclient nbformat jupyter
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


ROOT = Path(__file__).resolve().parent

# Notebook da eseguire, in ordine.
# cwd indica la directory di lavoro con cui il kernel viene avviato:
#   - il notebook medico si aspetta di partire dalla root (usa "from src.*")
#   - il notebook MVTec sta in mvtec/ e risale a ../ via sys.path al runtime
NOTEBOOKS = [
    {
        "path": ROOT / "medical_image_svd_pca.ipynb",
        "cwd":  ROOT,
        "label": "1. Dataset medico (Chest X-Ray, 4 classi)",
    },
    {
        "path": ROOT / "mvtec" / "mvtec_analysis.ipynb",
        "cwd":  ROOT / "mvtec",
        "label": "2. Dataset MVTec AD Screw (good vs defective)",
    },
]


def run_notebook(nb_path: Path, cwd: Path, timeout_per_cell: int = 900) -> None:
    """Esegue un notebook in-place, salvando output e figure nel file .ipynb."""
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook non trovato: {nb_path}")

    print(f"  -> caricamento {nb_path.relative_to(ROOT)}")
    nb = nbformat.read(nb_path, as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout_per_cell,
        kernel_name="python3",
        resources={"metadata": {"path": str(cwd)}},
    )

    print(f"  -> esecuzione celle (cwd = {cwd.relative_to(ROOT) or '.'})")
    t0 = time.time()
    client.execute()
    dt = time.time() - t0

    nbformat.write(nb, nb_path)
    print(f"  -> completato in {dt:.1f}s, notebook ri-salvato")


def main() -> int:
    print("#" * 70)
    print("  PROGETTO — SVD e PCA per compressione e analisi di immagini")
    print("  Esecuzione sequenziale dei due notebook")
    print("#" * 70)

    t_start = time.time()
    for i, nb in enumerate(NOTEBOOKS, start=1):
        print(f"\n[{i}/{len(NOTEBOOKS)}] {nb['label']}")
        print("-" * 70)
        try:
            run_notebook(nb["path"], nb["cwd"])
        except CellExecutionError as e:
            print(f"\n[FALLITO] Errore in una cella di {nb['path'].name}:")
            print(str(e)[:1500])
            print(f"\nNota: l'esecuzione si ferma qui. "
                  f"I notebook successivi NON verranno eseguiti.")
            return 1
        except FileNotFoundError as e:
            print(f"\n[FALLITO] {e}")
            return 1

    dt = time.time() - t_start
    print("\n" + "#" * 70)
    print(f"  [OK] Entrambi i notebook eseguiti con successo in {dt:.1f}s")
    print(f"  Output: output/ (medico), outputs_mvtec/ (MVTec)")
    print("#" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
