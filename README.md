# SVD or PCA for Image Analysis in Medical Diagnostics

**Corso:** Statistical Methods — Politecnico di Bari, 1° Anno  
**Dataset:** Lung X-Rays Grayscale — 4 classi diagnostiche  
**Tecnologie:** Python · NumPy · scikit-learn · Pillow · Matplotlib/Seaborn

---

## Indice

1. [Motivazione Clinica](#1-motivazione-clinica)
2. [Dataset e Classi Diagnostiche](#2-dataset-e-classi-diagnostiche)
3. [Fondamenti Matematici](#3-fondamenti-matematici)
   - 3.1 [Singular Value Decomposition (SVD)](#31-singular-value-decomposition-svd)
   - 3.2 [Principal Component Analysis (PCA)](#32-principal-component-analysis-pca)
   - 3.3 [Relazione SVD ↔ PCA](#33-relazione-svd--pca)
4. [Pipeline di Analisi](#4-pipeline-di-analisi)
   - 4.1 [Fase 1 — Esplorazione dei Dati](#41-fase-1--esplorazione-dei-dati)
   - 4.2 [Fase 2 — Pre-elaborazione](#42-fase-2--pre-elaborazione)
   - 4.3 [Fase 3 — SVD (Motore Matematico)](#43-fase-3--svd-motore-matematico)
   - 4.4 [Fase 4 — Ricostruzione e Visualizzazione](#44-fase-4--ricostruzione-e-visualizzazione)
   - 4.5 [Fase 5 — Analisi Statistica e PCA](#45-fase-5--analisi-statistica-e-pca)
   - 4.6 [Fase 6 — Classificazione e Confronto Feature](#46-fase-6--classificazione-e-confronto-feature)
5. [Architettura del Codice](#5-architettura-del-codice)
6. [Discussione Critica](#6-discussione-critica)
7. [Conclusioni](#7-conclusioni)

---

## 1. Motivazione Clinica

Le radiografie toraciche sono uno degli strumenti diagnostici più diffusi in medicina: ogni ospedale produce migliaia di immagini al giorno. Questo genera due problemi pratici:

- **Spazio di archiviazione:** una radiografia ad alta risoluzione può occupare decine di MB. Moltiplicando per i volumi ospedalieri, la necessità di compressione è concreta.
- **Trasmissione in telemedicina:** la diagnostica remota richiede l'invio di immagini su reti a banda limitata; la compressione senza perdita diagnostica è fondamentale.

Il progetto si chiede: **è possibile comprimere significativamente una radiografia polmonare mantenendo intatta l'informazione diagnosticamente rilevante?** E, in parallelo: **le tecniche di riduzione dimensionale riescono a distinguere automaticamente diverse patologie polmonari?**

SVD e PCA sono gli strumenti matematici scelti per rispondere a queste domande.

---

## 2. Dataset e Classi Diagnostiche

### Composizione

Il dataset contiene radiografie toraciche in scala di grigi suddivise in **4 classi**:

| Classe | Patologia | N° immagini disponibili | N° immagini usate |
|---|---|---|---|
| **Corona Virus Disease** | COVID-19 | 122 | ≤ 120 |
| **Normal** | Polmoni sani | 540 | ≤ 120 |
| **Pneumonia** | Polmonite batterica/virale | 186 | ≤ 120 |
| **Tuberculosis** | Tubercolosi polmonare | 333 | ≤ 120 |

### Caratteristiche delle immagini grezze

Le immagini originali hanno dimensioni eterogenee:

| Classe | Dimensione campione | Range intensità | Media | Dev. Std. |
|---|---|---|---|---|
| COVID-19 | 230 × 203 px | 0–255 | 116.07 | 44.54 |
| Normal | 1654 × 1678 px | 0–255 | 124.91 | 57.82 |
| Pneumonia | 1216 × 1512 px | 0–255 | 77.16 | 41.98 |
| Tuberculosis | 512 × 512 px | 0–193 | 126.62 | 39.60 |


### Osservazioni diagnostiche dagli istogrammi (Fase 1)

- **COVID-19:** distribuzione concentrata nelle tonalità medio-alte (100–170), coerente con le opacità a vetro smerigliato tipiche del COVID.
- **Normal:** distribuzione più uniforme su tutto il range, buon contrasto tra tessuti.
- **Pneumonia:** forte picco nelle tonalità scure (30–80), coerente con addensamenti polmonari densi.
- **Tuberculosis:** distribuzione bimodale con picco nelle tonalità chiare (150–190), indicativo di lesioni fibrotiche calcificate.

---

## 3. Fondamenti Matematici

### 3.1 Singular Value Decomposition (SVD)

Data una matrice immagine $A \in \mathbb{R}^{m \times n}$ (con $m = n = 256$), la SVD la fattorizza come:

$$A = U \cdot \Sigma \cdot V^T$$

dove:
- $U \in \mathbb{R}^{m \times r}$: **vettori singolari sinistri** — catturano le strutture spaziali lungo le righe
- $\Sigma \in \mathbb{R}^{r \times r}$: matrice diagonale con **valori singolari** $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r \geq 0$, con $r = \min(m, n)$
- $V^T \in \mathbb{R}^{r \times n}$: **vettori singolari destri** — catturano le strutture lungo le colonne

I valori singolari quantificano l'"importanza" di ciascuna direzione: $\sigma_1$ codifica la variazione di maggiore energia (es. contrasto globale), i valori successivi pattern via via più fini.

#### Implementazione (svd_engine.py)

```python
U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)  # economy SVD
```

#### Approssimazione di rango k

Mantenendo solo le prime $k$ componenti si ottiene la **ricostruzione troncata**:

$$A_k = \sum_{i=1}^{k} \sigma_i \cdot \mathbf{u}_i \cdot \mathbf{v}_i^T = U_k \cdot \Sigma_k \cdot V_k^T$$

```python
reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :]
```

#### Teorema di Eckart-Young (1936) — Il nucleo teorico del progetto

> **$A_k$ è la migliore approssimazione di rango $k$ di $A$ in norma di Frobenius:**
> $$A_k = \arg\min_{\operatorname{rank}(B) = k} \|A - B\|_F$$
> con errore minimo:
> $$\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

Questo è il fondamento teorico che rende la SVD **ottimale** per la compressione: non esiste nessun'altra decomposizione di rango $k$ che approssimi meglio l'immagine originale.

#### Rapporto di compressione

Con $k$ componenti si memorizzano $k(m + n + 1)$ valori anziché $m \cdot n$:

$$\text{compression ratio} = \frac{k(m + n + 1)}{m \cdot n}$$

Per $m = n = 256$: con $k = 20$, compression ratio ≈ **15.7%**.

#### Metriche di qualità

| Metrica | Formula | Soglia pratica in radiologia |
|---|---|---|
| **MSE** | $\frac{1}{mn}\sum_{i,j}(A_{ij} - \hat{A}_{ij})^2$ | Più basso = meglio |
| **PSNR** | $10 \log_{10}\!\left(\frac{1}{\text{MSE}}\right)$ [dB] | **> 30 dB** considerato accettabile diagnosticamente |

---

### 3.2 Principal Component Analysis (PCA)

La PCA cerca le direzioni di **massima varianza** nello spazio delle immagini. Applicata a un dataset di $N$ immagini di $d = 256 \times 256 = 65\,536$ pixel:

1. Si costruisce la matrice dati $X \in \mathbb{R}^{N \times d}$ (ogni riga = un'immagine appiattita)
2. Si centra: $\tilde{X} = X - \bar{X}$
3. La PCA trova le componenti principali $\mathbf{v}_1, \mathbf{v}_2, \ldots$ che massimizzano $\operatorname{Var}(\tilde{X} \mathbf{v}_i)$ con vincolo di ortogonalità

#### Implementazione (src/analysis.py)

```python
X = all_images.reshape(all_images.shape[0], -1)  # (N, 65536)
pca_model = PCA(n_components=50)
X_pca = pca_model.fit_transform(X)               # (N, 50)
```

La libreria `sklearn.decomposition.PCA` utilizza internamente la SVD randomizzata per efficienza.

#### Eigenfaces / Eigen-Xrays

Le **componenti principali** possono essere visualizzate come immagini 256 × 256. Mostrano i pattern visivi dominanti nel dataset:

| Componente | Varianza spiegata | Pattern clinico associato |
|---|---|---|
| **PC1** | 25.6% | Contrasto globale luminosità/sfondo |
| **PC2** | 17.4% | Simmetria laterale del torace |
| **PC3** | 5.1% | Pattern mediastinico |
| **PC4** | 4.8% | Strutture costali |
| **PC5–10** | 1.6–3.9% | Dettagli fini: campi polmonari, diaframma, apici |

---

### 3.3 Relazione SVD ↔ PCA

> **PCA e SVD non sono due metodi diversi: la PCA è la SVD applicata alla matrice dei dati centrata.**

| Passo | Operazione |
|---|---|
| 1. Flatten | Ogni immagine $(256 \times 256)$ → vettore $\mathbf{x} \in \mathbb{R}^{65536}$ |
| 2. Stack | Matrice dati $X \in \mathbb{R}^{N \times 65536}$ |
| 3. Centra | $\tilde{X} = X - \bar{X}$ |
| 4. SVD | $\tilde{X} = U \Sigma V^T$ |
| 5. Componenti principali | Colonne di $V^T$ (= direzioni di massima varianza) |
| 6. Proiezione | $X_\text{PCA} = \tilde{X} \cdot V_k^T \in \mathbb{R}^{N \times k}$ |

Mentre la SVD per la compressione opera su **una singola immagine**, la PCA opera sull'**intero dataset**. Le due tecniche sono complementari, non alternative:

| | SVD per compressione | PCA per analisi |
|---|---|---|
| **Input** | Singola immagine $A \in \mathbb{R}^{m \times n}$ | Dataset $X \in \mathbb{R}^{N \times d}$ |
| **Output** | Approssimazione $A_k$ | Proiezione $X_\text{PCA} \in \mathbb{R}^{N \times k}$ |
| **Obiettivo** | Ridurre i dati da memorizzare | Trovare struttura latente comune |
| **Uso pratico** | Compressione, trasmissione | Classificazione esplorativa, visualizzazione |

---

## 4. Pipeline di Analisi

### 4.1 Fase 1 — Esplorazione dei Dati

**Obiettivo:** verificare la validità strutturale delle immagini e capire le loro caratteristiche statistiche prima di qualsiasi elaborazione.

**Operazioni:**
- Conteggio immagini per classe (`count_images()`)
- Ispezione di shape, dtype, range e distribuzione dei pixel (`print_sample_properties()`)
- Verifica che ogni immagine sia 2D (scala di grigi)
- Esclusione preventiva delle **radiografie laterali** (proiezione di profilo, non PA)

**Rilevamento radiografie laterali (`is_lateral_xray`):**  
Si calcola il profilo medio di intensità per colonna e si conta la frazione di colonne con segnale significativo (> 30% del max). Se < 60% delle colonne è attiva → immagine laterale, scartata.

![Fase 1 — Campioni e istogrammi per classe](output/fase1_esplorazione.png)

---

### 4.2 Fase 2 — Pre-elaborazione

**Obiettivo:** portare tutte le immagini in un formato uniforme adatto all'analisi matriciale.

```
File .jpg/.png (immagine grezza, dimensioni originali)
  │
  ├─ 1. Caricamento e conversione in scala di grigi (PIL, modalità 'L')
  │
  ├─ 2. Filtraggio radiografie laterali  ──► SCARTA se profilo attivo < 60%
  │      (su immagine grezza, dimensioni originali)
  │
  ├─ 3. Validazione immagine  ──────────────► SCARTA se non valida
  │      · file noti corrotti (blacklist 12 file)
  │      · padding artificiale (≥ 3 angoli con media > 130)
  │      · tilt eccessivo (|θ| > 10°)
  │      (su immagine grezza, dimensioni originali)
  │
  ├─ 4. Resize 256×256 (interpolazione LANCZOS)
  │
  ├─ 5. Normalizzazione → float64 in [0, 1]
  │
  └─ 6. Correzione inclinazione (se |θ| ≥ 0.5°)
         · scipy_rotate(image, -θ, order=1, mode='constant', cval=0.0)
         · clip finale in [0, 1]
```

> **Nota:** la validazione (step 2–3) avviene sull'immagine originale a piena risoluzione. Solo le immagini che superano i controlli vengono ridimensionate. La correzione tilt (step 6) opera sull'immagine già ridimensionata a 256×256.

#### Scarto vs correzione del tilt

Le due operazioni riguardano **immagini diverse**:

| Operazione | Soglia | Cosa succede |
|---|---|---|
| **Scarto** (`is_valid_image`) | `\|θ\| > 10°` | Inclinazione grave → irrecuperabile, rimossa dal dataset |
| **Correzione** (`correct_tilt`) | `0.5° ≤ \|θ\| ≤ 10°` | Inclinazione lieve → raddrizzata automaticamente |
| **Nessuna azione** | `\|θ\| < 0.5°` | Già correttamente orientata |

#### Correzione dell'inclinazione (Tilt Detection)

**Principio:** nelle rx toraciche le costole creano strutture fortemente orizzontali. Un'immagine correttamente orientata produce un profilo riga con alta varianza del gradiente. L'algoritmo cerca l'angolo che massimizza questa varianza:

$$\theta^* = \arg\max_\theta \operatorname{Var}\!\left(\nabla_\text{row}\left[\text{rotate}(A, -\theta)\right]\right)$$

**Ricerca a due passi per efficienza:**
1. **Coarse** — step 2°, range ±25°, immagine downscaled a 128px per velocità
2. **Fine** — step 0.5°, intorno al massimo coarse ±3°

#### Filtraggio immagini non valide

| Criterio | Soglia | Azione |
|---|---|---|
| Rx laterale | profilo attivo < 60% | Scartata |
| Tilt eccessivo | `|θ| > 10°` | Scartata |
| Padding corrotto | ≥ 3 angoli con media > 130/255 | Scartata |
| File noti corrotti | lista hardcoded (12 file) | Scartati |

![Fase 2 — Correzione dell'inclinazione](output/fase2_correzione_tilt.png)

**Parametri globali (config.py):**

| Parametro | Valore | Descrizione |
|---|---|---|
| `TARGET_SIZE` | `(256, 256)` | Dimensione di ridimensionamento |
| `MAX_PER_CLASS` | `120` | Immagini massime per classe |
| `CORRECT_TILT` | `True` | Abilita correzione inclinazione |
| `MAX_TILT_ANGLE` | `10` | Soglia scarto inclinazione (gradi) |

---

### 4.3 Fase 3 — SVD (Motore Matematico)

**Obiettivo:** applicare la SVD a una singola immagine campione e comprendere la struttura della decomposizione.

```python
U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
```

Per un'immagine 256 × 256, la SVD economica produce:
- `U`: `(256, 256)` — pattern strutturali per riga
- `S`: `(256,)` — i 256 valori singolari in ordine decrescente
- `Vt`: `(256, 256)` — pattern strutturali per colonna

**Informazioni chiave dalla decomposizione:**
- **Primo valore singolare** ≈ 130 (cattura la luminosità media globale)
- **Decadimento rapido:** dopo i primi 10–15, i valori singolari sono prossimi a zero
- 90% della varianza spiegata in pochissime componenti (k ≈ 1–3)

---

### 4.4 Fase 4 — Ricostruzione e Visualizzazione

#### 4.4.1 Ricostruzione a diversi livelli di k

Valori di $k$ analizzati: `[1, 5, 10, 20, 50, 100, 200]`

| k | Compression Ratio | PSNR (dB) | Qualità visiva |
|---|---|---|---|
| 1 | 0.8% | 17.3 | Solo luminosità media |
| 5 | 3.9% | 23.8 | Sagoma del torace |
| 10 | 7.8% | 27.8 | Costole, mediastino visibili |
| **20** | **15.7%** | **32.4** | **Qualità diagnostica ✓ (PSNR > 30 dB)** |
| 50 | 39.1% | 40.1 | Dettaglio preservato |
| 100 | 78.3% | 50.0 | Quasi identico all'originale |
| 200 | 100% | 71.5 | Ricostruzione perfetta |

> 🔑 **Risultato chiave:** con $k = 20$ (soli 15.7% dei dati memorizzati) il PSNR supera **30 dB** — soglia convenzionale di qualità accettabile in diagnostica per immagini.

Il **MSE** è calcolato direttamente dai valori singolari (ottimizzazione che sfrutta l'Eckart-Young theorem):
```python
mse_vals = (total_energy - cumsum_S_sq[k_range - 1]) / (m * n)
```
Questo evita 200 moltiplicazioni matriciali complete.

![Fase 4 — Ricostruzione SVD con diversi k](output/fase4_ricostruzione_svd.png)

#### 4.4.2 Confronto tra classi

Valori di $k$ per il confronto: `[5, 20, 50, 100]`

Il comportamento è **coerente tra le 4 classi**: tutte raggiungono PSNR > 30 dB con $k = 20$ e PSNR > 48 dB con $k = 100$.

Pneumonia e Tuberculosis raggiungono PSNR leggermente più alti a parità di $k$ → le loro immagini hanno struttura a **più basso rango** (più comprimibili). COVID-19 e Normal presentano MSE leggermente più alti a basso $k$ → **maggiore complessità strutturale**.

![Fase 4 — Confronto SVD tra classi](output/fase4_confronto_classi.png)

---

### 4.5 Fase 5 — Analisi Statistica e PCA

#### 5.1 Scree Plot e Decadimento dei Valori Singolari

Tutte le classi mostrano lo stesso pattern di decadimento rapido:

| Soglia varianza | k necessario (tipico) |
|---|---|
| 90% | k ≈ 1–3 |
| 95% | k ≈ 5–10 |
| 99% | k ≈ 20–40 |
| 100% | k = 256 |

Le radiografie polmonari hanno una **fortissima struttura di basso rango**: il 90% dell'informazione è concentrata in pochissime componenti. Questo non è un risultato banale — dipende dall'alta correlazione spaziale nelle rx toraciche (sfondo uniforme, strutture curve ripetute come le costole).

![Fase 5 — Scree plot](output/fase5_scree_plot.png)

#### 5.2 Curve MSE e PSNR vs k

- **MSE:** decresce rapidamente fino a $k \approx 20$–30, poi si stabilizza
- **PSNR:** cresce logaritmicamente; il "punto di gomito" è a $k \approx 20$–50
- COVID-19 e Normal presentano MSE leggermente più alti a basso $k$

![Fase 5 — MSE e PSNR vs k](output/fase5_mse_psnr.png)

#### 5.3 PCA — Proiezione 2D e Separabilità

```python
X = all_images.reshape(N, -1)    # (N, 65536)
pca_model = PCA(n_components=50)
X_pca = pca_model.fit_transform(X)
```

**Risultati della proiezione PC1–PC2:**
- Le prime 2 componenti spiegano il **43% della varianza totale** (PC1: 25.6%, PC2: 17.4%)
- Le prime 50 componenti coprono circa il **90% della varianza**
- **Normal** e **Pneumonia** → metà sinistra (PC1 < 0)
- **COVID-19** e **Tuberculosis** → metà destra (PC1 > 0)
- C'è **sovrapposizione** — la separabilità è parziale, non perfetta

> **Interpretazione:** la sovrapposizione indica che SVD/PCA non bastano da soli per la diagnosi automatica. Sono strumenti di **analisi esplorativa** utili, non sostitutivi di classificatori supervisionati.

![Fase 5 — PCA scatter plot](output/fase5_pca_scatter.png)

#### 5.4 Eigenfaces (Eigen-Xrays)

Le prime 10 componenti principali, visualizzate come immagini 256 × 256 con colormap divergente (RdBu):

| Componente | Varianza | Pattern clinico |
|---|---|---|
| **PC1** | 25.6% | Contrasto globale — rx chiare vs scure |
| **PC2** | 17.4% | Simmetria laterale — profilo destro/sinistro |
| **PC3** | 5.1% | Pattern mediastinico |
| **PC4** | 4.8% | Strutture costali |
| **PC5–10** | 1.6–3.9% | Dettagli fini: campi polmonari, diaframma, apici |

![Fase 5 — Eigen-Xrays](output/fase5_eigenfaces.png)

---

### 4.6 Fase 6 — Classificazione e Confronto Feature

**Obiettivo:** validare quantitativamente che la compressione (SVD) e la riduzione dimensionale (PCA) **preservano la capacità diagnostica** confrontandole con la classificazione sui pixel grezzi.

Si addestrano e confrontano due classificatori (**KNN** con k=5 e **Logistic Regression**) usando una **Stratified 5-Fold Cross-Validation**, organizzati in due strategie distinte:

#### Strategia 1: SVD — Compressione dell'Immagine

Si applica la SVD troncata **a ogni singola immagine** per comprimerla, poi si classificano i 65.536 pixel ricostruiti.

| Scenario | Dati memorizzati | Dim. al classificatore |
|---|---|---|
| **Raw Pixels** | 100% | 65.536 |
| **SVD k=10** | ~7.8% | 65.536 |
| **SVD k=50** | ~39.1% | 65.536 |

**Messaggio:** la SVD rimuove rumore ad alta frequenza senza perdere informazione diagnostica. Con k=10 (~8% dei dati) si mantengono performance competitive con i pixel grezzi.

#### Strategia 2: PCA — Riduzione Dimensionale del Dataset

Si applica PCA **all'intero dataset** (matrice 320×65.536) per estrarre le direzioni di massima varianza. Il KNN lavora nello spazio ridotto.

| Scenario | Componenti | Riduzione dim. |
|---|---|---|
| **Raw Pixels** | 65.536 | 0% |
| **PCA (25)** | 25 | 99.96% |
| **PCA (150)** | 150 | 99.77% |

**Messaggio:** PCA agisce come feature extractor, selezionando le direzioni di massima varianza inter-immagine.

> **Nota metodologica:** per evitare **data leakage**, `StandardScaler` e `PCA` vengono fittati solo sul training set di ogni fold tramite `sklearn.Pipeline`.

#### Risultati

La **Logistic Regression** ha dimostrato un primato assoluto rispetto al KNN, superando i limiti della distanza euclidea nello spazio dei pixel. 
- La baseline (Raw Pixels) con Logistic Regression si attesta intorno all'**85.1%** di accuratezza.
- **Miglior Scenario Assoluto:** PCA con 150 componenti fornite alla Logistic Regression raggiunge l'**86.1%** di accuratezza.
- Con la compressione estrema della SVD (k=5, 4% dei dati), la Logistic Regression migliora le performance sfiorando l'**85.7%**, dimostrando l'effetto di denoising implicito della decomposizione singolare.

![Fase 6 — Confronto metriche](output/fase6_classification_comparison.png)

#### Hero Chart: Trade-off Compressione vs Diagnostica

I due grafici affiancati mostrano che:
- **SVD** (sinistra): comprimere le immagini fino al 96% (k=5) non degrada l'accuracy
- **PCA** (destra): ridurre la dimensionalità del 99.96% (25 componenti) mantiene o migliora l'accuracy

![Fase 6 — Hero chart](output/fase6_hero_tradeoff.png)

---

## 5. Architettura del Codice

```
Medical Image Compression and Analysis/
├── main.py                      # Script orchestratore (esegue le 6 fasi)
├── data/
│   └── raw/                     # Dataset originale (non versionato)
│       ├── Corona Virus Disease/
│       ├── Normal/
│       ├── Pneumonia/
│       └── Tuberculosis/
├── src/
│   ├── config.py                # Parametri globali (percorsi, costanti, stile grafici)
│   ├── data_loader.py           # Caricamento, validazione, tilt correction (Fasi 1–2)
│   ├── svd_engine.py            # SVD, ricostruzione troncata, metriche MSE/PSNR (Fase 3)
│   ├── visualization.py         # Plot delle fasi 1, 2, 4
│   ├── analysis.py              # Scree plot, MSE/PSNR, PCA, eigenfaces, tabella (Fase 5)
│   └── classification.py        # KNN, cross-validation, confronto feature (Fase 6)
├── output/                      # Grafici salvati (.png, 150 dpi)
└── medical_image_svd_pca.ipynb  # Notebook interattivo
```

### Dipendenze principali

| Libreria | Uso |
|---|---|
| `numpy` | SVD (`np.linalg.svd`), operazioni matriciali |
| `scipy.ndimage` | Rotazione immagini (correzione tilt) |
| `Pillow` | I/O immagini, resize Lanczos |
| `scikit-learn` | PCA, KNN, Pipeline, cross-validation, metriche |
| `matplotlib` + `seaborn` | Grafici e stile |

### Parametri configurabili (config.py)

| Parametro | Default | Descrizione |
|---|---|---|
| `TARGET_SIZE` | `(256, 256)` | Dimensione di ridimensionamento |
| `MAX_PER_CLASS` | `120` | Immagini massime per classe |
| `CORRECT_TILT` | `True` | Abilita correzione inclinazione |
| `MAX_TILT_ANGLE` | `10` | Soglia scarto inclinazione (gradi) |
| `K_VALUES_DEMO` | `[1,5,10,20,50,100,200]` | k per demo ricostruzione |
| `K_VALUES_COMPARE` | `[5,20,50,100]` | k per confronto tra classi |
| `K_RANGE_METRICS` | `range(1, 201)` | Range per curve MSE/PSNR |
| `PCA_N_COMPONENTS` | `50` | Numero componenti PCA (analisi) |
| `KNN_N_NEIGHBORS` | `5` | k per KNN classificazione |
| `CV_N_FOLDS` | `5` | Fold per cross-validation stratificata |
| `SVD_K_VALUES` | `[10, 50]` | k SVD per classificazione |
| `PCA_COMPONENTS_LIST` | `[25, 150]` | Componenti PCA per classificazione |

---

## 6. Discussione Critica

### Cosa funziona bene

1. **La SVD è efficace per la compressione.** Il Teorema di Eckart-Young garantisce l'ottimalità teorica; i risultati empirici confermano che $k = 20$ è sufficiente per qualità diagnostica (PSNR > 30 dB) usando solo il 15.7% dei dati.

2. **La struttura di basso rango è una proprietà robusta.** Tutte e 4 le classi mostrano lo stesso pattern di decadimento rapido dei valori singolari. Coerenza tra patologie molto diverse conferma la generalità del risultato.

3. **Gli eigenfaces rivelano pattern clinicamente interpretabili.** PC1 = contrasto globale, PC2 = simmetria laterale, PC3 = mediastino. L'algoritmo non conosce l'anatomia ma la "scopre" dai dati.

4. **Il pre-processing è robusto.** Il sistema di validazione a più livelli (laterali, tilt eccessivo, padding corrotto, file noti) garantisce che solo immagini valide entrino nell'analisi.

5. **La SVD preserva e migliora la capacità diagnostica.** La classificazione su pixel ricostruiti con SVD troncata estrema (k=5, k=10) sfiora l'**85.7%** con la Logistic Regression, battendo leggermente la baseline sui raw pixels. Elimininare l'alta frequenza agisce come potente filtro anti-rumore diagnostico.

6. **La Logistic Regression scala linearmente superando la Maledizione della Dimensionalità.** Al contrario del KNN (che collassa dopo le prime componenti PCA), la Logistic Regression impara ad assegnare pesi proporzionali sfruttando sempre più varianza; con 150 componenti PCA ottiene l'**86.1%**, massimizzando la differenziazione clinica.

### Limitazioni

| Limitazione | Impatto |
|---|---|
| **Dataset sbilanciato** (122 COVID vs 540 Normal) | Pattern PCA sbilanciati verso Normal |
| **Risoluzioni eterogenee** (203px vs 1678px) | Diversi livelli di perdita dopo resize |
| **MSE/PSNR non sono metriche cliniche** | PSNR = 32 dB può ancora avere artefatti diagnostici rilevanti |
| **PCA non supervisionata** | Non ottimizzata per separare le classi patologiche |
| **Analisi SVD su immagine singola** | La comprimibilità varia tra immagini della stessa classe |

### Possibili estensioni

- **Confronto con standard industriali:** a parità di compression ratio, la SVD è competitiva con gli standard (es. JPEG 2000)?
- **t-SNE/UMAP al posto di PCA 2D:** per visualizzare e "sbrogliare" le feature in maniera non-lineare massimizzando i cluster delle classi.
- **Mappa di errore spaziale:** analizzare *dove* si concentra l'errore di ricostruzione sulla radiografia (zone anatomiche critiche vs background)
- **Classificatori avanzati (Deep Learning):** sostituire i modelli classici con ResNet o ViT (Vision Transformes) sfruttando il dataset come Feature Bank.

---

## 7. Conclusioni

| Risultato | Evidenza quantitativa |
|---|---|
| **SVD comprime efficacemente le rx polmonari** | $k = 20$ (15.7% dei dati) → PSNR > 32 dB (qualità diagnostica preservata) |
| **Le radiografie hanno struttura di basso rango** | 90% della varianza in $k \approx 1$–3; 99% in $k \approx 20$–40 |
| **Eckart-Young garantisce ottimalità teorica** | Non esiste approssimazione di rango $k$ migliore della SVD troncata |
| **PCA = SVD su dati centrati** | Stessa base matematica, obiettivi complementari |
| **Le eigenfaces rivelano pattern anatomo-clinici** | PC1: contrasto, PC2: simmetria, PC3: mediastino |
| **PCA mostra separabilità parziale tra classi** | COVID-19/TB vs Normal/Pneumonia si separano su PC1; sovrapposizione residua |
| **SVD + LogReg battono i pixel grezzi** | Classificazione su pixel compressi (k=5) → accuracy **85.7%**, superiore alla baseline degli 85.1% Raw Pixels |
| **PCA è il feature extractor clinico ideale** | 150 componenti → accuracy massima dell'**86.1%** aggirando il Curse of Dimensionality del KNN |
| **Il pre-processing è critico** | Senza correzione tilt e filtraggio, i risultati PCA sarebbero degradati |

