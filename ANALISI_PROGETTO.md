# Compressione e Analisi di Immagini Mediche tramite SVD e PCA

**Corso:** Statistical Methods — Politecnico di Bari, 1° Anno  
**Dataset:** Lung X-Rays Grayscale (4 classi diagnostiche)

---

## 1. Obiettivo del Progetto

Il progetto applica due tecniche fondamentali dell'algebra lineare — la **Singular Value Decomposition (SVD)** e la **Principal Component Analysis (PCA)** — a un dataset di radiografie polmonari in scala di grigi. Gli obiettivi principali sono:

1. **Compressione delle immagini**: dimostrare come la SVD permetta di approssimare un'immagine con un numero ridotto di componenti, riducendo lo spazio di memorizzazione.
2. **Analisi della qualità**: quantificare il trade-off tra compressione e qualità visiva tramite metriche oggettive (MSE, PSNR).
3. **Riduzione dimensionale e classificazione esplorativa**: utilizzare la PCA per proiettare le immagini in uno spazio a bassa dimensionalità e verificare la separabilità tra le 4 patologie.

---

## 2. Dataset e Classi Diagnostiche

Il dataset contiene radiografie toraciche in scala di grigi suddivise in **4 classi**:

| Classe | Descrizione |
|---|---|
| **Corona Virus Disease** | Radiografie di pazienti affetti da COVID-19 |
| **Normal** | Radiografie di polmoni sani |
| **Pneumonia** | Radiografie con polmonite batterica/virale |
| **Tuberculosis** | Radiografie con tubercolosi polmonare |

Vengono caricate fino a **80 immagini per classe**. Ogni immagine viene ridimensionata a **256 × 256 pixel** e normalizzata nell'intervallo [0, 1].

---

## 3. Pipeline di Analisi (5 Fasi)

### Fase 1 — Esplorazione dei Dati

Si carica un campione per classe e se ne analizzano le proprietà (dimensione, range di intensità, distribuzione dei pixel). Il filtro automatico esclude le **radiografie laterali** (profilo), rilevate attraverso l'analisi del profilo orizzontale di intensità.

![Fase 1 — Campioni e istogrammi per classe](output/fase1_esplorazione.png)

**Osservazioni dagli istogrammi:**
- **COVID-19**: distribuzione concentrata nelle tonalità medio-alte (100–170), polmoni con opacità diffuse.
- **Normal**: distribuzione più uniforme su tutto il range, buon contrasto tra tessuti.
- **Pneumonia**: forte picco nelle tonalità scure (30–80), coerente con addensamenti polmonari.
- **Tuberculosis**: bimodale con picco nelle tonalità chiare (150–190), indicativo di lesioni fibrotiche.

---

### Fase 2 — Pre-elaborazione

Ogni immagine viene sottoposta a:

1. **Conversione in scala di grigi** (canale L)
2. **Ridimensionamento** a 256 × 256 pixel (interpolazione Lanczos)
3. **Normalizzazione** dei pixel in [0, 1]
4. **Rilevamento e correzione dell'inclinazione (tilt)**, basato sulla massimizzazione della varianza del gradiente della proiezione orizzontale. L'algoritmo usa una ricerca a due passi (coarse 2° → fine 0.5°).
5. **Filtraggio** delle immagini corrotte (angoli con padding bianco) e di quelle con inclinazione > 10°.

![Fase 2 — Correzione dell'inclinazione](output/fase2_correzione_tilt.png)

---

### Fase 3 — SVD (Motore Matematico)

La **Singular Value Decomposition** scompone una matrice immagine *A* (256 × 256) come:

$$A = U \cdot \Sigma \cdot V^T$$

dove:
- **U** (256 × 256): vettori singolari sinistri — catturano le strutture lungo le righe
- **Σ** (diagonale): valori singolari σ₁ ≥ σ₂ ≥ … ≥ 0 — quantificano l'"importanza" di ciascuna componente
- **V^T** (256 × 256): vettori singolari destri — catturano le strutture lungo le colonne

L'**approssimazione di rango k** (teorema di Eckart–Young) ricostruisce l'immagine usando solo le prime *k* componenti:

$$A_k = \sum_{i=1}^{k} \sigma_i \cdot \mathbf{u}_i \cdot \mathbf{v}_i^T$$

Con *k* componenti, si memorizzano **k × (256 + 256 + 1)** valori anziché **65.536**, ottenendo un **rapporto di compressione** significativo.

---

### Fase 4 — Ricostruzione e Visualizzazione

#### 4.1 Ricostruzione a diversi livelli di k

![Fase 4 — Ricostruzione SVD con diversi k](output/fase4_ricostruzione_svd.png)

| k | Compressione | PSNR (dB) | Qualità visiva |
|---|---|---|---|
| 1 | 0.8% | 17.3 | Solo luminosità media, nessun dettaglio |
| 5 | 3.9% | 23.8 | Sagoma del torace appena riconoscibile |
| 10 | 7.8% | 27.8 | Strutture principali visibili (costole, mediastino) |
| 20 | 15.7% | 32.4 | Buona ricostruzione, bordi leggermente sfumati |
| 50 | 39.1% | 40.1 | Dettaglio diagnostico preservato |
| 100 | 78.3% | 50.0 | Quasi indistinguibile dall'originale |
| 200 | 100% | 71.5 | Ricostruzione virtualmente perfetta |

> **Risultato chiave**: con sole **20–50 componenti** (su 256 possibili) si preserva oltre il 90% dell'informazione visiva, ottenendo una compressione dell'84–61%.

#### 4.2 Confronto tra classi

![Fase 4 — Confronto SVD tra classi](output/fase4_confronto_classi.png)

Il comportamento della compressione SVD è coerente tra le classi: tutte raggiungono un PSNR > 30 dB con k = 20 e > 48 dB con k = 100. Le classi Pneumonia e Tuberculosis, avendo immagini leggermente più "semplici" in termini di contrasto, raggiungono PSNR più alti a parità di k.

---

### Fase 5 — Analisi Statistica e PCA

#### 5.1 Scree Plot e Decadimento dei Valori Singolari

![Fase 5 — Scree plot](output/fase5_scree_plot.png)

**Varianza cumulativa** (pannello sinistro):
- Tutte le classi raggiungono il **90% di varianza spiegata** con le primissime componenti (k ≈ 1–3).
- Il **95%** viene raggiunto con k ≈ 5–10.
- Il **99%** con k ≈ 20–40.

Questo conferma che le radiografie polmonari hanno una **forte struttura di basso rango**: l'informazione è altamente concentrata nelle prime componenti singolari.

**Decadimento dei valori singolari** (pannello destro): il primo valore singolare è ≈ 130, e i successivi calano rapidamente. Dopo i primi 10–15, i valori singolari sono prossimi a zero.

#### 5.2 Curve MSE e PSNR vs k

![Fase 5 — MSE e PSNR vs k](output/fase5_mse_psnr.png)

- **MSE**: decresce rapidamente nei primi 20–30 componenti e raggiunge valori trascurabili dopo k ≈ 50.
- **PSNR**: cresce logaritmicamente. Il "punto di gomito" si colloca intorno a k = 20–50, dove un ulteriore aumento di k produce miglioramenti marginali.
- Il comportamento è simile tra le classi, ma COVID-19 e Normal presentano MSE leggermente più alti a basso k, indicando una maggiore complessità strutturale.

#### 5.3 PCA — Proiezione 2D e Separabilità

![Fase 5 — PCA scatter plot](output/fase5_pca_scatter.png)

**Scatter plot PCA** (pannello sinistro):
- **PC1** (25.6% varianza) e **PC2** (17.4% varianza) separano parzialmente le classi.
- **Normal** e **Pneumonia** si concentrano nella metà sinistra (PC1 < 0).
- **COVID-19** e **Tuberculosis** si posizionano nella metà destra (PC1 > 0).
- C'è **sovrapposizione** tra le classi, ma la tendenza alla separazione è chiara. Questo suggerisce che SVD/PCA possono supportare la classificazione diagnostica, sebbene un classificatore più avanzato sia necessario.

**Varianza spiegata per componente** (pannello destro):
- Le prime 2 componenti catturano il **43%** della varianza totale.
- Le prime 50 componenti raggiungono circa il **90%**.

#### 5.4 Eigenfaces (Eigen-Xrays)

![Fase 5 — Eigen-Xrays](output/fase5_eigenfaces.png)

Le **prime 10 componenti principali**, visualizzate come immagini (colormap RdBu), rivelano i pattern strutturali dominanti:

| Componente | Varianza | Interpretazione |
|---|---|---|
| **PC1** | 25.6% | Contrasto globale luminosità/sfondo — distingue radiografie chiare da scure |
| **PC2** | 17.4% | Simmetria laterale del torace — differenzia i profili destro/sinistro |
| **PC3** | 5.1% | Pattern mediastinico — variazioni nella zona centrale |
| **PC4** | 4.8% | Strutture costali dominate — pattern della gabbia toracica |
| **PC5–10** | 1.6–3.9% | Dettagli progressivamente più fini (campi polmonari, diaframma, apici) |

---

## 4. Architettura del Codice

Il progetto è organizzato in moduli Python con responsabilità ben separate:

```
Medical Image Compression and Analysis/
├── config.py            # Parametri globali (percorsi, costanti, stile grafici)
├── data_loader.py       # Caricamento, validazione, tilt correction
├── svd_engine.py        # SVD, ricostruzione, metriche (MSE, PSNR)
├── visualization.py     # Plot delle fasi 1, 2, 4
├── src/analysis.py      # Scree plot, MSE/PSNR, PCA, eigenfaces, tabella
├── main.py              # Script orchestratore (esegue le 5 fasi)
├── medical_image_svd_pca.py  # Versione notebook-style (.py con celle %%)
└── output/              # Grafici salvati in PNG (150 dpi)
```

---

## 5. Conclusioni

| Risultato | Evidenza |
|---|---|
| **La SVD è efficace per la compressione** | Con k = 20 (15.7% dei dati), il PSNR supera 32 dB — qualità diagnostica preservata |
| **Le radiografie hanno struttura di basso rango** | Il 90% della varianza è catturato dalle prime 1–3 componenti singolari |
| **La PCA mostra separabilità parziale** | Le 4 classi occupano regioni distinte (ma con sovrapposizione) nello spazio PC1–PC2 |
| **Le eigen-xrays rivelano pattern clinici** | Le prime componenti codificano contrasto globale, simmetria e struttura costale |
| **Il pre-processing è critico** | La correzione del tilt e il filtraggio delle immagini corrotte migliorano la qualità dell'analisi |
