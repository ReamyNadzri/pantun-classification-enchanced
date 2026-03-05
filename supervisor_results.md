# 📊 Model Performance Results

**To:** [Supervisor's Name]  
**Subject:** Quantitative Results & Analysis of The 3 Classification Models

As requested, here is the detailed breakdown of the results and performance metrics for the three machine learning models trained for the Generative Pantun System. All three models were evaluated against unseen testing data from our dataset ([pantun_dataset.json](file:///c:/Users/rahim/pantun-classification-enchanced-2/pantun_dataset.json)).

---

## 1. Topline Metrics Comparison

| Model Architecture | Accuracy | F1-Score | Parameter Size | Training Method |
| :--- | :--- | :--- | :--- | :--- |
| **MalayBERT** (Transformer) | **~59.7%** | **~0.60** | 400+ MB | Fine-tuned (10 epochs) |
| **SVM 90/10 Split** (Baseline) | **~55.1%** | **~0.53** | 2.1 MB | TF-IDF (9,500 features) |
| **TextCNN** (Deep Learning) | **~46.9%** | **~0.46** | 2.6 MB | CNN 1D (20 epochs) |

**Winner:** **MalayBERT** outperformed the others by roughly 5-10% across the board. 

---

## 2. In-Depth Model Analysis

### 🏆 1st Place: MalayBERT
* **Foundation:** Built on top of `mesolitica/bert-base-standard-bahasa-cased` (pre-trained on massive Malaysian text corpuses).
* **The "Why":** As a transformer model, BERT utilizes the "Attention" mechanism. Rather than just hunting for specific keywords like "budi", it evaluates how the words contextually relate to the rest of the stanza. 
* **Quirks:** In live testing, we found MalayBERT struggles with ambiguity. Because traditional Pantun are innately instructional, MalayBERT aggressively defaults to classifying anything containing a moral lesson as **"Nasihat dan Pendidikan" (Advice & Education)**, even if it is technically regarding *Adat* or *Agama*.

### 🥈 2nd Place: Support Vector Machine (SVM)
* **Foundation:** Classic TF-IDF (Term Frequency-Inverse Document Frequency) keyword mapping.
* **The "Why":** SVM relies purely on mathematical word-frequency. Despite being the simplest model, it scored 55%. Why? Because certain pantun themes rely entirely on explicit trigger words. For example, if a pantun contains "Tuhan" or "Masjid", SVM correctly categorizes it as **Agama (Religion)** with extremely high precision.
* **Quirks:** It fails miserably on allegories. If a pantun discusses "Kasih sayang ibu bapaku" (A mother's love), SVM assumes the word "Kasih" means it is **"Percintaan" (Romantic Love)**.

### 🥉 3rd Place: TextCNN
* **Foundation:** A custom Convolutional Neural Network built with PyTorch using 1D convolutions across word vectors.
* **The "Why":** TextCNN performed the weakest (46%). This is generally expected for CNN architectures trained on exceedingly small Natural Language datasets. TextCNN requires massive volumes of data to derive meaningful dimensional kernels, and our dataset sizes for minority categories (like *Teka-Teker* or *Jenaka*) have fewer than 150 samples.
* **Quirks:** It acts like a slightly weaker version of the SVM—depending heavily on keyword feature extraction but struggling with vocabulary sparsity.

---

## 3. Data Imbalance (The Bottleneck)

The primary reason no model has cracked ~75%+ accuracy is **severe data imbalance**. As seen in the generated Confusion Matrices, the models score well (>75% precision) on overrepresented classes but fail on minority classes.

* **Majority Classes (Good Accuracy):** *Nasihat dan Pendidikan* (596 test samples), *Peribahasa* (507 samples), *Budi* (300+ samples).
* **Minority Classes (Poor Accuracy):** *Teka-Teki* (42 test samples), *Jenaka* (53 samples), *Agama* (36 samples).

The models mathematically bias their predictions toward the majority classes (like *Nasihat*) to artificially inflate their overall accuracy scores because guessing those classes simply has a higher probability of being correct.

## 4. Conclusion
Integrating **MalayBERT** was a massive success as it provides true semantic context to our generative platform. To push accuracy past 60% for future versions, the immediate next step must lie in **Data Engineering**—specifically gathering >1,000 new pantuns mathematically dedicated to minority classes (Agama, Teka-Teki, Jenaka) to re-balance the training distribution.
