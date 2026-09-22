# Bengali Pitha Classification (PithaBD-24)

Fine-grained image classification of 24 traditional Bangladeshi pitha varieties, using frozen ImageNet backbones for feature extraction and classical machine learning classifiers on top. Built as the final project for **CSE 4112 — Machine Learning Laboratory**, Department of CSE, Khulna University of Engineering \& Technology (KUET).

## Overview

* **24 pitha varieties**, 2,150 unique curated images
* **6 frozen ImageNet backbones** (EfficientNet-B0, DenseNet121, MobileNetV2, Xception, ConvNeXt-Tiny, ResNet50) as feature extractors — no fine-tuning
* **PCA-256** dimensionality reduction, fitted on training data only
* **4 classical classifiers** per backbone (Logistic Regression, SVM/RBF, Random Forest, XGBoost) — 24 configurations benchmarked
* **Hand-selected, leakage-controlled test set** of 352 images
* Each class carries cultural/nutritional metadata (district, division, cooking method, ingredients, recipe summary, nutrition estimates)

## Dataset

**2,197 images collected → 47 exact duplicates removed (MD5) → 2,150 unique images**, from four sources:

|Source|Images|Share|
|-|-|-|
|YouTube (video frames)|≈1,405|63.9%|
|Google Images|≈520|23.7%|
|Smartphone (team, exact)|140|6.4%|
|Facebook|≈132|6.0%|
|**Total**|**2,197**|**100%**|

Collected 01/08/2026 – 01/09/2026 via Bengali and English name searches. Web-source counts are estimates; the smartphone count is exact.

* **Coverage:** 7 divisions (Barishal 7 varieties, Sylhet 5, Khulna 4, Chittagong 3, Mymensingh 3, Rajshahi 1, Rangpur 1) and 8 cooking methods (deep-fried dominates at 12 varieties, pan-cooked 4, steamed 3, the rest 1 each).
* **Class balance:** 64–111 unique images per class (patishapta lowest at 64, narkel\_puli\_pitha highest at 111). Imbalance ratio ≈ 1.7:1 — mild.

### Curation and labelling protocol

|Included|Excluded|
|-|-|
|One dominant variety in frame|Mixed plates|
|Recognisable|Unconfirmed identity|
|Shorter side ≥ 140 px|Drawings / AI-generated images|
|Readable|Byte-identical duplicates|

Folder name = label. Two annotators reviewed the set; unclear images were discarded rather than guessed at.

### Metadata

Every class has an entry (`pitha\_metadata.json`-style) with: display name and alternative names, district and division, cooking method and key ingredients, a recipe summary, and per-class nutrition estimates (calories, protein, carbs, fat).

## Data Split \& Leakage Control

|Partition|Originals|Share of 2,150|Augmented copies|Total files|
|-|-|-|-|-|
|Training|1,588|≈75.0%|7,940|9,528|
|Validation|210|≈10.0%|—|210|
|Test (hand-selected)|352|≈15.0%|—|352|

The 352 test images were chosen by hand **first**; the remaining pool was then split per class with seed 42 into train/validation. All notebooks reproduce the identical split (fingerprint `93e293cf08d9e44e`).

**Augmentation** (training images only, 5 label-preserving copies per original):

* Horizontal flip, p = 0.5
* Rotation ±15°
* Brightness / contrast / saturation jitter, ±20%
* Random crop and resize, 85–100% of area
* Gaussian noise, σ = 0.01

**Leakage controls, verified rather than assumed:**

* Test set isolated before augmentation
* No duplicate image across partitions; no original image in two partitions
* Augmented copies grouped with their source image (1,588 groups) so a photo and its augmented copies never split across a cross-validation fold — all 10 folds checked and passed
* Every original scored exactly once across cross-validation

## Method

1. Each image is passed through a **frozen** backbone (ImageNet weights, classification head removed, global average pooling) to produce a fixed-length feature vector. No backbone weights are updated.
2. **PCA to 256 components**, fitted on the training split only.
3. Four classical classifiers are trained on the PCA-reduced features per backbone: **Logistic Regression** (SMO C=1, γ=0.01), **SVM (RBF kernel)**, **Random Forest** (500 trees), **XGBoost** (500 trees).
4. The classifier is selected on validation accuracy; the test set (352 hand-selected images) is scored once.

**Environment:** Google Colab, NVIDIA Tesla T4 GPU, Python 3.13, TensorFlow 2.20, scikit-learn 1.4.2, XGBoost 2.1.

## Results

|Backbone|Classifier|Top-1|Top-3|Top-5|Macro F1|κ|MCC|ROC-AUC|Errors /352|
|-|-|-|-|-|-|-|-|-|-|
|**EfficientNet-B0**|**Logistic Regression**|**82.67%**|**95.74%**|**98.01%**|**0.820**|**0.819**|**0.820**|**0.989**|**61**|
|DenseNet121|Logistic Regression|80.11%|92.33%|95.74%|0.793|0.792|0.794|0.985|70|
|MobileNetV2|Logistic Regression|79.26%|92.05%|96.31%|0.785|0.783|0.784|0.983|73|
|Xception|SVM (RBF)|79.26%|92.61%|97.16%|0.782|0.783|0.785|0.987|73|
|ConvNeXt-Tiny|Logistic Regression|77.56%|92.33%|95.74%|0.767|0.766|0.766|0.985|79|
|ResNet50|Logistic Regression|76.14%|90.91%|95.17%|0.751|0.751|0.752|0.982|84|

**Best configuration: EfficientNet-B0 + Logistic Regression** — 82.67% top-1 / 95.74% top-3 / 98.01% top-5, 61 errors out of 352 test images.

## Limitations

* Modest dataset size (2,150 images across 24 classes).
* \~94% of images are sourced from the web rather than the team's own photography; smartphone photos cover a minority of the varieties.
* Possible near-duplicates surviving from video-frame extraction (exact MD5 dedup was applied, but visually-near duplicates from adjacent video frames are not caught by it).
* No inter-annotator agreement score was computed for the labelling protocol.
* Nutrition values in the metadata are estimates, not lab-verified.

## Responsible Use

* Food images only — no personal data was collected.
* Web-sourced images are used for non-commercial academic purposes and are not redistributed.
* Nutrition figures are estimates and **not intended as clinical or dietary advice**.

## Team

**Group 05 — CSE 4112, Department of Computer Science and Engineering, KUET**

|Name|Roll|
|-|-|
|Shahriar Aziz Khan|2107034|
|MD. Abdullah Sheikh|2107035|
|Md. Zunaied Nudar|2107041|
|Abdullah Saeid Bin Omar|2107053|
|Mustafizur Rahman|2107058|
|Mamunar Rahman|1807051|



## License

MIT — see [LICENSE](LICENSE). This covers the code (notebooks, scripts, metadata schema). It does **not** extend to the web-sourced images themselves (YouTube frames, Google Images, Facebook) — those remain subject to their original sources' rights and are covered by the non-commercial, non-redistribution terms in [Responsible Use](#responsible-use), not by the MIT grant.

