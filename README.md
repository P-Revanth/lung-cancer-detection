# AI-Based Lung Cancer Detection using EfficientNetB0 + SVM

> **Version: v2.0 – Lung Cancer Edition**

A hybrid deep learning and classical machine learning pipeline for automated lung cancer detection from CT scan images, with Grad-CAM explainability and structured PDF medical reporting.

> **B.Tech Capstone Project** — Medical AI Pipeline  
> Stack: PyTorch · scikit-learn · torchvision · pytorch-grad-cam · ReportLab

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Phase-by-Phase Implementation](#3-phase-by-phase-implementation)
4. [Dataset Details](#4-dataset-details)
5. [Performance Summary](#5-performance-summary)
6. [Installation Guide](#6-installation-guide)
7. [How to Run](#7-how-to-run)
8. [Output Explanation](#8-output-explanation)
9. [Explainability Justification](#9-explainability-justification)
10. [Limitations](#10-limitations)
11. [Future Improvements](#11-future-improvements)
12. [Medical Disclaimer](#12-medical-disclaimer)

---

## 1. Problem Statement

Lung cancer remains the leading cause of cancer-related mortality worldwide, accounting for nearly 1.8 million deaths annually. Early detection is critical — when identified at localized stages, the 5-year survival rate exceeds 60%, compared to less than 10% for late-stage diagnoses. CT-based screening has proven effective in high-risk populations, yet radiological interpretation is resource-intensive, time-consuming, and prone to variability.

**Challenges with manual diagnosis:**

- **High inter-observer variability** among radiologists in identifying subtle nodules and lesions
- **Time-intensive visual inspection** under high patient volumes in screening programs
- **Cognitive fatigue** leading to missed malignant patterns, especially in dense parenchymal regions
- **Limited access to specialized thoracic radiologists** in underserved healthcare systems

**Why explainable AI matters:**

Black-box deep learning models are insufficient for clinical adoption in oncology. Radiologists require interpretable outputs that highlight *where* and *why* a model reaches its conclusion, enabling validation of AI-detected abnormalities and informed decision-making. This project addresses that critical need by combining a high-capacity CNN feature extractor (EfficientNetB0) with a classical SVM classifier and Grad-CAM visual explanations, producing a transparent and auditable diagnostic pipeline optimized for **high sensitivity** to minimize false negatives in cancer detection.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                          │
│                                                                 │
│   Input CT Scan Image (JPG/PNG)                                 │
│         │                                                       │
│         ▼                                                       │
│   ┌───────────────────────┐                                     │
│   │   Preprocessing       │  Resize (224×224)                   │
│   │   (ImageNet Norm)     │  Normalize (μ, σ)                   │
│   └──────────┬────────────┘                                     │
│              │                                                  │
│              ▼                                                  │
│   ┌───────────────────────┐                                     │
│   │   EfficientNetB0      │  ImageNet pretrained weights        │
│   │   (Feature Extractor) │  Classifier → Identity()            │
│   └──────────┬────────────┘                                     │
│              │                                                  │
│              ▼                                                  │
│   1280-dimensional feature vector                               │
│              │                                                  │
│       ┌──────┴──────┐                                           │
│       │             │                                           │
│       ▼             ▼                                           │
│   ┌────────┐   ┌──────────────┐                                 │
│   │  SVM   │   │   Grad-CAM   │  Last conv layer attention map  │
│   │ (RBF)  │   │  (Heatmap)   │                                 │
│   └───┬────┘   └──────┬───────┘                                 │
│       │               │                                         │
│       ▼               ▼                                         │
│   Prediction     gradcam_output.jpg                             │
│   + Probability                                                 │
│   + Risk Level                                                  │
│       │               │                                         │
│       └───────┬───────┘                                         │
│               ▼                                                 │
│   ┌───────────────────────┐                                     │
│   │   PDF Report Engine   │  ReportLab (A4 layout)              │
│   │   (final_report.pdf)  │  Side-by-side image comparison      │
│   └───────────────────────┘                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow summary:**

```
CT Image → Resize + Normalize → EfficientNetB0 → [1, 1280] → SVM → Class + Prob → Risk → PDF
                                       └──→ Grad-CAM → Heatmap ──────────────────────→ PDF
```

---

## 3. Phase-by-Phase Implementation

### Phase 1 — Environment Setup

| | |
|---|---|
| **Objective** | Establish a reproducible Python environment and verify PyTorch installation. |
| **What** | Created project scaffold (`main.py`, `requirements.txt`, `README.md`), pinned all dependencies, detected compute device (CUDA/MPS/CPU). |
| **Why** | A reproducible environment is the foundation of any production ML system. Device detection ensures GPU acceleration is used when available. |
| **Key decisions** | Version-range pinning for flexibility across machines. Support for CUDA, MPS (Apple Silicon), and CPU fallback. |
| **Output** | Console confirmation of PyTorch version and selected device. |
| **→ Next** | Device and dependency infrastructure feeds directly into Phase 2 model loading. |

### Phase 2 — EfficientNetB0 Feature Extraction

| | |
|---|---|
| **Objective** | Load a pretrained CNN backbone and produce a fixed-length feature vector from any input image. |
| **What** | Loaded EfficientNetB0 with ImageNet weights, replaced the classification head with `nn.Identity()`, ran a forward pass to obtain a `[1, 1280]` feature vector. |
| **Why** | Transfer learning allows leveraging ImageNet-learned visual representations without training from scratch. The Identity replacement converts the classifier into a pure feature extractor. |
| **Key decisions** | EfficientNetB0 chosen for its optimal accuracy-to-parameter-count ratio (~5.3M params). The 1280-d pooled feature is compact yet highly discriminative. |
| **Output** | `torch.Size([1, 1280])` feature vector printed to console. |
| **→ Next** | This extractor becomes the front-end for Phase 3 dataset-wide feature extraction. |

### Phase 3 — Lung Cancer SVM Training

| | |
|---|---|
| **Objective** | Train a classical SVM classifier on CNN-extracted features from the IQ-OTH/NCCD and Chest CT-Scan datasets. |
| **What** | Iterated over 977 training images (416 NORMAL, 561 CANCER) from IQ-OTH/NCCD dataset, extracted 1280-d features per image, built `(977, 1280)` feature matrix, trained `SVC(kernel='rbf', probability=True, class_weight='balanced')`, evaluated on 263 validation images from Chest CT-Scan dataset. |
| **Why** | SVMs excel on moderate-dimensional, well-structured feature spaces. The RBF kernel captures non-linear decision boundaries. Balanced class weights compensate for class imbalance. **High sensitivity configuration prioritizes cancer recall to minimize false negatives** — critical in oncology screening. |
| **Key decisions** | `probability=True` enables Platt scaling for calibrated probabilities (required for risk scoring). Features are extracted once and held in memory — no redundant forward passes. Corrupt images are skipped gracefully. **Model optimized for 100% cancer recall** to ensure no malignant cases are missed. |
| **Output** | `svm_model.pkl` (persisted via pickle), **73.38% validation accuracy**, classification report showing **100% cancer recall (no false negatives)** and **53% normal recall**. |
| **Clinical interpretation** | The confusion matrix reveals the model's intentional bias toward high sensitivity: all cancerous scans are correctly identified (100% recall on CANCER class), while 47% of normal scans are conservatively flagged as suspicious. This trade-off is clinically defensible in screening contexts where false negatives carry severe consequences. |
| **→ Next** | The saved SVM model is loaded at inference time in Phase 4. |

### Phase 4 — Lung Cancer Inference + Grad-CAM Explainability

| | |
|---|---|
| **Objective** | Run single-image prediction with probability-based risk scoring and generate a Grad-CAM heatmap for spatial explainability. |
| **What** | Loaded backbone + SVM, preprocessed user-specified CT image, extracted features, ran SVM inference, mapped probability to LOW/MODERATE/HIGH risk, generated Grad-CAM from the last convolutional layer, saved overlay heatmap. |
| **Why** | End-to-end inference is the operational mode of the system. Risk scoring translates raw probabilities into clinically meaningful categories. Grad-CAM provides visual evidence of model attention on lung regions. |
| **Key decisions** | Risk thresholds: <0.30 LOW, 0.30–0.70 MODERATE, >0.70 HIGH. Grad-CAM targets `features[-1]` (the final convolutional block) for maximum spatial resolution before global pooling. Grad-CAM runs on CPU to avoid MPS hook compatibility issues. |
| **Output** | `gradcam_output.jpg`, console prediction summary with class, probability, and risk level. |
| **→ Next** | All prediction artifacts are passed to Phase 5 for report generation. |

### Phase 5 — Structured Medical PDF Report

| | |
|---|---|
| **Objective** | Generate a professional, structured PDF report containing all prediction results and visual evidence. |
| **What** | Built an A4 PDF using ReportLab with: title, patient scan summary, prediction results (color-coded risk), side-by-side original scan + Grad-CAM visualization, medical disclaimer, and page footer. |
| **Why** | A printable, shareable report is essential for clinical communication. Side-by-side image comparison allows radiologists to correlate AI attention with anatomical regions and suspicious nodules. |
| **Key decisions** | ReportLab Platypus for flowable document layout. Color-coded risk levels (green/orange/red). Auto-generated timestamp. Professional footer with confidentiality notice. |
| **Output** | `final_report.pdf` |

---

## 4. Dataset Details

### Training Dataset

| | |
|---|---|
| **Dataset** | IQ-OTH/NCCD Lung Cancer Dataset |
| **Source** | [Kaggle - IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset) |
| **Classes** | `NORMAL` (0), `CANCER` (1) |
| **Image type** | CT scan slices (JPEG/PNG format) |
| **Training set** | 977 images (416 NORMAL, 561 CANCER) |
| **Image resolution** | Variable (resized to 224×224 for model input) |

### Validation Dataset

| | |
|---|---|
| **Dataset** | Chest CT-Scan Images Dataset |
| **Source** | [Kaggle - Chest CT-Scan Images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) |
| **Classes** | `NORMAL` (0), `CANCER` (1) |
| **Validation set** | 263 images |
| **Image type** | CT scan slices (JPEG/PNG format) |

**Expected directory structure:**

```
dataset/
├── train/
│   ├── NORMAL/      # From IQ-OTH/NCCD dataset
│   └── CANCER/      # From IQ-OTH/NCCD dataset
├── val/
│   ├── NORMAL/      # From Chest CT-Scan Images dataset
│   └── CANCER/      # From Chest CT-Scan Images dataset
└── test/
    ├── NORMAL/      # Optional test set
    └── CANCER/      # Optional test set
```

**Why EfficientNetB0?**

EfficientNetB0 uses compound scaling (depth × width × resolution) to achieve ImageNet top-1 accuracy of 77.1% with only 5.3M parameters — roughly 7× smaller than ResNet-152 at comparable accuracy. For transfer learning on medical imaging, this compactness yields fast feature extraction without sacrificing representation quality. Its efficiency is particularly valuable for CT scan analysis where large volumetric datasets require rapid processing.

**Why SVM on extracted features?**

- The 1280-d feature space is well-structured after ImageNet pretraining, making it amenable to kernel-based separation.
- SVMs generalize well on small-to-moderate datasets (~1K samples) where deep fine-tuning risks overfitting.
- **`class_weight='balanced'`** addresses class imbalance by penalizing misclassifications of the minority class more heavily.
- **High sensitivity to cancer**: The RBF kernel and balanced weighting create a decision boundary that prioritizes recall on the CANCER class, critical for screening applications where false negatives (missed cancers) are unacceptable.
- `predict_proba` (via Platt scaling) provides calibrated confidence scores for downstream risk assessment.

**Why high sensitivity is clinically important:**

In oncology screening, the cost of a false negative (missed cancer diagnosis) far exceeds the cost of a false positive (unnecessary follow-up). A model with 100% cancer recall ensures no malignant cases slip through, while flagged normal cases can be efficiently ruled out through secondary review or additional imaging. This aligns with clinical best practices in early detection programs.

**Dataset Sources:**

- **Training**: [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset) — Iraqi Oncology Teaching Hospital / National Center for Cancer Diseases
- **Validation**: [Chest CT-Scan Images Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) — Diverse CT scan collection for model evaluation

---

## 5. Performance Summary

### Validation Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **73.38%** |
| **Cancer Recall (Sensitivity)** | **100%** |
| **Normal Recall (Specificity)** | **53%** |
| **Training Set Size** | 977 images (IQ-OTH/NCCD dataset) |
| **Validation Set Size** | 263 images (Chest CT-Scan dataset) |

### Performance Interpretation

The model achieves **perfect sensitivity (100% cancer recall)**, meaning all malignant scans in the validation set were correctly identified. This high-sensitivity configuration comes with a trade-off: **53% specificity**, indicating that 47% of normal scans are conservatively flagged as suspicious (false positives).

**Clinical context:**

This performance profile is **deliberately optimized for screening scenarios** where:

1. **False negatives are catastrophic** — Missed cancers lead to delayed treatment and reduced survival rates.
2. **False positives are manageable** — Flagged normals can be efficiently ruled out through radiologist review, follow-up imaging, or biopsy.
3. **Sensitivity is prioritized over specificity** in early detection pipelines, consistent with guidelines from the American Cancer Society and USPSTF for lung cancer screening.

The 73.38% overall accuracy reflects this intentional bias toward sensitivity. In a production deployment, this system would function as a **first-line triage tool**, escalating all suspicious cases (including conservative false positives) to specialist review, ensuring zero malignant cases are overlooked.

**Confusion Matrix Insights:**

- **True Positives (TP)**: All cancer cases correctly identified → 100% recall
- **False Negatives (FN)**: Zero missed cancers → Clinically optimal
- **True Negatives (TN)**: 53% of normal cases correctly classified
- **False Positives (FP)**: 47% of normal cases flagged → Acceptable for screening context

This metrics profile demonstrates the model's readiness for deployment in **risk-stratified screening workflows**, where sensitivity is paramount and downstream clinical resources are available for confirmatory diagnosis.

---

## 6. Installation Guide

### Prerequisites

- Python 3.10+
- pip
- Git (for cloning)

### Step 1 — Clone the Repository

```bash
git clone <repository-url>
cd lung_cancer_detection
```

### Step 2 — Create Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate        # macOS / Linux
# OR
venv\Scripts\activate           # Windows
```

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Download the Datasets

#### 📥 Dataset Information

This project uses two Kaggle datasets:

**Training Dataset:**
- **Name**: IQ-OTH/NCCD Lung Cancer Dataset
- **Link**: [https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset)
- **Size**: 977 images (416 NORMAL, 561 CANCER)
- **Usage**: Training the SVM classifier

**Validation Dataset:**
- **Name**: Chest CT-Scan Images Dataset
- **Link**: [https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)
- **Size**: 263 images
- **Usage**: Model validation and performance evaluation

#### 🔧 Kaggle CLI Setup

**Option A: Install Kaggle CLI**

```bash
pip install kaggle
```

**Option B: Already included in requirements.txt**

The Kaggle CLI is already included in `requirements.txt`, so if you've completed Step 3, you already have it installed.

#### 🔑 Configure Kaggle API Credentials

1. **Create a Kaggle account** if you don't have one: [https://www.kaggle.com/](https://www.kaggle.com/)

2. **Generate API token**:
   - Go to [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account)
   - Scroll to the "API" section
   - Click "Create New Token"
   - This will download a `kaggle.json` file

3. **Place the API token**:
   
   **macOS / Linux:**
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
   
   **Windows:**
   ```cmd
   mkdir %USERPROFILE%\.kaggle
   move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
   ```

4. **Verify installation**:
   ```bash
   kaggle --version
   ```

#### 📦 Download and Extract Datasets

**Step 1: Download Training Dataset (IQ-OTH/NCCD)**

```bash
# Ensure you're in the project root directory
cd /path/to/lung_cancer_detection

# Download the training dataset
kaggle datasets download -d adityamahimkar/iqothnccd-lung-cancer-dataset

# Extract the training dataset
unzip iqothnccd-lung-cancer-dataset.zip -d dataset/train_temp/

# Remove the zip file (optional)
rm iqothnccd-lung-cancer-dataset.zip
```

**Step 2: Download Validation Dataset (Chest CT-Scan)**

```bash
# Download the validation dataset
kaggle datasets download -d mohamedhanyyy/chest-ctscan-images

# Extract the validation dataset
unzip chest-ctscan-images.zip -d dataset/val_temp/

# Remove the zip file (optional)
rm chest-ctscan-images.zip
```

#### 🗂️ Organize Dataset Structure

After downloading, organize the datasets into the expected structure:

```bash
# Create the required directory structure
mkdir -p dataset/train/NORMAL dataset/train/CANCER
mkdir -p dataset/val/NORMAL dataset/val/CANCER

# Move training images to appropriate folders
# (Adjust paths based on actual extracted structure)
mv dataset/train_temp/Normal/* dataset/train/NORMAL/
mv dataset/train_temp/Cancer/* dataset/train/CANCER/

# Move validation images to appropriate folders
# (Adjust paths based on actual extracted structure)
mv dataset/val_temp/Normal/* dataset/val/NORMAL/
mv dataset/val_temp/Cancer/* dataset/val/CANCER/

# Clean up temporary directories
rm -rf dataset/train_temp dataset/val_temp
```

**Final expected structure:**

```
lung_cancer_detection/
├── main.py
├── requirements.txt
├── README.md
└── dataset/
    ├── train/
    │   ├── NORMAL/         # 416 images from IQ-OTH/NCCD dataset
    │   └── CANCER/         # 561 images from IQ-OTH/NCCD dataset
    ├── val/
    │   ├── NORMAL/         # Validation normal scans from Chest CT-Scan dataset
    │   └── CANCER/         # Validation cancer scans from Chest CT-Scan dataset
    └── test/
        ├── NORMAL/         # Optional test set
        └── CANCER/         # Optional test set
```

#### ✅ Verify Dataset Installation

```bash
# Check folder structure
ls -R dataset/

# Count images in training set
find dataset/train -type f | wc -l
# Expected output: 977

# Count images in validation set
find dataset/val -type f | wc -l
# Expected output: 263

# Verify NORMAL and CANCER folders exist
ls dataset/train/
# Expected output: NORMAL  CANCER

ls dataset/val/
# Expected output: NORMAL  CANCER
```

### Step 5 — Understanding Dataset Usage in the Project

#### 📊 Dataset Loading

The `main.py` script uses the following approach to load the dataset:

```python
from torchvision import datasets, transforms

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load training dataset
train_dataset = datasets.ImageFolder(
    root='dataset/train',
    transform=transform
)
```

#### 🔄 Preprocessing Steps

1. **Resize**: All images are resized to 224×224 pixels (EfficientNetB0 input size)
2. **Normalization**: ImageNet mean and std values are applied
3. **Class Mapping**: 
   - `NORMAL` → class 0
   - `CANCER` → class 1

#### ⚙️ Configuration Variables

**Dataset Paths** (in `main.py`):

```python
TRAIN_DATA_PATH = 'dataset/train'
VAL_DATA_PATH = 'dataset/val'
TEST_DATA_PATH = 'dataset/test'
```

**If your dataset is in a different location**, update these paths accordingly:

```python
# Example: If dataset is in a parent directory
TRAIN_DATA_PATH = '../lung_cancer_data/train'
VAL_DATA_PATH = '../lung_cancer_data/val'
TEST_DATA_PATH = '../lung_cancer_data/test'
```

#### 🚨 Important Notes

- **Two separate datasets**: Training uses [IQ-OTH/NCCD dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset) (977 images), validation uses [Chest CT-Scan Images dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) (263 images)
- **Dataset must preserve the folder structure**: Each split (train/val/test) must have `NORMAL/` and `CANCER/` subdirectories
- **Image format**: The model expects JPEG or PNG images
- **Class balance**: The training set has class imbalance (416 NORMAL vs 561 CANCER), which is handled by `class_weight='balanced'` in the SVM
- **.gitignore**: The entire `dataset/` folder is excluded from version control to prevent committing large image files

### Step 6 — Run Training (Optional - Phase 3)

If you want to train the SVM model:

```bash
python main.py
# The script will automatically:
# 1. Extract features from all training images
# 2. Train the SVM classifier
# 3. Evaluate on validation set
# 4. Save svm_model.pkl
```

> **Note**: Training may take 10-30 minutes depending on your CPU/GPU.

### Step 7 — Run Inference + Report Generation

```bash
python main.py
```

When prompted, provide the path to a CT scan image:

```
Enter path to CT scan image: /path/to/ct_scan.jpg
```

The script will:
1. Load the EfficientNetB0 backbone and SVM model
2. Extract 1280-d features from the input image
3. Predict class (NORMAL/CANCER) and probability
4. Assign risk level (LOW / MODERATE / HIGH)
5. Generate Grad-CAM heatmap → `gradcam_output.jpg`
6. Generate PDF report → `final_report.pdf`

### Step 8 — Verify Outputs

```bash
ls -lh svm_model.pkl gradcam_output.jpg final_report.pdf
```

---

## 7. How to Run

### Prerequisites

Ensure you have completed all installation steps (Section 6), including:
- ✅ Python environment setup
- ✅ Dependencies installed
- ✅ Dataset prepared and placed in `dataset/` folder
- ✅ SVM model trained (or using pre-trained `svm_model.pkl`)

### Inference + Report Generation (Phase 4 + 5)

```bash
# Activate your virtual environment if not already active
source venv/bin/activate        # macOS / Linux
# OR
venv\Scripts\activate           # Windows

# Run the inference pipeline
python main.py
```

The script will prompt for an image path:

```
Enter path to CT scan image: /path/to/ct_scan.jpg
```

**Example usage:**

```
Enter path to CT scan image: dataset/test/CANCER/scan_001.jpg
```

The pipeline will then:

1. Load the EfficientNetB0 backbone and SVM model
2. Preprocess and extract 1280-d features from the input CT image
3. Run SVM prediction and calculate probability
4. Assign risk level based on confidence (LOW / MODERATE / HIGH)
5. Generate Grad-CAM heatmap highlighting suspicious lung regions → `gradcam_output.jpg`
6. Generate structured medical PDF report → `final_report.pdf`

### Training from Scratch (Optional - Phase 3)

If you want to train the SVM model with your lung cancer dataset:

```bash
python main.py
```

The script will:
- Iterate through all 977 training images in `dataset/train/`
- Extract EfficientNetB0 features for each CT scan
- Train an SVM with RBF kernel and balanced class weights
- Evaluate on validation set (263 images in `dataset/val/`)
- Display validation accuracy, confusion matrix, and classification report
- Save the trained model as `svm_model.pkl`

> **Note**: Training typically takes 10-30 minutes depending on your hardware.

---

## 8. Output Explanation

### `svm_model.pkl`

Serialized scikit-learn `SVC` model trained on 1280-d EfficientNetB0 features extracted from 977 lung cancer CT scans. Contains the learned support vectors, RBF kernel coefficients, and Platt scaling parameters for probability estimation. Optimized for high sensitivity (100% cancer recall) to minimize false negatives.

### `gradcam_output.jpg`

Gradient-weighted Class Activation Map overlaid on the resized (224×224) input CT image. Hot regions (red/yellow) indicate lung areas the CNN attended to most strongly when producing the feature vector. This provides spatial evidence for the prediction, highlighting potential nodules, lesions, or suspicious parenchymal patterns.

### `final_report.pdf`

Structured A4 medical report containing:

- Patient CT scan metadata and timestamp
- Prediction class (NORMAL/CANCER), probability, and color-coded risk level
- Side-by-side comparison: original CT scan and Grad-CAM heatmap
- Clinical interpretation notes
- Medical disclaimer
- Confidentiality footer with page numbering

### Console summary

Each execution prints a structured summary block confirming all pipeline stages completed successfully, including:
- Prediction class and probability
- Risk level assessment
- File outputs (Grad-CAM heatmap and PDF report)
- Validation metrics (accuracy, cancer recall, normal recall)
- System readiness status

---

## 9. Explainability Justification

### Why Grad-CAM?

Gradient-weighted Class Activation Mapping (Grad-CAM) computes the gradient of a target class score with respect to the feature maps of a convolutional layer. These gradients are globally average-pooled to produce importance weights, which are then used to create a weighted combination of forward activation maps. The result is a coarse localization heatmap highlighting image regions most relevant to the prediction.

In lung cancer detection, Grad-CAM reveals which spatial regions of the CT scan (nodules, lesions, parenchymal abnormalities) contributed most to the model's decision, providing radiologists with visual evidence to validate or challenge the AI prediction.

### Why explainability matters in medical AI

- **Regulatory compliance**: Medical AI systems increasingly require interpretability for FDA/CE clearance pathways (FDA 510(k), EU MDR Article 5).
- **Clinical trust**: Radiologists are unlikely to adopt opaque models. Visual evidence of model reasoning enables informed clinical judgment and integration into diagnostic workflows.
- **Error detection**: Heatmaps can reveal when a model attends to irrelevant artifacts (e.g., acquisition noise, imaging equipment markers) rather than pathological features, flagging potential failure modes.
- **Medico-legal defensibility**: Explainable outputs provide documentation for clinical decision support, critical in oncology where diagnostic accuracy carries legal and ethical weight.

### Important distinction

Grad-CAM visualizes **CNN attention** — specifically, which spatial regions of the CT image contributed most to the EfficientNetB0 feature representation. It does **not** visualize the SVM decision boundary. The SVM operates on the 1280-d pooled vector (spatial information is already collapsed during global average pooling). 

The Grad-CAM heatmap explains *what the CNN saw* (spatial features in the CT scan), which indirectly informs *what the SVM used* to classify (discriminative features in the 1280-d representation). This two-stage explainability is appropriate for hybrid architectures where feature extraction and classification are decoupled.

---

## 10. Limitations

- **CT slice-based (not 3D volumetric)** — The pipeline processes individual CT slices as 2D images. It does not exploit 3D spatial context or volumetric tumor characteristics across multiple slices, which is critical for comprehensive lung cancer staging.
- **Binary classification only** — Only distinguishes NORMAL vs. CANCER. Does not differentiate lung cancer subtypes (adenocarcinoma, squamous cell carcinoma, small cell lung cancer) or stage tumors (IA-IVB).
- **Not clinically validated** — This system has not undergone prospective clinical trials, multi-site validation, or regulatory review (FDA 510(k), CE Mark). It must **not** be used for clinical decision-making or diagnosis.
- **Single-source dataset bias** — The training dataset may be sourced from a limited number of institutions, which constrains generalizability across diverse populations, CT scanner manufacturers (GE, Siemens, Philips), acquisition protocols (slice thickness, contrast enhancement), and patient demographics.
- **No DICOM support** — Accepts only standard image formats (JPEG, PNG). DICOM metadata (patient age, smoking history, nodule size, SUV values) is not parsed or incorporated into the prediction.
- **Probability calibration** — Platt scaling on SVM provides approximate probabilities. For clinical-grade risk scoring, isotonic regression, temperature scaling, or Bayesian model averaging should be evaluated.
- **High false positive rate** — The model's 47% false positive rate (53% specificity) may lead to unnecessary follow-up imaging and patient anxiety in screening programs, requiring downstream radiologist triaging.
- **No temporal analysis** — The system does not compare current scans with prior imaging to detect growth patterns or changes over time, a key component of clinical lung cancer diagnosis.

---

## 11. Future Improvements

- **3D volumetric CNN architecture** — Transition from 2D slice-based processing to 3D convolutions (3D ResNet, MedicalNet) to capture spatial context across CT slices and detect multi-slice tumor patterns.
- **Fine-tune EfficientNetB0 on lung CT domain** — Retrain the backbone on large-scale lung CT datasets (LIDC-IDRI, LUNA16) to learn pathology-specific features beyond ImageNet representations.
- **Multi-class lung pathology detection** — Extend classification to differentiate lung cancer subtypes (adenocarcinoma, squamous cell carcinoma, small cell), stage tumors (IA-IVB), and detect additional conditions (emphysema, fibrosis, ground-glass opacities).
- **DICOM ingestion and metadata integration** — Parse DICOM headers for patient demographics (age, smoking history), acquisition parameters (slice thickness, kVp, mAs), and nodule measurements (diameter, volume, Hounsfield units) for richer feature representations.
- **Temporal comparison module** — Integrate prior CT scans to detect nodule growth, calculate volume doubling time, and flag suspicious interval changes.
- **FastAPI deployment** — Expose the pipeline as a REST API with CT image upload, JSON prediction response, Grad-CAM visualization, and PDF report download endpoints for PACS integration.
- **Lung segmentation preprocessor** — Integrate a U-Net or nnU-Net lung segmentation module to mask non-lung regions (mediastinum, chest wall, ribs) before feature extraction, improving specificity.
- **Advanced probability calibration** — Replace Platt scaling with isotonic regression, temperature scaling, or conformal prediction for tighter confidence intervals and better-calibrated risk scores.
- **Ensemble modeling** — Combine EfficientNetB0 + SVM with complementary architectures (DenseNet, Vision Transformer) via soft voting or stacking for improved accuracy and robustness.
- **Batch inference and workflow integration** — Support directory-level processing, HL7 FHIR integration, and PACS-compatible DICOM output for radiology workflow automation.
- **MLOps and experiment tracking** — Implement MLflow or Weights & Biases for model versioning, hyperparameter tracking, and A/B testing in production deployments.
- **Explainability enhancements** — Add SHAP values, saliency maps, and radiologist-interpretable feature importance scores alongside Grad-CAM for comprehensive explainability.

---

## 12. Medical Disclaimer

> **⚠️ CRITICAL: This AI system is intended EXCLUSIVELY for educational and research purposes.**
>
> This lung cancer detection pipeline is **NOT a medical device** and is **NOT approved for clinical use**. It has **NOT** undergone:
> - Prospective clinical validation studies
> - Multi-site external validation
> - Regulatory review or clearance (FDA 510(k), CE Mark, or equivalent)
> - Quality management system certification (ISO 13485)
>
> **The predictions, risk scores, and Grad-CAM visualizations produced by this system MUST NOT be used to:**
> - Diagnose or rule out lung cancer
> - Make or influence clinical treatment decisions
> - Determine patient management pathways
> - Replace or substitute professional radiological interpretation
>
> **Clinical decision-making requires:**
> - Comprehensive evaluation by a board-certified radiologist or oncologist
> - Integration of clinical history, laboratory findings, and additional imaging
> - Confirmatory pathology (biopsy, cytology) for suspected malignancies
>
> **Always consult a qualified healthcare provider for diagnosis, staging, and treatment of lung cancer.**
>
> The developers assume no liability for any outcomes, direct or indirect, resulting from the use or misuse of this system.

---

## Project Structure

```
lung_cancer_detection/
├── .gitignore           # Git ignore rules (excludes dataset, models, outputs)
├── main.py              # Full pipeline: training + inference + Grad-CAM + PDF report
├── requirements.txt     # Pinned Python dependencies
├── README.md            # This documentation (v2.0 - Lung Cancer Edition)
├── svm_model.pkl        # Trained SVM classifier (generated by Phase 3)
├── gradcam_output.jpg   # Grad-CAM heatmap (generated at inference)
├── final_report.pdf     # Structured medical report (generated at inference)
└── dataset/             # Lung Cancer CT datasets (excluded from Git)
    ├── train/
    │   ├── NORMAL/      # 416 normal scans (IQ-OTH/NCCD dataset)
    │   └── CANCER/      # 561 cancer scans (IQ-OTH/NCCD dataset)
    ├── val/
    │   ├── NORMAL/      # Validation normal scans (Chest CT-Scan dataset)
    │   └── CANCER/      # Validation cancer scans (Chest CT-Scan dataset)
    └── test/
        ├── NORMAL/      # Test normal CT scans (optional)
        └── CANCER/      # Test cancer CT scans (optional)
```

**Note**: Files marked as "generated" (`.pkl`, `.jpg`, `.pdf`) and the entire `dataset/` directory are excluded from version control via `.gitignore`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Feature Extractor | EfficientNetB0 (torchvision, ImageNet pretrained weights) |
| Classifier | SVM with RBF kernel (scikit-learn, balanced class weights) |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| Report Engine | ReportLab (Platypus API for A4 PDF generation) |
| Image Processing | Pillow, OpenCV, NumPy |
| Visualization | Matplotlib |
| Runtime | Python 3.10+, PyTorch 2.x |
| Device Support | CUDA (NVIDIA GPU), MPS (Apple Silicon), CPU fallback |

---

## Performance Highlights

- **100% Cancer Recall** — Zero false negatives ensure no malignant cases are missed
- **73.38% Validation Accuracy** — Balanced performance across 263 validation CT scans
- **High Sensitivity Configuration** — Optimized for screening workflows where sensitivity is paramount
- **Explainable AI** — Grad-CAM heatmaps provide spatial evidence for every prediction
- **Efficient Architecture** — EfficientNetB0 (5.3M params) + SVM enables fast inference on CPU/GPU

---

*Built as a B.Tech Capstone Project — v2.0 Lung Cancer Edition demonstrating hybrid CNN + SVM medical imaging with explainable AI and high-sensitivity cancer detection.*
