# AI-Based Lung Abnormality Detection using EfficientNetB0 + SVM

A hybrid deep learning and classical machine learning pipeline for automated pneumonia detection from chest X-ray images, with Grad-CAM explainability and structured PDF medical reporting.

> **B.Tech Capstone Project** — Medical AI Pipeline
> Stack: PyTorch · scikit-learn · torchvision · pytorch-grad-cam · ReportLab

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Phase-by-Phase Implementation](#3-phase-by-phase-implementation)
4. [Dataset Details](#4-dataset-details)
5. [Installation Guide](#5-installation-guide)
6. [How to Run](#6-how-to-run)
7. [Output Explanation](#7-output-explanation)
8. [Explainability Justification](#8-explainability-justification)
9. [Limitations](#9-limitations)
10. [Future Improvements](#10-future-improvements)
11. [Medical Disclaimer](#11-medical-disclaimer)

---

## 1. Problem Statement

Pneumonia is a leading cause of mortality worldwide, particularly in children under five and immunocompromised adults. Early and accurate detection from chest X-rays is critical for timely intervention, yet radiological diagnosis depends heavily on the availability of trained specialists — a resource that remains scarce in many healthcare systems.

**Challenges with manual diagnosis:**

- High inter-observer variability among radiologists
- Time-intensive visual inspection under high patient volumes
- Subtle radiographic patterns that can be missed under fatigue or workload pressure

**Why explainable AI matters:**

Black-box deep learning models are insufficient for clinical adoption. Physicians require interpretable outputs that highlight *where* and *why* a model reaches its conclusion. This project addresses that need by combining a high-capacity CNN feature extractor with a classical SVM classifier and Grad-CAM visual explanations, producing a transparent and auditable diagnostic pipeline.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                          │
│                                                                 │
│   Input X-ray Image (JPG/PNG)                                   │
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
Image → Resize + Normalize → EfficientNetB0 → [1, 1280] → SVM → Class + Prob → Risk → PDF
                                    └──→ Grad-CAM → Heatmap ──────────────────────→ PDF
```

---

## 3. Phase-by-Phase Implementation

### Phase 1 — Environment Setup

| | |
|---|---|
| **Objective** | Establish a reproducible Python environment and verify PyTorch installation. |
| **What** | Created project scaffold (`main.py`, `requirements.txt`, `README.md`), pinned all 9 dependencies, detected compute device. |
| **Why** | A reproducible environment is the foundation of any production ML system. Device detection ensures GPU acceleration is used when available. |
| **Key decisions** | Version-range pinning (not exact pins) for flexibility across machines. Support for CUDA, MPS (Apple Silicon), and CPU fallback. |
| **Output** | Console confirmation of PyTorch version and selected device. |
| **→ Next** | Device and dependency infrastructure feeds directly into Phase 2 model loading. |

### Phase 2 — EfficientNetB0 Feature Extraction

| | |
|---|---|
| **Objective** | Load a pretrained CNN backbone and produce a fixed-length feature vector from any input image. |
| **What** | Loaded EfficientNetB0 with ImageNet weights, replaced the classification head with `nn.Identity()`, ran a forward pass to obtain a `[1, 1280]` feature vector. |
| **Why** | Transfer learning allows leveraging ImageNet-learned visual representations without training from scratch. The Identity replacement converts the classifier into a pure feature extractor. |
| **Key decisions** | EfficientNetB0 chosen for its optimal accuracy-to-parameter-count ratio (~4M params). The 1280-d pooled feature is compact yet highly discriminative. |
| **Output** | `torch.Size([1, 1280])` feature vector printed to console. |
| **→ Next** | This extractor becomes the front-end for Phase 3 dataset-wide feature extraction. |

### Phase 3 — SVM Training on Extracted Features

| | |
|---|---|
| **Objective** | Train a classical SVM classifier on CNN-extracted features from the Chest X-Ray Pneumonia dataset. |
| **What** | Iterated over all training images, extracted 1280-d features per image, built `(N, 1280)` feature matrix, trained `SVC(kernel='rbf', probability=True, class_weight='balanced')`, evaluated on validation split. |
| **Why** | SVMs excel on moderate-dimensional, linearly-separable-ish feature spaces. The RBF kernel captures non-linear decision boundaries. Balanced class weights compensate for dataset imbalance. |
| **Key decisions** | `probability=True` enables Platt scaling for calibrated probabilities (required for risk scoring). Features are extracted once and held in memory — no redundant forward passes. Corrupt images are skipped gracefully. |
| **Output** | `svm_model.pkl` (persisted via pickle), validation accuracy, confusion matrix, classification report. |
| **→ Next** | The saved SVM model is loaded at inference time in Phase 4. |

### Phase 4 — Inference + Grad-CAM Explainability

| | |
|---|---|
| **Objective** | Run single-image prediction with probability-based risk scoring and generate a Grad-CAM heatmap. |
| **What** | Loaded backbone + SVM, preprocessed user-specified image, extracted features, ran SVM inference, mapped probability to LOW/MODERATE/HIGH risk, generated Grad-CAM from the last convolutional layer, saved overlay heatmap. |
| **Why** | End-to-end inference is the operational mode of the system. Risk scoring translates raw probabilities into clinically meaningful categories. Grad-CAM provides visual evidence of model attention. |
| **Key decisions** | Risk thresholds: <0.30 LOW, 0.30–0.70 MODERATE, >0.70 HIGH. Grad-CAM targets `features[-1]` (the final convolutional block) for maximum spatial resolution before global pooling. Grad-CAM runs on CPU to avoid MPS hook compatibility issues. |
| **Output** | `gradcam_output.jpg`, console prediction summary with class, probability, and risk level. |
| **→ Next** | All prediction artifacts are passed to Phase 5 for report generation. |

### Phase 5 — Structured Medical PDF Report

| | |
|---|---|
| **Objective** | Generate a professional, structured PDF report containing all prediction results and visual evidence. |
| **What** | Built an A4 PDF using ReportLab with: title, patient scan summary, prediction results (color-coded risk), side-by-side original scan + Grad-CAM visualization, medical disclaimer, and page footer. |
| **Why** | A printable, shareable report is essential for clinical communication. Side-by-side image comparison allows physicians to correlate AI attention with anatomical regions. |
| **Key decisions** | ReportLab Platypus for flowable document layout. Color-coded risk levels (green/orange/red). Auto-generated timestamp. Professional footer with confidentiality notice. |
| **Output** | `final_report.pdf` |

---

## 4. Dataset Details

| | |
|---|---|
| **Dataset** | [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Source** | Guangzhou Women and Children's Medical Center |
| **Classes** | `NORMAL` (0), `PNEUMONIA` (1) |
| **Image type** | Anterior-posterior chest X-rays (JPEG) |

**Expected directory structure:**

```
dataset/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
```

**Why EfficientNetB0?**

EfficientNetB0 uses compound scaling (depth × width × resolution) to achieve ImageNet top-1 accuracy of 77.1% with only 5.3M parameters — roughly 7× smaller than ResNet-152 at comparable accuracy. For transfer learning, this compactness yields fast feature extraction without sacrificing representation quality.

**Why SVM on extracted features?**

- The 1280-d feature space is well-structured after ImageNet pretraining, making it amenable to kernel-based separation.
- SVMs generalize well on small-to-moderate datasets (~5K samples) where deep fine-tuning risks overfitting.
- `predict_proba` (via Platt scaling) provides calibrated confidence scores for downstream risk assessment.

---

## 5. Installation Guide

### Prerequisites

- Python 3.10+
- pip
- Git (for cloning)
- Kaggle account (for dataset download)

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

### Step 4 — Download the Dataset

#### 📥 Dataset Information

- **Dataset Name**: Chest X-Ray Images (Pneumonia)
- **Kaggle Link**: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size**: ~1.15 GB (5,863 images)
- **Classes**: NORMAL and PNEUMONIA
- **Format**: JPEG images

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

#### 📦 Download and Extract Dataset

```bash
# Ensure you're in the project root directory
cd /path/to/lung_cancer_detection

# Download the dataset (this may take a few minutes)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract the dataset
unzip chest-xray-pneumonia.zip -d dataset/

# Remove the zip file (optional)
rm chest-xray-pneumonia.zip
```

The extraction will create the following structure:

```
dataset/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

#### 🗂️ Reorganize Dataset Structure

The downloaded dataset has an extra `chest_xray` folder. Move the contents up one level:

```bash
mv dataset/chest_xray/* dataset/
rmdir dataset/chest_xray
```

**Final expected structure:**

```
lung_cancer_detection/
├── main.py
├── requirements.txt
├── README.md
└── dataset/
    ├── train/
    │   ├── NORMAL/         # 1,341 images
    │   └── PNEUMONIA/      # 3,875 images
    ├── val/
    │   ├── NORMAL/         # 8 images
    │   └── PNEUMONIA/      # 8 images
    └── test/
        ├── NORMAL/         # 234 images
        └── PNEUMONIA/      # 390 images
```

#### ✅ Verify Dataset Installation

```bash
# Check folder structure
ls -R dataset/

# Count images in training set
find dataset/train -type f | wc -l
# Expected output: 5216

# Verify NORMAL and PNEUMONIA folders exist
ls dataset/train/
# Expected output: NORMAL  PNEUMONIA
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
   - `PNEUMONIA` → class 1

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
TRAIN_DATA_PATH = '../chest_xray_data/train'
VAL_DATA_PATH = '../chest_xray_data/val'
TEST_DATA_PATH = '../chest_xray_data/test'
```

#### 🚨 Important Notes

- **Dataset must preserve the folder structure**: Each split (train/val/test) must have `NORMAL/` and `PNEUMONIA/` subdirectories
- **Image format**: The model expects JPEG or PNG images
- **Class balance**: The training set has significant class imbalance (~3:1 pneumonia:normal ratio), which is handled by `class_weight='balanced'` in the SVM
- **.gitignore**: The entire `dataset/` folder is excluded from version control to prevent committing large image files

### Step 6 — Run Training (Optional - Phase 3)

If you want to retrain the SVM model:

```bash
python main.py
# The script will automatically:
# 1. Extract features from all training images
# 2. Train the SVM classifier
# 3. Save svm_model.pkl
```

> **Note**: Training may take 10-30 minutes depending on your CPU/GPU.

### Step 7 — Run Inference + Report Generation

```bash
python main.py
```

When prompted, provide the path to a chest X-ray image:

```
Enter path to chest X-ray image: /path/to/xray.jpg
```

The script will:
1. Load the EfficientNetB0 backbone and SVM model
2. Extract 1280-d features from the input image
3. Predict class and probability
4. Assign risk level (LOW / MODERATE / HIGH)
5. Generate Grad-CAM heatmap → `gradcam_output.jpg`
6. Generate PDF report → `final_report.pdf`

### Step 8 — Verify Outputs

```bash
ls -lh svm_model.pkl gradcam_output.jpg final_report.pdf
```

---

## 6. How to Run

### Prerequisites

Ensure you have completed all installation steps (Section 5), including:
- ✅ Python environment setup
- ✅ Dependencies installed
- ✅ Dataset downloaded and placed in `dataset/` folder
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
Enter path to chest X-ray image: /path/to/xray.jpg
```

**Example usage:**

```
Enter path to chest X-ray image: dataset/test/PNEUMONIA/person1_virus_6.jpeg
```

The pipeline will then:

1. Load the EfficientNetB0 backbone and SVM model
2. Preprocess and extract 1280-d features from the input image
3. Run SVM prediction and calculate probability
4. Assign risk level based on confidence (LOW / MODERATE / HIGH)
5. Generate Grad-CAM heatmap → `gradcam_output.jpg`
6. Generate structured medical PDF report → `final_report.pdf`

### Training from Scratch (Optional - Phase 3)

If you want to retrain the SVM model with your own dataset:

```bash
python main.py
```

The script will:
- Iterate through all images in `dataset/train/`
- Extract EfficientNetB0 features for each image
- Train an SVM with RBF kernel
- Evaluate on validation set (`dataset/val/`)
- Save the trained model as `svm_model.pkl`

> **Note**: Training typically takes 10-30 minutes depending on your hardware.

---

## 7. Output Explanation

### `svm_model.pkl`

Serialized scikit-learn `SVC` model trained on 1280-d EfficientNetB0 features. Contains the learned support vectors, kernel coefficients, and Platt scaling parameters for probability estimation.

### `gradcam_output.jpg`

Gradient-weighted Class Activation Map overlaid on the resized (224×224) input image. Hot regions (red/yellow) indicate areas the CNN attended to most strongly when producing the feature vector. This provides spatial evidence for the prediction.

### `final_report.pdf`

Structured A4 medical report containing:

- Patient scan metadata and timestamp
- Prediction class, probability, and color-coded risk level
- Side-by-side comparison: original scan and Grad-CAM heatmap
- Medical disclaimer
- Confidentiality footer with page numbering

### Console summary

Each execution prints a structured summary block confirming all pipeline stages completed successfully, including prediction details, file outputs, and system readiness status.

---

## 8. Explainability Justification

### Why Grad-CAM?

Gradient-weighted Class Activation Mapping (Grad-CAM) computes the gradient of a target class score with respect to the feature maps of a convolutional layer. These gradients are globally average-pooled to produce importance weights, which are then used to create a weighted combination of forward activation maps. The result is a coarse localization heatmap highlighting image regions most relevant to the prediction.

### Why explainability matters in medical AI

- **Regulatory compliance**: Medical AI systems increasingly require interpretability for FDA/CE clearance pathways.
- **Clinical trust**: Physicians are unlikely to adopt opaque models. Visual evidence of model reasoning enables informed clinical judgment.
- **Error detection**: Heatmaps can reveal when a model attends to irrelevant artifacts (e.g., chest tube markers, text annotations) rather than pathological features, flagging potential failure modes.

### Important distinction

Grad-CAM visualizes **CNN attention** — specifically, which spatial regions of the input contributed most to the EfficientNetB0 feature representation. It does **not** visualize the SVM decision boundary. The SVM operates on the 1280-d pooled vector (spatial information is already collapsed). The Grad-CAM heatmap explains *what the CNN saw*, which indirectly informs *what the SVM used* to classify.

---

## 9. Limitations

- **2D radiographs only** — The pipeline processes standard anterior-posterior chest X-rays. It does not support CT, MRI, or 3D volumetric data.
- **Binary classification** — Only distinguishes NORMAL vs. PNEUMONIA. Does not differentiate bacterial vs. viral pneumonia or detect other pulmonary conditions.
- **Not clinically validated** — This system has not undergone prospective clinical trials or regulatory review. It must not be used for clinical decision-making.
- **Dataset bias** — The Kaggle Chest X-Ray dataset is sourced from a single institution (Guangzhou Women and Children's Medical Center), which limits generalizability across populations, imaging equipment, and acquisition protocols.
- **No DICOM support** — Accepts only standard image formats (JPEG, PNG). DICOM metadata (patient demographics, acquisition parameters) is not parsed.
- **Probability calibration** — Platt scaling on SVM provides approximate probabilities. For clinical-grade risk scoring, isotonic regression or temperature scaling should be evaluated.

---

## 10. Future Improvements

- **Fine-tune EfficientNetB0** on the chest X-ray domain to learn pathology-specific features beyond ImageNet representations.
- **DICOM ingestion** — Parse DICOM headers for patient metadata, acquisition parameters, and automatic report population.
- **Multi-class extension** — Distinguish bacterial vs. viral pneumonia, detect additional conditions (tuberculosis, cardiomegaly, pleural effusion).
- **FastAPI deployment** — Expose the pipeline as a REST API with image upload, JSON prediction response, and PDF download endpoint.
- **Segmentation module** — Integrate a U-Net lung segmentation preprocessor to mask non-lung regions before feature extraction.
- **Calibrated probability scaling** — Replace Platt scaling with isotonic regression or Venn prediction for tighter confidence intervals.
- **Batch inference** — Support directory-level or PACS-integrated batch processing for radiology workflow integration.
- **Model versioning** — Track backbone and SVM versions with MLflow or DVC for reproducible experiment management.

---

## 11. Medical Disclaimer

> **This AI-generated system is intended for educational and research purposes only.**
>
> It is **NOT** a substitute for professional medical diagnosis. The predictions, risk scores, and visual explanations produced by this pipeline have not been validated in a clinical setting and must not be used to make or influence clinical decisions.
>
> **Always consult a certified radiologist or healthcare provider for diagnosis and treatment.**

---

## Project Structure

```
lung_cancer_detection/
├── .gitignore           # Git ignore rules (excludes dataset, models, outputs)
├── main.py              # Full pipeline: inference + Grad-CAM + PDF report
├── requirements.txt     # Pinned Python dependencies
├── README.md            # This documentation
├── svm_model.pkl        # Trained SVM classifier (generated by Phase 3)
├── gradcam_output.jpg   # Grad-CAM heatmap (generated at inference)
├── final_report.pdf     # Structured medical report (generated at inference)
└── dataset/             # Chest X-Ray Pneumonia dataset (excluded from Git)
    ├── train/
    │   ├── NORMAL/      # 1,341 normal chest X-rays
    │   └── PNEUMONIA/   # 3,875 pneumonia chest X-rays
    ├── val/
    │   ├── NORMAL/      # 8 validation normal images
    │   └── PNEUMONIA/   # 8 validation pneumonia images
    └── test/
        ├── NORMAL/      # 234 test normal images
        └── PNEUMONIA/   # 390 test pneumonia images
```

**Note**: Files marked as "generated" and the entire `dataset/` directory are excluded from version control via `.gitignore`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Feature Extractor | EfficientNetB0 (torchvision, ImageNet weights) |
| Classifier | SVM with RBF kernel (scikit-learn) |
| Explainability | Grad-CAM (pytorch-grad-cam) |
| Report Engine | ReportLab (Platypus) |
| Image Processing | Pillow, OpenCV, NumPy |
| Visualization | Matplotlib |
| Runtime | Python 3.10+, PyTorch 2.x |

---

*Built as a B.Tech Capstone Project — demonstrating hybrid CNN + SVM medical imaging with explainable AI.*
