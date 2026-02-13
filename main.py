"""lung_project — Lung Cancer Detection: Training + Inference + Grad-CAM + PDF."""

import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from PIL import Image as PILImage  # type: ignore
from pytorch_grad_cam import GradCAM  # type: ignore
from pytorch_grad_cam.utils.image import show_cam_on_image  # pyright: ignore[reportMissingImports]
from reportlab.lib import colors  # type: ignore
from reportlab.lib.pagesizes import A4  # type: ignore
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
from reportlab.lib.units import inch, mm  # type: ignore
from reportlab.platypus import (  # type: ignore
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore
from sklearn.svm import SVC  # type: ignore
from torchvision import transforms  # type: ignore
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0  # type: ignore
from tqdm import tqdm  # type: ignore

PROJECT_DIR = Path(__file__).parent
SVM_MODEL_PATH = PROJECT_DIR / "svm_model.pkl"
GRADCAM_OUTPUT_PATH = PROJECT_DIR / "gradcam_output.jpg"
REPORT_PATH = PROJECT_DIR / "final_report.pdf"

DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LABEL_MAP = {"NORMAL": 0, "CANCER": 1}
CLASS_NAMES = {0: "NORMAL", 1: "CANCER"}
CLINICAL_LABELS = {
    0: "No Suspicious Lung Malignancy Detected",
    1: "Suspicious Lung Malignancy Detected",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# --- Optimization Configuration ---
USE_CLAHE = True
BEST_THRESHOLD = 0.5
THRESHOLD_PATH = PROJECT_DIR / "best_threshold.pkl"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_clahe(image):
    """Apply CLAHE preprocessing: grayscale → CLAHE → 3-channel."""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_3ch = cv2.merge([enhanced, enhanced, enhanced])
    return PILImage.fromarray(enhanced_3ch)


def load_backbone(device: torch.device) -> torch.nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.to(device)
    model.eval()
    print("[INFO] EfficientNetB0 backbone loaded (classifier → Identity).")
    return model


def load_svm_model(path: Path):
    if not path.exists():
        print(f"[ERROR] SVM model not found: {path}")
        sys.exit(1)
    with open(path, "rb") as f:
        clf = pickle.load(f)
    print(f"[INFO] SVM model loaded from {path.name}")
    return clf


def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(
    image_path: Path, transform: transforms.Compose, device: torch.device
) -> torch.Tensor:
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)
    image = PILImage.open(image_path).convert("RGB")
    if USE_CLAHE:
        image = apply_clahe(image)
    tensor = transform(image).unsqueeze(0).to(device)
    print(f"[INFO] Image preprocessed: {image_path.name} (CLAHE={'ON' if USE_CLAHE else 'OFF'})")
    return tensor


def extract_features(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        features = model(tensor)
    return features.cpu().numpy()


def predict_with_svm(clf, features: np.ndarray, threshold: float | None = None) -> tuple[int, str, str, float, str]:
    probabilities = clf.predict_proba(features)[0]
    cancer_prob = float(probabilities[1])
    t = threshold if threshold is not None else BEST_THRESHOLD
    predicted_class = 1 if cancer_prob >= t else 0

    class_name = CLASS_NAMES[predicted_class]
    clinical_label = CLINICAL_LABELS[predicted_class]

    if cancer_prob < 0.30:
        risk_level = "LOW"
    elif cancer_prob <= 0.70:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    print(f"[INFO] Prediction: {clinical_label}")
    print(f"[INFO] Probability of Cancer: {cancer_prob * 100:.2f}% | Risk Category: {risk_level}")
    return predicted_class, class_name, clinical_label, cancer_prob, risk_level


# ---------------------------------------------------------------------------
# Phase 3 — SVM Training
# ---------------------------------------------------------------------------


def load_dataset_features(
    data_dir: Path,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    skipped = 0
    class_counts: dict[str, int] = {}

    image_paths: list[tuple[Path, int]] = []
    for class_name, label in LABEL_MAP.items():
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            print(f"[WARN] Directory not found: {class_dir}")
            continue
        count = 0
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append((img_path, label))
                count += 1
        class_counts[class_name] = count

    for cls, cnt in class_counts.items():
        print(f"[INFO] {split_name}/{cls}: {cnt} images")

    print(f"[INFO] Extracting features from {split_name} ({len(image_paths)} images)...")

    for img_path, label in tqdm(image_paths, desc=f"  {split_name}", unit="img"):
        try:
            image = PILImage.open(img_path).convert("RGB")
            if USE_CLAHE:
                image = apply_clahe(image)
            tensor = transform(image).unsqueeze(0).to(device)
        except Exception as exc:
            print(f"[WARN] Skipping corrupt image {img_path.name}: {exc}")
            skipped += 1
            continue
        feat = extract_features(model, tensor).flatten()
        features_list.append(feat)
        labels_list.append(label)

    if skipped > 0:
        print(f"[WARN] Skipped {skipped} corrupt images in {split_name}.")

    X = np.array(features_list)
    y = np.array(labels_list)
    print(f"[INFO] {split_name} features shape: {X.shape}")
    return X, y


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    print("[INFO] Training SVM (kernel=rbf, class_weight=balanced)...")
    clf = SVC(kernel="rbf", probability=True, class_weight="balanced")
    clf.fit(X_train, y_train)
    print("[INFO] SVM training complete.")
    return clf


def evaluate_model(
    clf: SVC, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[float, np.ndarray, str]:
    print("[INFO] Evaluating on validation set...")
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=["NORMAL", "CANCER"])
    return acc, cm, report


def save_model(clf: SVC, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[INFO] Model saved to {path.name}")


def tune_threshold(
    clf: SVC, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """Sweep thresholds on validation set and select optimal one."""
    global BEST_THRESHOLD

    probs = clf.predict_proba(X_val)[:, 1]  # P(CANCER)

    # --- Default 0.5 accuracy ---
    y_pred_default = (probs >= 0.5).astype(int)
    default_acc = float(accuracy_score(y_val, y_pred_default))

    thresholds = np.arange(0.30, 0.91, 0.05)

    print("\n" + "=" * 62)
    print("THRESHOLD TUNING \u2014 Validation Set")
    print("=" * 62)
    print(f"{'Threshold':>10} | {'Sensitivity':>12} | {'Specificity':>12} | {'Balanced Acc':>14}")
    print("-" * 62)

    results: list[tuple[float, float, float, float]] = []

    for t in thresholds:
        y_pred_t = (probs >= t).astype(int)
        cm = confusion_matrix(y_val, y_pred_t, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_acc = (sensitivity + specificity) / 2.0
        results.append((float(t), sensitivity, specificity, bal_acc))
        print(f"{t:>10.2f} | {sensitivity:>11.4f} | {specificity:>11.4f} | {bal_acc:>13.4f}")

    # Select: sensitivity >= 0.95, then highest balanced accuracy
    eligible = [(t, se, sp, ba) for t, se, sp, ba in results if se >= 0.95]
    if eligible:
        best = max(eligible, key=lambda x: x[3])
    else:
        # Fallback: pick highest sensitivity, break ties by balanced accuracy
        best = max(results, key=lambda x: (x[1], x[3]))
        print("[WARN] No threshold reached 95% sensitivity — selecting highest sensitivity.")

    best_thresh, best_sens, best_spec, best_bal_acc = best
    BEST_THRESHOLD = best_thresh

    # Save threshold to disk
    with open(THRESHOLD_PATH, "wb") as f:
        pickle.dump(BEST_THRESHOLD, f)
    print(f"[INFO] Threshold saved to {THRESHOLD_PATH.name}")

    # Optimized accuracy
    y_pred_opt = (probs >= BEST_THRESHOLD).astype(int)
    opt_acc = float(accuracy_score(y_val, y_pred_opt))

    print("-" * 62)
    print(f"""
--------------------------------------
THRESHOLD TUNING COMPLETE
--------------------------------------
Selected Threshold: {BEST_THRESHOLD:.2f}
Sensitivity: {best_sens:.4f}
Specificity: {best_spec:.4f}
Balanced Accuracy: {best_bal_acc:.4f}
--------------------------------------
""")

    return BEST_THRESHOLD, best_sens, best_spec, best_bal_acc, default_acc, opt_acc


def print_training_summary(
    n_train: int,
    n_val: int,
    train_time: float,
    accuracy: float,
    cm: np.ndarray,
    report: str,
) -> None:
    print(
        f"""
------------------------------------------
LUNG CANCER SVM TRAINING COMPLETED
------------------------------------------

Dataset: Lung Cancer CT Dataset
Classes:
    NORMAL → 0
    CANCER → 1

Training Samples: {n_train}
Validation Samples: {n_val}
Feature Vector Size: 1280

SVM Configuration:
    Kernel: RBF
    Class Weight: Balanced
    Probability Enabled: Yes

Validation Accuracy: {accuracy * 100:.2f} %
Confusion Matrix:
{cm}

Classification Report:
{report}
Model Saved As:
    {SVM_MODEL_PATH.name}

System Status:
READY FOR PHASE 4 (Lung Cancer Inference + Grad-CAM)

------------------------------------------"""
    )


def run_training() -> None:
    device = get_device()
    print(f"[INFO] PyTorch {torch.__version__} | Device: {device}")
    print(f"[INFO] CLAHE Preprocessing: {'ENABLED' if USE_CLAHE else 'DISABLED'}")

    model = load_backbone(device)
    transform = get_transform()

    X_train, y_train = load_dataset_features(TRAIN_DIR, model, transform, device, "train")
    X_val, y_val = load_dataset_features(VAL_DIR, model, transform, device, "val")

    t0 = time.time()
    clf = train_svm(X_train, y_train)
    train_time = time.time() - t0

    accuracy, cm, report = evaluate_model(clf, X_val, y_val)
    save_model(clf, SVM_MODEL_PATH)

    print_training_summary(len(y_train), len(y_val), train_time, accuracy, cm, report)

    # --- Threshold Tuning ---
    threshold, sensitivity, specificity, bal_acc, default_acc, opt_acc = tune_threshold(
        clf, X_val, y_val
    )

    # --- Final Optimization Summary ---
    print(f"""
------------------------------------------
MODEL OPTIMIZATION COMPLETE
------------------------------------------

Validation Accuracy (default 0.5): {default_acc * 100:.2f} %
Validation Accuracy (optimized threshold): {opt_acc * 100:.2f} %

Cancer Recall (optimized): {sensitivity * 100:.2f} %
Normal Recall (optimized): {specificity * 100:.2f} %

Selected Threshold: {threshold:.2f}
CLAHE Enabled: {USE_CLAHE}

System Status:
OPTIMIZED FOR REVIEW

------------------------------------------""")


def generate_gradcam(image_path: Path) -> str:
    device = torch.device("cpu")

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()

    target_layers = [model.features[-1]]
    layer_name = f"features[-1] ({model.features[-1].__class__.__name__})"

    transform = get_transform()
    image = PILImage.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    original_resized = image.resize((224, 224))
    original_np = np.array(original_resized).astype(np.float32) / 255.0

    visualization = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
    PILImage.fromarray(visualization).save(GRADCAM_OUTPUT_PATH)
    print(f"[INFO] Grad-CAM heatmap saved: {GRADCAM_OUTPUT_PATH.name}")

    return layer_name


def generate_pdf_report(
    image_name: str,
    predicted_class: str,
    probability: float,
    risk_level: str,
    image_path: Path | None = None,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        spaceAfter=4,
        alignment=1,
        textColor=colors.HexColor("#1a1a2e"),
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#16213e"),
        borderPadding=(0, 0, 2, 0),
    )
    body_style = ParagraphStyle(
        "BodyText2",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=4,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#8b0000"),
        borderPadding=6,
        spaceBefore=10,
    )

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(25 * mm, 15 * mm, "AI-Based Lung Cancer Detection System")
        canvas.drawCentredString(A4[0] / 2, 15 * mm, "Confidential \u2013 For Academic Use Only")
        canvas.drawRightString(A4[0] - 25 * mm, 15 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        topMargin=30 * mm,
        bottomMargin=25 * mm,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
    )

    elements: list = []

    elements.append(Paragraph("AI-Based Lung Cancer Detection Report", title_style))
    elements.append(Spacer(1, 2))

    line_table = Table([[""] ], colWidths=[doc.width])
    line_table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1.2, colors.HexColor("#0f3460")),
    ]))
    elements.append(line_table)
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Patient Scan Summary", section_style))
    scan_data = [
        ["Input Image:", image_name],
        ["Date & Time:", timestamp],
        ["Model:", "EfficientNetB0 (Feature Extractor) + SVM (RBF Kernel)"],
        ["Training Dataset:", "Lung Cancer CT Dataset"],
        ["Model Version:", "v2.0 (Lung Cancer Edition)"],
    ]
    scan_table = Table(scan_data, colWidths=[130, 320])
    scan_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(scan_table)

    elements.append(Paragraph("Prediction Results", section_style))

    risk_color = {"LOW": "#228B22", "MODERATE": "#FF8C00", "HIGH": "#B22222"}.get(
        risk_level, "#000000"
    )
    pred_data = [
        ["Prediction Outcome:", predicted_class],
        ["Probability of Malignancy:", f"{probability * 100:.2f} %"],
        ["Risk Level:", risk_level],
    ]
    pred_table = Table(pred_data, colWidths=[160, 290])
    pred_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("TEXTCOLOR", (1, 2), (1, 2), colors.HexColor(risk_color)),
        ("FONTNAME", (1, 2), (1, 2), "Helvetica-Bold"),
    ]))
    elements.append(pred_table)

    if risk_level in ("MODERATE", "HIGH"):
        interp_style = ParagraphStyle(
            "Interpretation", parent=body_style, textColor=colors.HexColor("#8B4513"),
            spaceBefore=6, fontSize=9, leading=12,
        )
        elements.append(Paragraph(
            "If risk level is MODERATE or HIGH, further radiological evaluation is recommended.",
            interp_style,
        ))

    # --- Model Sensitivity Notice ---
    elements.append(Paragraph("Model Sensitivity Notice", section_style))
    sensitivity_text = (
        "This model is configured with high sensitivity to minimize the likelihood "
        "of missing malignant cases. As a result, false positives may occur and "
        "require professional clinical confirmation."
    )
    elements.append(Paragraph(sensitivity_text, body_style))

    elements.append(Paragraph("Original Scan &amp; Grad-CAM Visualization", section_style))

    img_cell_w = 220
    original_cell = Paragraph("<i>[Original image not available]</i>", body_style)
    gradcam_cell = Paragraph("<i>[Grad-CAM image not available]</i>", body_style)

    if image_path and image_path.exists():
        orig = PILImage.open(image_path)
        ow, oh = orig.size
        o_scale = min(img_cell_w / ow, img_cell_w / oh, 1.0)
        original_cell = Image(str(image_path), width=ow * o_scale, height=oh * o_scale)

    if GRADCAM_OUTPUT_PATH.exists():
        gc = PILImage.open(GRADCAM_OUTPUT_PATH)
        gw, gh = gc.size
        g_scale = min(img_cell_w / gw, img_cell_w / gh, 1.0)
        gradcam_cell = Image(str(GRADCAM_OUTPUT_PATH), width=gw * g_scale, height=gh * g_scale)

    label_style = ParagraphStyle("ImgLabel", parent=body_style, alignment=1, spaceBefore=4)
    caption_style = ParagraphStyle(
        "ImgCaption", parent=body_style, alignment=1, fontSize=8, leading=11,
        textColor=colors.HexColor("#444444"),
    )
    img_table = Table(
        [
            [original_cell, gradcam_cell],
            [Paragraph("Original Scan", label_style), Paragraph("Grad-CAM Heatmap", label_style)],
        ],
        colWidths=[img_cell_w + 10, img_cell_w + 10],
    )
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 4))
    elements.append(Paragraph(
        "Grad-CAM heatmap highlighting regions that most influenced the CNN feature extraction stage.",
        caption_style,
    ))
    elements.append(Paragraph(
        "This visualization reflects spatial attention of the CNN backbone and does not represent the SVM decision boundary.",
        caption_style,
    ))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Medical Disclaimer", section_style))
    disclaimer_text = (
        "This AI-generated report is intended strictly for educational and research purposes. "
        "It is NOT a substitute for professional medical diagnosis, treatment, or clinical decision-making. "
        "All predictions must be reviewed and validated by a certified radiologist or healthcare professional."
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(elements, onFirstPage=_footer, onLaterPages=_footer)
    print(f"[INFO] PDF report saved: {REPORT_PATH.name}")
    return timestamp


def print_inference_summary(
    image_path: Path,
    clinical_label: str,
    probability: float,
    risk_level: str,
    device: torch.device,
    timestamp: str,
) -> None:
    print(
        f"""
----------------------------------------------
LUNG CANCER PDF REPORT GENERATED
----------------------------------------------

Report File: {REPORT_PATH.name}
Prediction: {clinical_label}
Probability: {probability * 100:.2f} %
Risk Level: {risk_level}

Report Version: Lung Cancer Edition (v2.0)
Disclaimer Included: Yes
Grad-CAM Embedded: Yes
Timestamp: {timestamp}

System Status:
FULL LUNG CANCER PIPELINE COMPLETE

----------------------------------------------"""
    )


def run_inference() -> None:
    global BEST_THRESHOLD

    raw_path = input("Enter path to CT scan image: ").strip()
    image_path = Path(raw_path).expanduser().resolve()
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    device = get_device()
    print(f"[INFO] PyTorch {torch.__version__} | Device: {device}")

    # Load optimized threshold if available
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH, "rb") as f:
            BEST_THRESHOLD = pickle.load(f)
        print(f"[INFO] Loaded optimized threshold: {BEST_THRESHOLD:.2f}")
    else:
        print(f"[WARN] No optimized threshold found \u2014 using default {BEST_THRESHOLD:.2f}")

    backbone = load_backbone(device)
    clf = load_svm_model(SVM_MODEL_PATH)

    transform = get_transform()
    tensor = preprocess_image(image_path, transform, device)

    features = extract_features(backbone, tensor)
    print(f"[INFO] Feature vector shape: {features.shape}")

    pred_idx, class_name, clinical_label, cancer_prob, risk_level = predict_with_svm(
        clf, features, threshold=BEST_THRESHOLD
    )

    generate_gradcam(image_path)

    timestamp = generate_pdf_report(image_path.name, clinical_label, cancer_prob, risk_level, image_path)

    print_inference_summary(image_path, clinical_label, cancer_prob, risk_level, device, timestamp)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        run_training()
    else:
        run_inference()


if __name__ == "__main__":
    main()
    sys.exit(0)
