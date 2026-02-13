"""lung_project — Phase 5: Inference + Grad-CAM + PDF report generation."""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np # type: ignore
import torch # type: ignore
from PIL import Image as PILImage # type: ignore
from pytorch_grad_cam import GradCAM # type: ignore
from pytorch_grad_cam.utils.image import show_cam_on_image # pyright: ignore[reportMissingImports]
from reportlab.lib import colors # type: ignore
from reportlab.lib.pagesizes import A4 # type: ignore
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet # type: ignore
from reportlab.lib.units import inch, mm # type: ignore
from reportlab.platypus import ( # type: ignore
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from torchvision import transforms # type: ignore
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0 # type: ignore

PROJECT_DIR = Path(__file__).parent
SVM_MODEL_PATH = PROJECT_DIR / "svm_model.pkl"
GRADCAM_OUTPUT_PATH = PROJECT_DIR / "gradcam_output.jpg"
REPORT_PATH = PROJECT_DIR / "final_report.pdf"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    tensor = transform(image).unsqueeze(0).to(device)
    print(f"[INFO] Image preprocessed: {image_path.name}")
    return tensor


def extract_features(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        features = model(tensor)
    return features.cpu().numpy()


def predict_with_svm(clf, features: np.ndarray) -> tuple[str, float, str]:
    predicted_class = int(clf.predict(features)[0])
    probabilities = clf.predict_proba(features)[0]
    pneumonia_prob = float(probabilities[1])

    class_name = CLASS_NAMES[predicted_class]

    if pneumonia_prob < 0.30:
        risk_level = "LOW"
    elif pneumonia_prob <= 0.70:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    print(f"[INFO] Prediction: {class_name} | Confidence: {pneumonia_prob * 100:.2f}% | Risk: {risk_level}")
    return class_name, pneumonia_prob, risk_level


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
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(30 * mm, 15 * mm, "System Generated Report  |  Confidential")
        canvas.drawRightString(A4[0] - 30 * mm, 15 * mm, f"Page {doc.page}")
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

    elements.append(Paragraph("AI-Based Lung Abnormality Detection Report", title_style))
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
        ["Model Used:", "EfficientNetB0 + SVM"],
        ["Model Version:", "v1.0"],
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
        ["Predicted Class:", predicted_class],
        ["Probability:", f"{probability * 100:.2f} %"],
        ["Risk Level:", risk_level],
    ]
    pred_table = Table(pred_data, colWidths=[130, 320])
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
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Medical Disclaimer", section_style))
    disclaimer_text = (
        "This AI-generated report is intended for educational and research purposes only. "
        "It is NOT a substitute for professional medical diagnosis. "
        "Please consult a certified radiologist or healthcare provider for clinical decisions."
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(elements, onFirstPage=_footer, onLaterPages=_footer)
    print(f"[INFO] PDF report saved: {REPORT_PATH.name}")
    return timestamp


def print_summary(
    predicted_class: str,
    probability: float,
    risk_level: str,
    timestamp: str,
) -> None:
    print(
        f"""
------------------------------------------
PHASE 5 COMPLETED SUCCESSFULLY
------------------------------------------

Report File: {REPORT_PATH.name}
Prediction: {predicted_class}
Probability: {probability * 100:.2f} %
Risk Level: {risk_level}
Grad-CAM Included: Yes
Timestamp: {timestamp}

System Status:
FULL PIPELINE COMPLETE

------------------------------------------"""
    )


def main() -> None:
    raw_path = input("Enter path to chest X-ray image: ").strip()
    image_path = Path(raw_path).expanduser().resolve()
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    device = get_device()
    print(f"[INFO] PyTorch {torch.__version__} | Device: {device}")

    backbone = load_backbone(device)
    clf = load_svm_model(SVM_MODEL_PATH)

    transform = get_transform()
    tensor = preprocess_image(image_path, transform, device)

    features = extract_features(backbone, tensor)
    print(f"[INFO] Feature vector shape: {features.shape}")

    class_name, pneumonia_prob, risk_level = predict_with_svm(clf, features)

    generate_gradcam(image_path)

    timestamp = generate_pdf_report(image_path.name, class_name, pneumonia_prob, risk_level, image_path)

    print_summary(class_name, pneumonia_prob, risk_level, timestamp)


if __name__ == "__main__":
    main()
    sys.exit(0)
