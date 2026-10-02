"""Streamlit web application - Handwritten Digit Recognition System (CNN).

Run with:  streamlit run app.py
The app loads the already-trained model (models/digit_cnn_model.keras);
it never retrains.
"""
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config as C
from src.data_loader import load_digits_dataset, make_raw_splits
from src.preprocessing import digit_array_to_pil
from src.utils import load_json

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="🔢", layout="centered")


# ------------------------------------------------------------------ loaders --
@st.cache_resource(show_spinner="Loading trained CNN...")
def get_model():
    from predict import load_trained_model

    return load_trained_model(C.MODEL_PATH)


@st.cache_data
def get_json(path: str):
    p = C.ROOT_DIR / path
    return load_json(p) if p.exists() else None


@st.cache_data
def get_test_samples():
    images, labels = load_digits_dataset()
    s = make_raw_splits(images, labels, C.SEED)
    return s["X_test_raw"], s["y_test"]


def render_preview(preview: np.ndarray) -> Image.Image:
    """8x8 model input -> enlarged black-ink-on-white image (nearest neighbour)."""
    img = Image.fromarray((255 - preview * 255).round().astype(np.uint8), mode="L")
    return img.resize((160, 160), Image.Resampling.NEAREST)


# ------------------------------------------------------------------- header --
st.title("🔢 Handwritten Digit Recognition System")
st.write(
    "This application uses a **Convolutional Neural Network (CNN)** to recognise a single "
    "handwritten digit (**0-9**) in an image and shows how confident the model is."
)

try:
    model = get_model()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

metrics = get_json("results/metrics.json")
model_cfg = get_json("models/model_config.json")
comparison = get_json("results/model_comparison.json")

# ------------------------------------------------------------------ sidebar --
with st.sidebar:
    st.header("Model information")
    st.markdown(
        "- **Model:** CNN (TensorFlow / Keras)\n"
        "- **Classes:** 10 (digits 0-9)\n"
        "- **Input:** 8 x 8 grayscale image\n"
        "- **Training data:** scikit-learn Digits dataset (1,797 images)"
    )
    if metrics:
        st.subheader("Test-set results")
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("F1-score (macro)", f"{metrics['f1_macro']:.3f}")
        st.caption(f"Computed on {metrics['test_set_size']} held-out images.")
    else:
        st.info("Run `python train.py` to generate evaluation metrics.")
    st.divider()
    polarity_label = st.selectbox(
        "Digit colour in your image",
        ["Auto-detect", "Dark digit on light background", "Light digit on dark background"],
        help="Auto-detect looks at the image border. Override it if the preview looks wrong.",
    )
    polarity = {"Auto-detect": "auto", "Dark digit on light background": "dark_on_light",
                "Light digit on dark background": "light_on_dark"}[polarity_label]
    default_thr = (model_cfg or {}).get("low_confidence_threshold", C.LOW_CONFIDENCE_THRESHOLD)
    threshold = st.slider("Low-confidence warning below", 0.50, 0.99, float(default_thr), 0.01)

tab_predict, tab_doc, tab_model, tab_ethics = st.tabs(
    ["Predict digit", "Document mode (all text)", "Model & evaluation", "Responsible use"])

# ------------------------------------------------------------------ predict --
with tab_predict:
    st.subheader("1. Provide a digit image")
    st.write(
        "Upload a **clear photo or scan of ONE digit**: dark pen on plain light paper, "
        "good lighting, digit filling most of the picture, no other writing. "
        "PNG, JPG and JPEG are supported."
    )
    source = st.radio("Image source", ["Upload an image", "Use a dataset sample (demo)"], horizontal=True)

    image = None
    if source == "Upload an image":
        uploaded = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            try:
                image = Image.open(uploaded)
                image.load()
            except Exception:
                st.error("This file could not be read as an image. Please upload a valid PNG or JPG.")
                image = None
    else:
        X_test_raw, y_test = get_test_samples()
        c1, c2 = st.columns(2)
        digit_choice = c1.selectbox("Digit", list(range(10)))
        idxs = np.where(y_test == digit_choice)[0]
        pos = c2.number_input("Sample number", 1, len(idxs), 1)
        image = digit_array_to_pil(X_test_raw[idxs[int(pos) - 1]])
        st.caption("Held-out test image from the dataset, rendered as black ink on white.")

    if image is not None:
        st.subheader("2. Preview")
        st.image(image, caption="Image to classify", width=240)

        st.subheader("3. Prediction")
        if st.button("Predict Digit", type="primary"):
            from predict import predict_image

            try:
                res = predict_image(model, image, polarity=polarity, threshold=threshold)
            except ValueError as err:
                st.error(f"{err}")
            else:
                st.markdown(f"## Predicted Digit: **{res['digit']}**")
                st.markdown(f"### Confidence: {res['confidence'] * 100:.2f}%")
                if res["low_confidence"]:
                    st.warning(
                        f"Low confidence (below {threshold:.0%}). The second most likely digit is "
                        f"{res['second_choice']} ({res['second_confidence']:.1%}). Please verify the "
                        "digit manually or upload a clearer, better-lit image."
                    )
                else:
                    st.success("The model is fairly confident - but it can still be wrong.")

                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.image(render_preview(res["preview"]),
                             caption=f"What the CNN sees (8x8, {res['polarity_used'].replace('_', ' ')})")
                with col_b:
                    st.write("**Probability for each digit**")
                    st.bar_chart(pd.DataFrame({"Probability": res["probabilities"]},
                                              index=[str(i) for i in range(10)]))
                st.caption("The model was trained on 8x8 images, so uploaded photos are cropped, "
                           "centred and shrunk to 8x8 first. Fine details are lost.")
    st.info("**Responsible-use notice:** predictions can be incorrect. A high confidence score is "
            "not a guarantee. When accuracy matters, a human must verify the result.")

# ----------------------------------------------------------------- document --
with tab_doc:
    st.subheader("Read a whole document (printed + handwritten)")
    st.write(
        "Upload a photo, scan or PDF of a document. A **pretrained OCR model (EasyOCR)** finds every "
        "line and converts letters, numbers and punctuation into typed, editable text. "
        "This mode does not use the digit CNN; it is a separate, pretrained model."
    )
    try:
        import easyocr  # noqa: F401
        ocr_ready = True
    except ImportError:
        ocr_ready = False

    if not ocr_ready:
        st.warning("Document mode is not installed in this deployment. It needs the extra packages in "
                   "`requirements-ocr.txt` (EasyOCR + PyTorch, about 2-3 GB). Run the app locally with "
                   "`pip install -r requirements-ocr.txt` to use it. The digit recognizer works normally.")
    else:
        from src.document_ocr import load_pages, pages_to_text, read_page

        MAX_PAGES = 10
        doc_file = st.file_uploader("Upload a document page (PNG, JPG or PDF)",
                                    type=["png", "jpg", "jpeg", "pdf"], key="doc_upload")
        if doc_file is not None:
            try:
                page_images = load_pages(doc_file.getvalue(), doc_file.name)[:MAX_PAGES]
            except Exception:
                st.error("This file could not be opened. Please upload a valid PNG, JPG or PDF.")
                page_images = []
            for i, img in enumerate(page_images[:3], 1):
                st.image(img, caption=f"Page {i}", width=320)
            if len(page_images) > 3:
                st.caption(f"{len(page_images)} pages loaded (first 3 previewed).")
            if page_images and st.button("Read document", type="primary"):
                with st.spinner("Reading... the first run downloads the OCR model and can take a few minutes."):
                    st.session_state["doc_pages"] = [read_page(p) for p in page_images]
                    st.session_state["doc_name"] = doc_file.name

        pages = st.session_state.get("doc_pages")
        if pages:
            lines = [l for pg in pages for l in pg]
            if not lines:
                st.error("No text was found. Try a sharper, better-lit image.")
            else:
                text = pages_to_text(pages)
                avg = sum(l["confidence"] for l in lines) / len(lines)
                c1, c2 = st.columns(2)
                c1.metric("Lines read", len(lines))
                c2.metric("Average confidence", f"{avg:.0%}")
                st.text_area("Recognised text (you can correct it here)", text, height=320)
                st.download_button("Download as .txt", text, file_name="recognised_text.txt")
                low = [l for l in lines if l["low_confidence"]]
                if low:
                    st.warning(f"{len(low)} line(s) have low confidence. Please check them against the original:")
                    for l in low:
                        st.write(f"- `{l['text']}` ({l['confidence']:.0%})")
        st.info("**Responsible use:** OCR makes mistakes, especially with messy handwriting, "
                "similar characters (O/0, l/1, S/5) and numbers. Always verify important values. "
                "Documents are processed in memory and are not stored by this app.")

# -------------------------------------------------------------------- model --
with tab_model:
    st.subheader("Configuration")
    if model_cfg:
        hp = model_cfg["hyperparameters"]
        st.table(pd.DataFrame({
            "Parameter": ["Selected model", "Conv filters", "Kernel", "Dense units", "Dropout", "Optimizer",
                          "Learning rate", "Loss", "Batch size", "Max epochs", "Augmentation", "Seed"],
            "Value": [model_cfg["model_name"], f"{hp['conv1_filters']} -> {hp['conv2_filters']}",
                      f"{hp['kernel_size']}x{hp['kernel_size']}", hp["dense_units"], hp["dropout"], "Adam",
                      hp["learning_rate"], "sparse categorical cross-entropy", hp["batch_size"],
                      hp["epochs"], "random +-1 px shifts" if hp["augment"] else "none", model_cfg["seed"]],
        }).astype(str))
    else:
        st.info("models/model_config.json not found - run `python train.py`.")

    st.subheader("Evaluation (held-out test set)")
    if metrics:
        st.table(pd.DataFrame({
            "Metric": ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"],
            "Value": [f"{metrics['accuracy']:.4f}", f"{metrics['precision_macro']:.4f}",
                      f"{metrics['recall_macro']:.4f}", f"{metrics['f1_macro']:.4f}"]}))
    if comparison:
        st.subheader("Baseline vs adjusted model")
        rows = []
        for k, label in (("accuracy", "Accuracy"), ("precision_macro", "Precision"),
                         ("recall_macro", "Recall"), ("f1_macro", "F1")):
            rows.append({"Metric": label,
                         "Baseline (test)": f"{comparison['baseline']['test'][k]:.4f}",
                         "Adjusted (test)": f"{comparison['adjusted']['test'][k]:.4f}"})
        st.table(pd.DataFrame(rows))
        st.caption(f"Selected on validation data: **{comparison['selected_model']}**.")
    for fname, cap in (("results/confusion_matrix.png", "Confusion matrix"),
                       ("results/training_accuracy.png", "Training vs validation accuracy"),
                       ("results/training_loss.png", "Training vs validation loss")):
        if (C.ROOT_DIR / fname).exists():
            st.image(str(C.ROOT_DIR / fname), caption=cap)

# ------------------------------------------------------------------- ethics --
with tab_ethics:
    st.subheader("Limitations and risks")
    st.markdown(
        "- The model learned from only 1,797 small 8x8 images, written by a limited group of people.\n"
        "- Handwriting styles, pens, paper, tilt, blur, shadows or poor contrast that differ from the "
        "training data can cause wrong predictions.\n"
        "- Uploaded photos must be shrunk to 8x8; small mistakes in cropping or inversion change the result.\n"
        "- The confidence score is **not** a guarantee: a model can be confidently wrong."
    )
    st.subheader("How the risks are reduced")
    st.markdown(
        "- Clear instructions for taking a good image.\n"
        "- A confidence threshold: low-confidence results show a warning and the second-best guess.\n"
        "- A preview of the exact 8x8 image the model sees, so users can spot preprocessing problems.\n"
        "- Human verification: this tool **supports** people; it must not make final decisions "
        "(e.g. grading, payments, records) on its own.\n"
        "- Images are processed in memory only; this app does not store uploads."
    )
