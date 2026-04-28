"""Enhanced Streamlit demo with educational walkthrough for beginners."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from pneumonia.inference.predictor import Predictor
from pneumonia.model.classifier import load_model
from pneumonia.utils.config import load_config
from pneumonia.utils.logging import setup_logging

setup_logging()

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    .pneumonia-positive {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border-left: 5px solid #f44336;
    }
    .pneumonia-negative {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 5px solid #4caf50;
    }
    .step-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #1976d2;
    }
    .metric-big {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1.05rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ─────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    config = load_config("configs/train_config.yaml")
    model = load_model(config.model, "checkpoints/best_model.pth", device=config.device)
    predictor = Predictor(
        model=model,
        image_size=config.data.image_size,
        device=config.device,
        enable_gradcam=True,
    )
    return predictor, config


predictor, config = load_predictor()


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR — Educational Guide
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📚 How It Works")
    st.caption("A beginner-friendly guide — no Googling needed.")

    st.divider()

    # ── Medical Background ───────────────────────────────────────────
    with st.expander("🏥 What is Pneumonia?", expanded=False):
        st.markdown("""
        **Pneumonia** is an infection that inflames the air sacs (alveoli) 
        in one or both lungs. The air sacs may fill with fluid or pus, 
        causing cough, fever, chills, and difficulty breathing.

        **How doctors detect it:**
        - A **chest X-ray** is the most common diagnostic tool
        - Pneumonia appears as **white/cloudy patches** (called "opacities") 
          in the lung areas
        - Normal lungs appear **dark** on X-rays because air doesn't block X-rays

        **Why automation helps:**
        - Radiologists review hundreds of X-rays daily
        - AI can flag suspicious cases for priority review
        - Especially useful in regions with limited access to specialists
        """)

    with st.expander("📷 What is a Chest X-Ray?", expanded=False):
        st.markdown("""
        A **chest X-ray** (CXR) is an image of your chest created using 
        small amounts of radiation. Think of it like a shadow photograph 
        of your insides.

        **What you see in the image:**
        - **Dark areas** = Air (lungs appear dark when healthy)
        - **White/bright areas** = Dense tissue (bones, heart, fluid)
        - **Gray areas** = Soft tissue (muscles, fat)

        **In pneumonia:**
        - Infected areas fill with fluid → they appear **whiter/hazier**
        - This is called "consolidation" or "infiltrate"
        - Can affect one lung (unilateral) or both (bilateral)
        """)

    st.divider()

    # ── ML Pipeline Steps ────────────────────────────────────────────
    st.subheader("🔬 The ML Pipeline")
    st.caption("What happens when you upload an image:")

    with st.expander("Step 1: Image Preprocessing", expanded=False):
        st.markdown("""
        **What happens:** Your X-ray is prepared for the model.

        - **Resize** to 224×224 pixels (the model's fixed input size)
        - **Convert** to RGB (3 color channels), even though X-rays are grayscale
        - **Normalize** pixel values using ImageNet statistics
          - This means adjusting brightness/contrast to match what the 
            model was originally trained on

        **Why?** Neural networks expect consistent input sizes and 
        value ranges. It's like converting all measurements to the 
        same unit before doing math.
        """)

    with st.expander("Step 2: Feature Extraction", expanded=False):
        st.markdown("""
        **What happens:** The model scans the image for patterns.

        We use **EfficientNet-B0** — a neural network architecture 
        that's really good at finding patterns in images.

        - **Layer 1-2**: Detects simple patterns (edges, lines, textures)
        - **Layer 3-5**: Combines simple patterns into shapes (curves, blobs)
        - **Layer 6-8**: Recognizes complex structures (lung boundaries, 
          rib patterns, fluid patches)

        **Analogy:** It's like reading — first you learn letters, 
        then words, then sentences, then you understand the meaning.

        **Transfer Learning:** The model first learned to recognize 
        objects in everyday photos (ImageNet — cats, dogs, cars). 
        Then we *fine-tuned* it on X-rays. The basic pattern-detection 
        skills transfer surprisingly well!
        """)

    with st.expander("Step 3: Classification", expanded=False):
        st.markdown("""
        **What happens:** The model makes a decision.

        After extracting 1,280 features from the image, a small 
        **classification head** (just 2 layers) produces a single number:

        - **Close to 0** → Likely **Normal**
        - **Close to 1** → Likely **Pneumonia**

        The default threshold is **0.5** — above it means pneumonia.

        **Confidence** tells you how sure the model is:
        - 95%+ → Very confident
        - 70-95% → Fairly confident
        - 50-70% → Uncertain (borderline case)
        """)

    with st.expander("Step 4: Grad-CAM Explanation", expanded=False):
        st.markdown("""
        **What happens:** The model shows *where* it's looking.

        **Grad-CAM** (Gradient-weighted Class Activation Mapping) 
        creates a heatmap showing which image regions influenced 
        the prediction most.

        **How to read the heatmap:**
        - 🔴 **Red/Hot** = High importance (model focused here)
        - 🔵 **Blue/Cool** = Low importance (model ignored this)
        - 🟡 **Yellow** = Moderate importance

        **For a good pneumonia prediction:**
        - Red areas should overlap with the **lung regions**
        - Specifically, with the cloudy/white patches indicating infection

        **For a good normal prediction:**
        - Activation should be more **spread out** or **less intense**
        - No strong focus on any particular region

        **Why this matters:** In healthcare, we can't use "black box" 
        models. Doctors need to verify the model is looking at the 
        right thing, not making decisions based on image artifacts.
        """)

    st.divider()

    # ── Model Stats ──────────────────────────────────────────────────
    st.subheader("📊 Model Report Card")

    st.markdown("""
    | Metric | Score |
    |--------|-------|
    | **Accuracy** | 97.1% |
    | **Recall** | 97.9% |
    | **Precision** | 98.1% |
    | **AUC** | 99.6% |
    """)

    with st.expander("📖 What do these metrics mean?", expanded=False):
        st.markdown("""
        - **Accuracy** (97.1%): Out of all X-rays, 97.1% were classified correctly.

        - **Recall / Sensitivity** (97.9%): Out of all actual pneumonia cases, 
          the model correctly identified 97.9%. *This is the most critical 
          metric — missing pneumonia can be dangerous.*

        - **Precision** (98.1%): Out of all images the model flagged as 
          pneumonia, 98.1% actually had pneumonia. High precision means 
          fewer unnecessary follow-ups.

        - **AUC** (99.6%): Area Under the ROC Curve. Measures how well 
          the model separates normal from pneumonia across all thresholds. 
          1.0 = perfect, 0.5 = random guess. **0.996 is excellent.**

        **The tradeoff:** In medical settings, we prioritize **recall** 
        (catching all pneumonia cases) over precision (some false alarms 
        are acceptable, but missed cases are not).
        """)

    st.divider()
    st.caption("⚠️ Research tool — not for clinical diagnosis.")


# ══════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🫁 Pneumonia Detection from Chest X-Rays</h1>
    <p style="font-size: 1.1rem; color: #666;">
        Upload a chest X-ray → Get instant prediction with visual explanation
    </p>
</div>
""", unsafe_allow_html=True)

# ── Input Section ────────────────────────────────────────────────────
st.divider()

col_input, col_spacer, col_output = st.columns([4, 0.5, 5.5])

with col_input:
    st.subheader("📤 Input")

    input_method = st.radio(
        "Choose input method:",
        ["Upload your own", "Use a sample image"],
        horizontal=True,
    )

    image_path = None

    if input_method == "Upload your own":
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray (JPEG/PNG)",
            type=["jpg", "jpeg", "png"],
            help="Works best with standard PA (posterior-anterior) chest X-rays",
        )
        if uploaded_file:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                image_path = tmp.name

    else:
        test_dir = Path(config.data.root) / "test"
        sample_type = st.radio(
            "Select sample type:",
            ["🟢 Normal", "🔴 Pneumonia"],
            horizontal=True,
        )

        cls = "NORMAL" if "Normal" in sample_type else "PNEUMONIA"
        samples = sorted((test_dir / cls).glob("*"))[:6]

        if samples:
            sample_names = [f.name for f in samples]
            selected = st.selectbox("Pick a sample:", sample_names)
            sample_path = test_dir / cls / selected
            image_path = str(sample_path)

    # Show input image
    if image_path:
        st.image(image_path, caption="Input Chest X-Ray", use_container_width=True)

        st.info("""
        💡 **What to look for in the X-ray:**
        - Dark areas = healthy air-filled lungs
        - White/cloudy patches = possible pneumonia (fluid/infection)
        - The heart appears as a white mass in the center
        """)


# ── Prediction Section ───────────────────────────────────────────────
with col_output:
    if image_path:
        st.subheader("📊 Analysis Results")

        # Pipeline progress
        with st.status("🔍 Running the ML pipeline...", expanded=True) as status:
            st.write("**Step 1:** Preprocessing image (resize, normalize)...")
            gradcam_path = tempfile.mktemp(suffix="_gradcam.png")

            st.write("**Step 2:** Extracting features with EfficientNet-B0...")
            st.write("**Step 3:** Classifying as Normal or Pneumonia...")
            result = predictor.predict(image_path, gradcam_output_path=gradcam_path)

            st.write("**Step 4:** Generating Grad-CAM explanation...")
            status.update(label="✅ Analysis complete!", state="complete")

        # Result card
        label = result["label"]
        confidence = result["confidence"]
        prob = result["probability_pneumonia"]

        if label == "PNEUMONIA":
            st.markdown(f"""
            <div class="result-card pneumonia-positive">
                <div class="metric-big" style="color: #d32f2f;">🔴 PNEUMONIA DETECTED</div>
                <p style="text-align: center; font-size: 1.2rem; margin-top: 0.5rem; color: #b71c1c;">
                    Confidence: <strong>{confidence:.1%}</strong> &nbsp;|&nbsp; 
                    Latency: <strong>{result['latency_ms']:.0f}ms</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card pneumonia-negative">
                <div class="metric-big" style="color: #2e7d32;">🟢 NORMAL</div>
                <p style="text-align: center; font-size: 1.2rem; margin-top: 0.5rem; color: #1b5e20;">
                    Confidence: <strong>{confidence:.1%}</strong> &nbsp;|&nbsp; 
                    Latency: <strong>{result['latency_ms']:.0f}ms</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("#### Probability Breakdown")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("🟢 P(Normal)", f"{(1 - prob):.1%}")
        with prob_col2:
            st.metric("🔴 P(Pneumonia)", f"{prob:.1%}")

        # Visual probability bar
        st.progress(prob, text=f"Pneumonia probability: {prob:.1%}")

        st.caption("""
        **Reading the probability:** The model outputs a number between 0 and 1. 
        Values above 0.5 are classified as Pneumonia. The higher the value, 
        the more confident the model is about its prediction.
        """)

        # Grad-CAM section
        st.markdown("---")
        st.markdown("#### 🔥 Grad-CAM — Where Is the Model Looking?")

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.image(image_path, caption="Original X-Ray", use_container_width=True)
        with gcol2:
            if Path(gradcam_path).exists():
                st.image(gradcam_path, caption="Grad-CAM Heatmap Overlay", use_container_width=True)

        if label == "PNEUMONIA":
            st.success("""
            🔍 **Interpreting the heatmap:** The red/warm regions show where the model 
            detected signs of pneumonia. These should overlap with cloudy/white areas 
            in the original X-ray — those are fluid-filled regions in the lungs.
            """)
        else:
            st.info("""
            🔍 **Interpreting the heatmap:** For normal X-rays, the activation is 
            typically more spread out with less intense hot spots. The model confirms 
            no localized areas of concern (no concentrated fluid or consolidation).
            """)

        # Cleanup temp gradcam file
        Path(gradcam_path).unlink(missing_ok=True)
        if input_method == "Upload your own" and image_path:
            Path(image_path).unlink(missing_ok=True)

    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #999;">
            <h3>👈 Upload an X-ray or select a sample to begin</h3>
            <p>The analysis will appear here with a full explanation.</p>
            <p style="font-size: 0.9rem;">💡 New to this? Open the <strong>📚 How It Works</strong> 
            guide in the sidebar to learn what each step means.</p>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ───────────────────────────────────────────────────────────
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("**Model:** EfficientNet-B0")
with col_f2:
    st.caption("**Dataset:** Kaggle Chest X-Ray (5,856 images)")
with col_f3:
    st.caption("⚠️ Research tool only — not for clinical use")
