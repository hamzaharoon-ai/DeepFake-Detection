# app.py
import os
import io
import json
import math
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

import base64
from io import BytesIO

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    #  page_title="DeepFake Detection — Images",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Helper utilities
# -------------------------
@st.cache_resource
def load_detection_model(path="deepfake_detection_model.h5"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    model = load_model(path, compile=False)
    return model

def infer_model_input_shape(model):
    inp_shape = model.input_shape
    if len(inp_shape) == 4:
        _, h, w, c = inp_shape
        if h is None or w is None or c is None:
            raise ValueError(f"Model has unknown input shape: {inp_shape}")
        return int(h), int(w), int(c)
    elif len(inp_shape) == 2:
        total = inp_shape[1]
        if total is None:
            raise ValueError(f"Model has unknown flattened input size: {inp_shape}")
        s = int(math.isqrt(total))
        if s * s == total:
            return s, s, 1
        if total % 3 == 0:
            s3 = int(math.isqrt(total // 3))
            if s3 * s3 * 3 == total:
                return s3, s3, 3
        raise ValueError(f"Cannot infer H,W,C from flattened size {total}. Please check your model input shape: {inp_shape}")
    else:
        raise ValueError(f"Unsupported model input shape: {inp_shape}")

def preprocess_image_adaptive(image_bgr, target_h, target_w, target_c):
    if target_c == 3:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    elif target_c == 1:
        img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb = img_gray
    else:
        raise ValueError(f"Unsupported target channels: {target_c}")

    resized = cv2.resize(img_rgb, (target_w, target_h))
    if target_c == 3:
        arr = img_to_array(resized).astype("float32") / 255.0
    else:
        arr = np.expand_dims(resized, axis=-1).astype("float32") / 255.0

    return np.expand_dims(arr, axis=0)

def predict_with_model_adaptive(model, processed):
    preds = model.predict(processed)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    preds = np.array(preds)

    if preds.ndim == 2 and preds.shape[1] >= 2:
        probs = preds[0]
        label_idx = int(np.argmax(probs))
        return probs, label_idx

    if preds.ndim == 2 and preds.shape[1] == 1:
        p = float(preds[0][0])
        probs = np.array([1 - p, p])
        label_idx = int(np.argmax(probs))
        return probs, label_idx

    flat = preds.ravel()
    if flat.sum() <= 0:
        flat = np.abs(flat)
    probs = flat / flat.sum()
    label_idx = int(np.argmax(probs))
    return probs, label_idx

def label_from_index(idx):
    return "Fake" if idx == 0 else "Real"

def pretty_probs(probs):
    if len(probs) >= 2:
        return {"Fake": float(probs[0]), "Real": float(probs[1])}
    else:
        return {f"Class_{i}": float(p) for i, p in enumerate(probs)}

def compute_gradcam(model, processed_image, last_conv_layer_name=None):
    try:
        preds = model(processed_image)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        pred_index = int(tf.argmax(preds[0]))
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if "conv" in layer.name.lower():
                    last_conv = layer
                    break
            else:
                raise ValueError("No conv layer found for Grad-CAM.")
        else:
            last_conv = model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(processed_image)
            loss = predictions[:, pred_index]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0].numpy()
        pooled = pooled_grads.numpy()
        heatmap = np.zeros(shape=conv_outputs.shape[:2], dtype=np.float32)
        for i in range(pooled.shape[-1]):
            heatmap += pooled[i] * conv_outputs[:, :, i]
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() == 0:
            return np.zeros_like(heatmap)
        heatmap /= heatmap.max()
        return heatmap
    except Exception as e:
        raise RuntimeError(f"Grad-CAM failed: {e}")

def overlay_heatmap(img_rgb, heatmap, alpha=0.5):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_colored, alpha, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# -------------------------
# Sidebar controls (keeps main area clean)
# -------------------------
with st.sidebar:
    st.title("DeepFake Detection — Controls")
    model_path = st.text_input("Model path (.h5)", value="deepfake_detection_model.h5")
    st.markdown("---")
    st.header("Cover image")
    st.caption("Place coverpage.png in app folder or upload one here")
    uploaded_cover = st.file_uploader("Upload cover (optional)", type=["png", "jpg", "jpeg"])
    st.markdown("---")
    threshold = st.slider("Threshold for 'Real' (0..1)", 0.0, 1.0, 0.5, 0.01)
    show_probs = st.checkbox("Show probability bars", value=True)
    enable_gradcam = st.checkbox("Enable Grad-CAM (experimental)", value=False)
    st.markdown("---")
    st.write("Preview settings")
    preview_width = st.slider("Preview width (px)", 200, 1200, 600, step=50)
    st.caption("Lower values reduce scroll and keep images nicely within layout.")

# -------------------------
# Load model
# -------------------------
model = None
model_load_exception = None
try:
    model = load_detection_model(model_path)
    try:
        target_h, target_w, target_c = infer_model_input_shape(model)
    except Exception as e:
        target_h, target_w, target_c = 96, 96, 3
        model_load_exception = f"Inferred input shape failed: {e}. Defaulting to 96x96x3 for preprocessing."
except Exception as e:
    model_load_exception = e

# -------------------------
# Header / Cover area
# -------------------------
st.title("🛡️ DeepFake Detection — Images")
st.write("Upload an image to detect whether it's Real or Fake. The app adapts preprocessing to your model where possible.")

def show_centered_cover():
    try:
        def show_image(img):
            st.markdown(
                f"""
                <div style='display:flex;justify-content:center;'>
                    <img src='data:image/png;base64,{image_to_base64(img)}'
                         style='width:95%;max-width:450px;height:auto;border-radius:10px;object-fit:contain;'/>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if uploaded_cover is not None:
            img = Image.open(uploaded_cover).convert("RGB")
            show_image(img)
            return

        if os.path.exists("cover.png"):
            img = Image.open("cover.png").convert("RGB")
            show_image(img)
            return

        st.info("No cover image found. Place `cover.png` in the app folder or upload one from the sidebar.")
    except Exception as e:
        st.error(f"Cover image failed to load: {e}")


# small helper to embed PIL image as base64 (keeps layout control)
def image_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    import base64
    return base64.b64encode(buffered.getvalue()).decode()

show_centered_cover()
st.markdown("---")

# -------------------------
# Main content organized in tabs to reduce scrolling
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Model Info", "Advanced", "About"])

# -------------------------
# Prediction tab
# -------------------------
with tab1:
    st.header("Prediction")
    st.write("Upload a single image below. Results and options are shown next to the preview to minimize scrolling.")

    # layout: left = image preview, right = results & controls
    left_col, right_col = st.columns([1, 1])

    with left_col:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="pred_uploader")
        st.caption("Supported: jpg, jpeg, png. Use moderate resolution images for faster inference.")

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image_bgr is None:
                st.error("Uploaded file could not be decoded as an image. Try another file.")
            else:
                img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                # safely display preview sized by preview_width but constrained to container
                st.image(img_rgb, caption="Uploaded image (preview)", width=preview_width)

    with right_col:
        # present model status and action buttons in right column
        st.subheader("Model & Run")
        if uploaded_file is None:
            st.info("Upload an image to run detection. Model loaded: " + ("✅" if model is not None else "❌"))
            if model_load_exception:
                st.warning(f"Model note: {model_load_exception}")
        else:
            if model is None:
                st.error(f"Model not loaded: {model_load_exception}")
            else:
                try:
                    # infer again if needed
                    if 'target_h' not in locals() or 'target_w' not in locals() or 'target_c' not in locals():
                        target_h, target_w, target_c = infer_model_input_shape(model)
                except Exception:
                    target_h, target_w, target_c = 96, 96, 3

                try:
                    processed = preprocess_image_adaptive(image_bgr, target_h, target_w, target_c)
                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")
                    st.stop()

                need_flatten = (len(model.input_shape) == 2)
                if need_flatten:
                    processed_for_predict = processed.reshape((processed.shape[0], -1)).astype("float32")
                else:
                    processed_for_predict = processed.astype("float32")

                try:
                    probs, idx = predict_with_model_adaptive(model, processed_for_predict)
                    probs_dict = pretty_probs(probs)
                except ValueError as e:
                    st.error(f"Model prediction error: {e}")
                    st.info("Attempting fallback: reshaping input to (1,H,W,C) and retrying...")
                    try:
                        fallback = preprocess_image_adaptive(image_bgr, target_h, target_w, target_c)
                        probs, idx = predict_with_model_adaptive(model, fallback)
                        probs_dict = pretty_probs(probs)
                        st.success("Fallback prediction succeeded.")
                    except Exception as e2:
                        st.error(f"Fallback also failed: {e2}")
                        st.stop()
                except Exception as e:
                    st.error(f"Unexpected prediction error: {e}")
                    st.stop()

                # Final label using threshold on Real prob (keeps original behavior)
                real_prob = probs_dict.get("Real", 0.0)
                final_label = "Real" if real_prob >= threshold else "Fake"
                model_label = label_from_index(idx)

                # display results neatly
                color = "#00B37E" if final_label == "Real" else "#FF4B4B"
                st.markdown(f"<h2 style='text-align:center;color:{color};'>The image is <strong>{final_label}</strong></h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center;color:gray;'>Model raw label (argmax): <strong>{model_label}</strong></p>", unsafe_allow_html=True)

                if show_probs:
                    st.subheader("Probabilities")
                    if "Fake" in probs_dict and "Real" in probs_dict:
                        fake_p = probs_dict["Fake"]
                        real_p = probs_dict["Real"]
                        # Show progressbar for Real (keeps concise)
                        st.progress(int(real_p * 100))
                        st.write(f"Real: {real_p:.2%} — Fake: {fake_p:.2%}")
                    else:
                        for k, v in probs_dict.items():
                            st.write(f"{k}: {v:.2%}")

                # Optional Grad-CAM shown under probabilities
                if enable_gradcam:
                    st.subheader("Grad-CAM (experimental)")
                    try:
                        if need_flatten:
                            conv_input = preprocess_image_adaptive(image_bgr, target_h, target_w, target_c)
                        else:
                            conv_input = processed_for_predict
                        heatmap = compute_gradcam(model, conv_input)
                        h0, w0 = img_rgb.shape[:2]
                        heatmap_resized = cv2.resize(heatmap, (w0, h0))
                        overlay = overlay_heatmap(img_rgb, heatmap_resized, alpha=0.45)
                        # display overlay sized same as preview
                        st.image(overlay, caption="Grad-CAM overlay", width=preview_width)
                    except Exception as e:
                        st.warning(f"Grad-CAM error: {e}")

                # Save JSON result and download button
                result = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "filename": uploaded_file.name,
                    "model_label": model_label,
                    "final_label_thresholded": final_label,
                    "probs": probs_dict,
                }
                json_bytes = json.dumps(result, indent=2).encode("utf-8")
                st.download_button("Download prediction JSON", data=json_bytes, file_name=f"prediction_{uploaded_file.name}.json", mime="application/json")

# -------------------------
# Model Info tab
# -------------------------
with tab2:
    st.header("Model information & expected input")
    if model is None:
        st.error(f"Model not loaded: {model_load_exception}")
    else:
        st.success("Model loaded.")
        st.write("Model summary (top layers):")
        buf = []
        model.summary(print_fn=lambda s: buf.append(s))
        # show only first ~400 lines to avoid long scroll
        st.text("\n".join(buf[:400]))
        st.markdown("---")
        try:
            h, w, c = infer_model_input_shape(model)
            st.markdown(f"**Expected input:** {h} x {w} x {c}")
        except Exception as e:
            st.warning(f"Could not infer input shape: {e}")
            st.markdown("Default preprocessing uses 96×96×3 unless you change the model.")

        # Show training figures if present
        if os.path.exists("Figure_2.png"):
            st.image("Figure_2.png", caption="Training accuracy", width='stretch')
        if os.path.exists("Figure_1.png"):
            st.image("Figure_1.png", caption="Training loss", width='stretch')

# -------------------------
# Advanced tab
# -------------------------
with tab3:
    st.header("Advanced tools")
    st.write("Utilities for debugging & batch inference.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Check model dummy inference"):
            if model is None:
                st.error("Model not loaded.")
            else:
                try:
                    try:
                        h, w, c = infer_model_input_shape(model)
                        dummy = np.zeros((1, h, w, c), dtype=np.float32)
                    except Exception:
                        dummy = np.zeros((1, 96, 96, 3), dtype=np.float32)
                    out = model.predict(dummy)
                    st.write("Dummy input shape:", dummy.shape)
                    st.write("Model output shape:", np.shape(out))
                except Exception as e:
                    st.error(f"Dummy inference failed: {e}")
    with c2:
        if st.button("Show last model layers"):
            if model is None:
                st.error("Model not loaded.")
            else:
                for i, layer in enumerate(model.layers[-12:][::-1]):
                    st.write(f"{i+1}. {layer.name} — {layer.__class__.__name__}")

    st.markdown("Batch predictions: upload multiple images (not implemented fully).")

# -------------------------
# About tab
# -------------------------
with tab4:
    st.header("About this app")
    st.markdown("""
    **DeepFake Detection — Image Classifier**

    - Loads a .h5 Keras model and detects whether an uploaded image is Real or Fake.
    - This app automatically adapts preprocessing to the model's input shape (best-effort).
    - Grad-CAM is available as an experimental explainability tool (works for conv-based models).
    """)
    st.caption("Built with Streamlit — put `deepfake_detection_model.h5` and `coverpage.png` in the same folder as this file.")

st.markdown("---")
st.caption("Hints: keep your model file and figures in the same folder as app.py. Use moderate resolution images to speed up inference.")
