"""
Gradio Web UI for Selfie-to-Anime (UGATIT)
Launch this in Google Colab to get a shareable web interface.
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import gradio as gr
import numpy as np
import cv2
from PIL import Image
from model.inference import SelfieToAnime

# ─────────────────────────────────────────────
# Initialize the model (loaded once)
# ─────────────────────────────────────────────
print("🚀 Loading UGATIT model...")
model = SelfieToAnime(checkpoint_dir='checkpoint')
model.load_model()
print("✅ Model ready!\n")


def selfie_to_anime(input_image):
    """
    Gradio handler: convert selfie to anime.
    Args:
        input_image: PIL Image from Gradio
    Returns:
        PIL Image of anime result
    """
    if input_image is None:
        return None

    # Convert PIL → numpy RGB
    img = np.array(input_image)

    # If RGBA, convert to RGB
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Run inference
    anime = model.transform(img)

    # Convert back to PIL
    return Image.fromarray(anime)


# ─────────────────────────────────────────────
# Build the Gradio Interface
# ─────────────────────────────────────────────

custom_css = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    max-width: 900px !important;
    margin: auto !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 12px 32px !important;
    transition: all 0.3s ease !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
}
footer { display: none !important; }
"""

with gr.Blocks(css=custom_css, title="Selfie2Anime — UGATIT",
               theme=gr.themes.Soft(
                   primary_hue="purple",
                   secondary_hue="violet",
                   neutral_hue="slate",
                   font=gr.themes.GoogleFont("Inter"))) as demo:

    gr.Markdown("""
    # 🎨 Selfie to Anime
    ### Transform your selfie into anime style using U-GAT-IT

    Upload a selfie photo and watch it transform into anime art using the
    UGATIT deep learning model. For best results, use a well-lit, front-facing
    photo with a clear face.
    """)

    with gr.Row(equal_height=True):
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="📸 Upload Your Selfie",
                height=350,
                sources=["upload", "webcam"]
            )
            transform_btn = gr.Button(
                "✨ Transform to Anime",
                variant="primary",
                size="lg"
            )

        with gr.Column():
            output_image = gr.Image(
                type="pil",
                label="🎌 Anime Result",
                height=350
            )

    gr.Markdown("""
    ---
    **How it works:** This app uses [U-GAT-IT](https://github.com/taki0112/UGATIT)
    (Unsupervised Generative Attentional Networks with Adaptive Layer-Instance
    Normalization), a GAN-based model that learns to translate between selfies
    and anime without paired training data.

    **Tips for best results:**
    - Use a front-facing portrait photo
    - Good lighting helps produce better results
    - The model works best with photos where the face is clearly visible
    - Output is 256×256 pixels
    """)

    # Event handlers
    transform_btn.click(
        fn=selfie_to_anime,
        inputs=input_image,
        outputs=output_image
    )
    input_image.change(
        fn=selfie_to_anime,
        inputs=input_image,
        outputs=output_image
    )

# Launch
if __name__ == '__main__':
    demo.launch(share=True, debug=True)
