# 🎨 Selfie to Anime — UGATIT

Transform your selfie photos into anime-style art using the **U-GAT-IT** (Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization) deep learning model.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Quick Start (Google Colab)

### Prerequisites
- A Google account (for Colab)
- Kaggle API token (`kaggle.json`) — [Get it here](https://www.kaggle.com/settings) → "Create New Token"

### Step-by-Step Instructions

Open a new [Google Colab notebook](https://colab.research.google.com/) with **GPU runtime**:
- Go to `Runtime → Change runtime type → T4 GPU`

---

#### Cell 1: Clone the Repository

```python
# Clone the project
!git clone https://github.com/YOUR_USERNAME/AAI_project.git
%cd AAI_project
```

> **Note:** If not using GitHub, upload the project files manually via the Colab file browser.

---

#### Cell 2: Setup & Download Model

```python
# Install dependencies and download pretrained model
!pip install -q gradio opencv-python-headless kaggle tf-keras

# Upload your kaggle.json when prompted
from google.colab import files
import os

# Upload Kaggle credentials
print("📤 Upload your kaggle.json file:")
uploaded = files.upload()

os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download pretrained model (~3GB, takes a few minutes)
!kaggle datasets download -d t04glovern/ugatit-selfie2anime-pretrained -p /tmp/ugatit --unzip
!mkdir -p checkpoint
!cp -r /tmp/ugatit/* checkpoint/
```

---

#### Cell 3: Verify Setup

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# Check checkpoint exists
ckpt_dir = 'checkpoint/UGATIT_light_selfie2anime_lsgan_4resblock_6dis_1_1_10_10_1000_sn_smoothing'
if os.path.exists(ckpt_dir):
    print(f"✅ Checkpoint found: {len(os.listdir(ckpt_dir))} files")
else:
    print("❌ Checkpoint not found!")
    # List what we have
    if os.path.exists('checkpoint'):
        for item in os.listdir('checkpoint'):
            print(f"  - {item}")
```

---

#### Cell 4: Test Inference (Optional)

```python
from model.inference import SelfieToAnime
import cv2
import matplotlib.pyplot as plt
from google.colab import files

# Load model
model = SelfieToAnime(checkpoint_dir='checkpoint')
model.load_model()

# Upload a test selfie
print("📤 Upload a selfie image:")
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Transform
anime = model.transform_file(filename, 'anime_output.png')

# Display
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
original = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
axes[0].imshow(original)
axes[0].set_title('Original Selfie')
axes[0].axis('off')
axes[1].imshow(anime)
axes[1].set_title('Anime Version')
axes[1].axis('off')
plt.tight_layout()
plt.show()
```

---

#### Cell 5: Launch Web UI 🎨

```python
# Launch the Gradio web interface with a shareable link
!python app.py
```

This will output a **public URL** you can share with anyone!

---

## 📁 Project Structure

```
AAI_project/
├── app.py                  # Gradio web UI
├── setup_colab.py          # Automated Colab setup script
├── model/
│   ├── __init__.py
│   ├── ops.py              # Neural network ops (ported to TF2 compat)
│   ├── networks.py         # UGATIT generator architecture
│   └── inference.py        # Model loading & inference pipeline
├── checkpoint/             # Pretrained model (downloaded from Kaggle)
├── requirements.txt
└── README.md
```

## 🧠 How It Works

**U-GAT-IT** is a GAN-based model for unpaired image-to-image translation. Key innovations:

1. **Attention Module** — Focuses on the most discriminative regions between domains
2. **AdaLIN (Adaptive Layer-Instance Normalization)** — Dynamically adjusts normalization to control how much shape and texture change
3. **Light Mode** — Uses global average pooling in the MLP for reduced memory usage

The pretrained model was trained on the selfie2anime dataset for 100 epochs.

## 📚 Credits

- **Paper:** [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830) (ICLR 2020)
- **Original Code:** [taki0112/UGATIT](https://github.com/taki0112/UGATIT)
- **Pretrained Model:** [Kaggle — t04glovern/ugatit-selfie2anime-pretrained](https://www.kaggle.com/datasets/t04glovern/ugatit-selfie2anime-pretrained)
- **Dataset:** Selfie2Anime from [UCF Selfie Dataset](https://www.crcv.ucf.edu/data/Selfie/) + [Danbooru2018](https://www.gwern.net/Danbooru2018)

## 📄 License

MIT License — see the original [UGATIT repo](https://github.com/taki0112/UGATIT) for model license details.
