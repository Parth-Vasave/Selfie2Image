import json

with open("selfie2anime.ipynb", "r") as f:
    notebook = json.load(f)

# Keep only the first cell (Kaggle creds)
cells = [notebook["cells"][0]]

# Add our clean cells
new_cells_source = [
    "!git clone https://github.com/Parth-Vasave/Selfie2Image\n%cd Selfie2Image",
    "!git pull origin main",
    "!pip install -q gradio opencv-python-headless kaggle tf-keras",
    "!kaggle datasets download -d t04glovern/ugatit-selfie2anime-pretrained -p /tmp/ugatit --unzip\n!mkdir -p checkpoint\n!cp -r /tmp/ugatit/* checkpoint/",
    "import os\nos.environ[\"TF_USE_LEGACY_KERAS\"] = \"1\"\nimport tensorflow as tf\n\nfrom model.inference import SelfieToAnime\n\nmodel = SelfieToAnime(checkpoint_dir='checkpoint')\nmodel.load_model()\nprint(\"✅ Model loaded successfully!\")",
    "!python app.py"
]

for source in new_cells_source:
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" if i < len(source.split('\n')) - 1 else line for i, line in enumerate(source.split('\n'))],
        "execution_count": None,
        "outputs": []
    })

notebook["cells"] = cells

with open("selfie2anime.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook updated.")
