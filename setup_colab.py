"""
Colab Setup Script for Selfie-to-Anime UGATIT
Run this in the first cell of your Colab notebook.
"""

import subprocess
import os
import sys


def install_dependencies():
    """Install required Python packages."""
    print("=" * 50)
    print("📦 Installing dependencies...")
    print("=" * 50)
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q',
        'gradio>=4.0.0',
        'opencv-python-headless>=4.8.0',
        'numpy>=1.23.0',
        'Pillow>=9.0.0',
    ])
    print("✅ Dependencies installed!\n")


def setup_kaggle():
    """Set up Kaggle API credentials."""
    print("=" * 50)
    print("🔑 Setting up Kaggle API...")
    print("=" * 50)

    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')

    if os.path.exists(kaggle_json):
        print("✅ Kaggle credentials found!\n")
        return True

    # If running in Colab, try to upload kaggle.json
    try:
        from google.colab import files
        print("📤 Please upload your kaggle.json file:")
        print("   (Get it from kaggle.com → Account → Create New Token)")
        uploaded = files.upload()

        if 'kaggle.json' in uploaded:
            os.makedirs(kaggle_dir, exist_ok=True)
            with open(kaggle_json, 'wb') as f:
                f.write(uploaded['kaggle.json'])
            os.chmod(kaggle_json, 0o600)
            print("✅ Kaggle credentials saved!\n")
            return True
    except ImportError:
        pass

    print("⚠️  No kaggle.json found.")
    print("   Option 1: Upload kaggle.json manually")
    print("   Option 2: Set KAGGLE_USERNAME and KAGGLE_KEY env vars")
    return False


def download_pretrained_model(target_dir='checkpoint'):
    """Download the pretrained UGATIT model from Kaggle."""
    print("=" * 50)
    print("⬇️  Downloading pretrained model (~3GB)...")
    print("   This may take a few minutes...")
    print("=" * 50)

    model_subdir = os.path.join(
        target_dir,
        'UGATIT_light_selfie2anime_lsgan_'
        '4resblock_6dis_1_1_10_10_1000_sn_smoothing'
    )

    # Check if already downloaded
    if os.path.exists(model_subdir) and os.listdir(model_subdir):
        print("✅ Pretrained model already exists!\n")
        return True

    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-q', 'kaggle'
        ])

        # Download the dataset
        subprocess.check_call([
            'kaggle', 'datasets', 'download',
            '-d', 't04glovern/ugatit-selfie2anime-pretrained',
            '-p', '/tmp/ugatit_download',
            '--unzip'
        ])

        # The Kaggle dataset extracts checkpoint files
        # Move them to the expected directory structure
        os.makedirs(target_dir, exist_ok=True)

        # Find and move checkpoint files
        download_dir = '/tmp/ugatit_download'
        if os.path.exists(download_dir):
            # Look for the checkpoint subdirectory
            for root, dirs, files_list in os.walk(download_dir):
                for d in dirs:
                    if 'UGATIT_light' in d:
                        src = os.path.join(root, d)
                        subprocess.check_call(['cp', '-r', src, target_dir])
                        print(f"✅ Model checkpoint copied to {target_dir}/{d}\n")
                        return True

            # If no subdirectory found, check for checkpoint files directly
            ckpt_files = [f for f in os.listdir(download_dir)
                          if 'checkpoint' in f.lower() or '.index' in f
                          or '.data' in f or '.meta' in f]
            if ckpt_files:
                os.makedirs(model_subdir, exist_ok=True)
                for f in os.listdir(download_dir):
                    src = os.path.join(download_dir, f)
                    dst = os.path.join(model_subdir, f)
                    if os.path.isfile(src):
                        subprocess.check_call(['cp', src, dst])
                print(f"✅ Model files copied to {model_subdir}\n")
                return True

        print("⚠️  Could not find checkpoint files in download.")
        print(f"   Contents of {download_dir}:")
        if os.path.exists(download_dir):
            for item in os.listdir(download_dir):
                print(f"   - {item}")
        return False

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/t04glovern/ugatit-selfie2anime-pretrained")
        print("2. Download and extract checkpoint.zip")
        print(f"3. Place files in: {model_subdir}/")
        return False


def verify_setup():
    """Verify that everything is set up correctly."""
    print("=" * 50)
    print("🔍 Verifying setup...")
    print("=" * 50)

    # Check TensorFlow
    import tensorflow as tf
    print(f"   TensorFlow version: {tf.__version__}")

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   GPU available: {gpus[0].name} ✅")
    else:
        print("   GPU: Not available (will use CPU - slower) ⚠️")

    # Check checkpoint
    model_subdir = (
        'checkpoint/UGATIT_light_selfie2anime_lsgan_'
        '4resblock_6dis_1_1_10_10_1000_sn_smoothing'
    )
    if os.path.exists(model_subdir):
        files = os.listdir(model_subdir)
        print(f"   Checkpoint files: {len(files)} files found ✅")
    else:
        print(f"   Checkpoint: NOT FOUND ❌")
        print(f"   Expected at: {model_subdir}/")

    print()


if __name__ == '__main__':
    install_dependencies()
    has_kaggle = setup_kaggle()
    if has_kaggle:
        download_pretrained_model()
    verify_setup()
