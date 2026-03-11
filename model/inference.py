"""
UGATIT Inference Pipeline
Loads pretrained checkpoint and runs selfie→anime translation.
"""

import os
import numpy as np
import cv2

os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

from model.networks import build_test_graph

tf.compat.v1.disable_eager_execution()


class SelfieToAnime:
    """High-level API for selfie→anime translation using UGATIT."""

    def __init__(self, checkpoint_dir='checkpoint', img_size=256):
        self.img_size = img_size
        self.checkpoint_dir = checkpoint_dir
        self.sess = None
        self.test_input = None
        self.test_output = None

        # The checkpoint subdirectory matches UGATIT's model_dir format
        self.model_subdir = (
            'UGATIT_light_selfie2anime_lsgan_'
            '4resblock_6dis_1_1_10_10_1000_sn_smoothing'
        )

    def _find_checkpoint_dir(self):
        """
        Auto-discover the checkpoint directory by searching recursively.
        Handles nested structures from Kaggle extraction
        (e.g., checkpoint/checkpoint/UGATIT_light_...).
        """
        # Try direct path first
        direct = os.path.join(self.checkpoint_dir, self.model_subdir)
        if os.path.exists(direct):
            return direct

        # Search recursively for the model subdirectory
        for root, dirs, files in os.walk(self.checkpoint_dir):
            if self.model_subdir in dirs:
                found = os.path.join(root, self.model_subdir)
                print(f"[*] Found checkpoint at: {found}")
                return found

        # Last resort: look for any directory containing a 'checkpoint' file
        for root, dirs, files in os.walk(self.checkpoint_dir):
            if 'checkpoint' in files:
                print(f"[*] Found checkpoint state in: {root}")
                return root

        return direct  # Fall back to default path for error message

    def load_model(self):
        """Build graph, create session, and restore checkpoint."""
        print("[*] Building computation graph...")
        tf.compat.v1.reset_default_graph()
        self.test_input, self.test_output, _ = build_test_graph(
            img_size=self.img_size)

        self.sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.compat.v1.global_variables_initializer())

        saver = tf.compat.v1.train.Saver()
        ckpt_path = self._find_checkpoint_dir()

        print(f"[*] Looking for checkpoint in: {ckpt_path}")
        ckpt = tf.train.get_checkpoint_state(ckpt_path)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess,
                          os.path.join(ckpt_path, ckpt_name))
            print(f"[✓] Successfully loaded: {ckpt_name}")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {ckpt_path}. "
                "Please download the pretrained model from Kaggle.\n"
                f"Contents of '{self.checkpoint_dir}': "
                f"{os.listdir(self.checkpoint_dir) if os.path.exists(self.checkpoint_dir) else 'DIR NOT FOUND'}"
            )

    def preprocess(self, image):
        """
        Preprocess image for the model.
        Args:
            image: numpy array (H, W, 3) in RGB, uint8
        Returns:
            numpy array (1, 256, 256, 3) normalized to [-1, 1]
        """
        img = cv2.resize(image, (self.img_size, self.img_size))
        img = np.expand_dims(img, axis=0)
        img = img / 127.5 - 1.0
        return img

    def postprocess(self, output):
        """
        Convert model output back to displayable image.
        Args:
            output: numpy array (1, 256, 256, 3) in [-1, 1]
        Returns:
            numpy array (256, 256, 3) in uint8 RGB
        """
        img = ((output[0] + 1.0) / 2.0 * 255.0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def transform(self, image):
        """
        Transform a selfie image to anime style.
        Args:
            image: numpy array (H, W, 3) in RGB, uint8
        Returns:
            numpy array (256, 256, 3) in RGB, uint8
        """
        if self.sess is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        preprocessed = self.preprocess(image)
        result = self.sess.run(
            self.test_output,
            feed_dict={self.test_input: preprocessed})
        return self.postprocess(result)

    def transform_file(self, input_path, output_path=None):
        """
        Transform a selfie image file to anime style.
        Args:
            input_path: path to input image
            output_path: path to save output (optional)
        Returns:
            anime image as numpy array (RGB, uint8)
        """
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image: {input_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anime = self.transform(img)

        if output_path:
            out_bgr = cv2.cvtColor(anime, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, out_bgr)
            print(f"[✓] Saved anime output to: {output_path}")

        return anime

    def close(self):
        """Close the TensorFlow session."""
        if self.sess:
            self.sess.close()
            self.sess = None
