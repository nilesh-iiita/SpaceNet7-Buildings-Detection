#!/usr/bin/env python
import torch
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
import os

from UNet_monai import Unet

def preprocess_image(image_path):
    """
    Loads and preprocesses a single image file for model prediction.
    """
    image = io.imread(image_path)
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    image = image[:,:,0:3]

    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())

    image = image.transpose((2, 0, 1))
    C, H, W = image.shape
    target_size = 1024

    if H > target_size or W > target_size:
        start_h = (H - target_size) // 2
        start_w = (W - target_size) // 2
        image_processed = image[:, start_h:start_h + target_size, start_w:start_w + target_size]
    else:
        pad_h = target_size - H
        pad_w = target_size - W
        image_processed = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), 'constant')

    original_display_image = image_processed
    image_tensor = torch.tensor(image_processed, dtype=torch.float32)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_normalized = normalize(image_tensor)
    return image_normalized.unsqueeze(0), original_display_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict building masks from an optical image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input optical image (.tif file).")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint (.ckpt file).")
    parser.add_argument("--output_path", type=str, default=None, help="Optional path to save the output plot image and matrix.")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--kernels", default=[[3, 3]] * 5)
    parser.add_argument("--strides", default=[[1, 1]] + [[2, 2]] * 4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weigh_decay", type=float, default=1e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Unet.load_from_checkpoint(args.checkpoint_path, args=args)
    model.to(device)
    model.eval()

    with torch.no_grad():
        image_tensor, original_padded_image = preprocess_image(args.image_path)
        image_tensor = image_tensor.to(device)
        logits = model.model(image_tensor)
        preds_tensor = (torch.sigmoid(logits) > 0.5).int()

    prediction_mask = preds_tensor.squeeze().cpu().numpy()
    original_display_image = original_padded_image.transpose(1, 2, 0)

    print("✅ Prediction complete. Generating plot...")
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(original_display_image)
    ax[0].set_title('Original Optical Image')
    ax[0].axis('off')
    ax[1].imshow(prediction_mask, cmap='gray')
    ax[1].set_title('Predicted Building Mask')
    ax[1].axis('off')

    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save the visual plot
        plt.savefig(args.output_path, bbox_inches='tight', dpi=300)
        print(f"✅ Plot saved to {args.output_path}")

        # --- NEW: Save the prediction matrix as a .npy file ---
        # Create a path for the numpy matrix (e.g., "image.png" -> "image.npy")
        base_path, _ = os.path.splitext(args.output_path)
        matrix_path = base_path + ".npy"

        # Save the prediction_mask array
        np.save(matrix_path, prediction_mask)
        print(f"✅ Prediction matrix saved to {matrix_path}")
        # ----------------------------------------------------

    plt.show()