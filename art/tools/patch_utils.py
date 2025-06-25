import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_trained_patch(patch: np.ndarray, save_dir: str, save_name: str) -> str:
    """
    Save trained adversarial patch to file.
    
    Args:
        patch: Trained adversarial patch as numpy array
        save_dir: Directory to save the patch
        save_name: Name for the patch file
        
    Returns:
        Path to the saved patch file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as .npy file for later loading
    patch_npy_path = os.path.join(save_dir, f'{save_name}.npy')
    np.save(patch_npy_path, patch)
    
    # Save as .png file for visualization
    patch_png_path = os.path.join(save_dir, f'{save_name}.png')
    patch_png = patch.astype(np.uint8)
    # patch_png = np.transpose(patch_png, (1, 2, 0))  # [C,H,W] -> [H,W,C]
    patch_png = patch_png[..., ::-1]  # BGR to RGB
    patch_png = Image.fromarray(patch_png)
    patch_png.save(patch_png_path)
    
    print(f"Trained patch saved to: {patch_npy_path}")
    print(f"Patch visualization saved to: {patch_png_path}")
    
    return patch_npy_path

def visualize_patch_only(adversarial_patch: np.ndarray, save_path: str = None):
    if adversarial_patch.shape[0] == 3 and adversarial_patch.shape[-1] != 3:
        patch_rgb = np.transpose(adversarial_patch, (1, 2, 0))
    else:
        patch_rgb = adversarial_patch
    patch_rgb = np.clip(patch_rgb, 0, 255).astype(np.uint8)
    plt.figure(figsize=(6, 6))
    plt.imshow(patch_rgb)
    plt.title('Generated Adversarial Patch', fontsize=14, weight='bold')
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()