import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def save_patch(patch, save_dir='dpatch_results/patch', save_name='patch', channels_first=True):
    # inputs:
    # patch: np.ndarray 补丁 BGR [C,H,W] 或 [H,W,C]
    # save_dir: str 保存路径
    # save_name: str 保存文件名
    # channel_first: bool 输入patch是否为channel_first格式 [C,H,W]，True为channel_first，False为channel_last [H,W,C]
    os.makedirs(save_dir, exist_ok=True)
    
    # 根据channel_first参数处理输入格式
    if channels_first:
        # 输入为 [C,H,W] 格式
        patch_npy = np.transpose(patch, (1, 2, 0))  # [H,W,C]
        patch_npy = patch_npy[..., ::-1]  # 转回RGB,必须要在[H,W,C]格式下
    else:
        # 输入为 [H,W,C] 格式
        patch_npy = patch[..., ::-1]  # 转回RGB
    
    # 保存npy
    save_name_npy = save_name + '.npy'
    save_path_npy = os.path.join(save_dir, save_name_npy)
    np.save(save_path_npy, patch_npy)
    
    # 保存png
    save_name_png = save_name + '.png'
    save_path_png = os.path.join(save_dir, save_name_png)
    patch_png = patch.astype(np.uint8)
    
    if channels_first:
        # 输入为 [C,H,W] 格式，需要转换为 [H,W,C]
        patch_png = np.transpose(patch_png, (1, 2, 0))
    
    patch_png = patch_png[..., ::-1]  # 转回RGB,必须要在[H,W,C]格式下
    patch_png = Image.fromarray(patch_png)
    patch_png.save(save_path_png)

    return save_path_npy

def load_trained_patch(patch_path: str, channels_first=True) -> np.ndarray:
    """
    Load trained adversarial patch from file.
    
    Args:
        patch_path: Path to the saved patch file (.npy)
        
    Returns:
        Loaded adversarial patch as numpy array
    """
    if not os.path.exists(patch_path):
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    
    patch = np.load(patch_path)
    
    # Ensure patch is in the correct format [C,H,W] BGR
    if len(patch.shape) == 3:
        if patch.shape[2] == 3:  # [H,W,C] format
            patch = np.transpose(patch, (2, 0, 1))  # [H,W,C] -> [C,H,W]
            # Note: We assume the saved patch is already in BGR format
            # If it's in RGB format, uncomment the next line:
            # patch = patch[..., ::-1]  # RGB to BGR
    
    print(f"Loaded trained patch from: {patch_path}")
    print(f"Patch shape: {patch.shape}")
    
    return patch

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