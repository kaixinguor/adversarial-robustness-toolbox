import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from collections import defaultdict
from typing import Dict, List, Tuple
from art.tools.coco_categories80 import COCO_INSTANCE_CATEGORY_NAMES as COCO80_NAMES

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
def coco90_to_80():
    """
    将COCO官方90类ID（含跳号）映射到连续80类索引（0-based）。
    输入: COCO原始类别ID（1-90）
    输出: 对应的80类索引（0-79），若ID无效则返回-1。
    
    注：COCO官方跳过部分ID（如12,26,29等），这些ID会返回-1。
    """

    # 映射到常见的80类索引（0-based）
    coco90_to_80_map = {
         1:  0,  2:  1,  3:  2,  4:  3,  5:  4,  6:  5,  7:  6,  8:  7,  9:  8, 10:  9,
        11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
        22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29,
        35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
        46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
        56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59,
        67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69,
        80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
    }

    # 内部函数
    def mapper(category_id):
        return coco90_to_80_map.get(category_id, -1)  # 返回 -1 表示该类不在 80 类中

    return mapper

def coco_label_mapping(label):
    """
    调用coco90_to_80的接口
    inputs:
    category_id: int 原始的类别标签
    outputs:
    label_map: int 对应的80类的标签，如果没有返回-1
    """

    map_fn = coco90_to_80()
    return map_fn(label)
def get_original_annotations(
    image_path: str,
    file_to_annotations: Dict[str, List[Dict]]
) -> List[Tuple]:
    """
    Get original annotations for an image without any scaling or preprocessing.
    
    Args:
        image_path: Path to the image file
        file_to_annotations: Mapping from filename to annotations
        
    Returns:
        List of (bbox, label) tuples in original image coordinates
    """
    filename = os.path.basename(image_path)
    annotations = file_to_annotations.get(filename, [])
    
    # Convert annotations to the format expected by visualization
    processed_annotations = []
    for annotation in annotations:
        bbox = annotation['bbox']  # [x, y, w, h] format
        label = annotation['category_id']
        
        # Convert [x, y, w, h] to [x_min, y_min, x_max, y_max]
        x_min, y_min, w, h = bbox
        x_max = x_min + w
        y_max = y_min + h
        
        # Map COCO label to vehicle label if needed
        mapped_label = coco_label_mapping(label)
        
        processed_annotations.append(([x_min, y_min, x_max, y_max], mapped_label))
    
    return processed_annotations

def visualize_original_annotations(
    image_path: str, 
    file_to_annotations: Dict[str, List[Dict]],
    class_names: List[str] = None, 
    save_path: str = None
) -> None:
    """
    Visualize original image annotations with bounding boxes and labels.
    This function works with original image coordinates, not scaled ones.
    
    Args:
        image_path: Path to the image file
        file_to_annotations: Mapping from filename to annotations
        class_names: List of class names
        save_path: Save path, if None then display the image
    """
    # Get original annotations
    annotations = get_original_annotations(image_path, file_to_annotations)
    
    # Read original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    print(f"Original image size: {img_width} x {img_height}")
    
    # Create matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Define color list
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    print(f"\n=== Original Annotation Info: {os.path.basename(image_path)} ===")
    print(f"Number of detected objects: {len(annotations)}")
    
    for i, (bbox, label) in enumerate(annotations):
        x_min, y_min, x_max, y_max = bbox
        
        # Check if bbox coordinates are within image bounds
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))
        
        # Get class name
        if class_names and 0 <= label < len(class_names):
            class_name = class_names[label]
        else:
            class_name = f"class_{label}"
        
        # Select color
        color = colors[label % len(colors)]
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label text
        ax.text(x_min, y_min - 5, class_name, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10, color='white', weight='bold')
        
        # Print annotation info
        print(f"Object {i+1}: {class_name} (ID: {label}) - BBox: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")
        print(f"  BBox size: {x_max - x_min:.1f} x {y_max - y_min:.1f}")
    
    ax.set_title(f'Original Image Annotations: {os.path.basename(image_path)}', fontsize=14, weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Original annotation visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_training_images(
    images_directory: str,
    file_to_annotations: Dict[str, List[Dict]],
    save_directory: str,
    max_images: int = 5
) -> None:
    """
    Visualize training images with their annotations.
    
    Args:
        images_directory: Directory containing image files
        file_to_annotations: Mapping from filename to annotations
        max_images: Maximum number of images to visualize
    """
    print(f"\nVisualizing training images (up to {max_images})...")
    
    # Create training images directory
    os.makedirs(save_directory, exist_ok=True)
    
    image_count = 0
    for filename in sorted(os.listdir(images_directory)):
        if image_count >= max_images:
            break
            
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            image_path = os.path.join(images_directory, filename)
            
            # Get annotations for this image
            annotations = get_original_annotations(image_path, file_to_annotations)
            
            if annotations:  # Only visualize images with annotations
                print(f"Visualizing training image {image_count+1}: {filename}")
                save_path = os.path.join(save_directory, f'training_image_{image_count+1}.png')
                visualize_original_annotations(
                    image_path=image_path,
                    file_to_annotations=file_to_annotations,
                    class_names=COCO80_NAMES,
                    save_path=save_path
                )
                image_count += 1


def load_annotation_data(annotation_path: str) -> Tuple[Dict[int, str], Dict[str, List[Dict]]]:
    """
    Load COCO annotation data and organize it by filename.
    
    Args:
        annotation_path: Path to the annotation JSON file
        
    Returns:
        Tuple of (image_id_to_filename mapping, file_to_annotations mapping)
    """
    with open(annotation_path, 'r') as file:
        annotation_data = json.load(file)
    
    # Create mapping from image_id to filename (e.g., 70000 -> 000000070000.jpg)
    image_id_to_filename = {
        img['id']: img['file_name'] 
        for img in annotation_data['images']
    }
    
    # Organize bbox and category_id by filename
    file_to_annotations = defaultdict(list)
    for annotation in annotation_data['annotations']:
        filename = image_id_to_filename[annotation['image_id']]
        file_to_annotations[filename].append({
            'bbox': annotation['bbox'],  # Format: [x, y, w, h]
            'category_id': annotation['category_id']  # Categories: 2,3,4,6,8
        })
    
    return image_id_to_filename, file_to_annotations