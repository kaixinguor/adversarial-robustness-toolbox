import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List
import os
import cv2
from typing import Dict, List
import time

def plot_image_with_boxes(img, boxes, pred_cls, title, scores=None):
    """
    Plot image with bounding boxes and labels using improved styling.
    
    Args:
        img: Input image as numpy array
        boxes: List of bounding boxes in format [[(x1, y1), (x2, y2)], ...]
        pred_cls: List of predicted class names
        title: Title for the plot
        scores: Optional list of confidence scores
    """
    # Define colors for different classes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 
              'cyan', 'magenta', 'yellow', 'lime', 'navy', 'teal', 'maroon', 'olive']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img.astype(np.uint8))
    ax.set_title(title, fontsize=16, weight='bold', pad=20)
    ax.axis('off')
    
    for i in range(len(boxes)):
        # Extract box coordinates
        x1, y1 = boxes[i][0]
        x2, y2 = boxes[i][1]
        
        # Get class name and score
        class_name = pred_cls[i]
        score_text = ""
        if scores is not None and i < len(scores):
            score_text = f" ({scores[i]:.2f})"
        
        # Choose color based on class (cycle through colors)
        color = colors[i % len(colors)]
        
        # Draw bounding box using matplotlib patches
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label with improved styling
        label_text = f"{class_name}{score_text}"
        ax.text(x1, y1 - 10, label_text, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8),
                fontsize=12, color='white', weight='bold')
    
    plt.tight_layout()
    plt.show()

def draw_detection_box(axe, img, title, detections, class_names, colors, is_gt=False):

    axe.imshow(img)
    axe.set_title(title, fontsize=14, weight='bold')
    axe.axis('off')

    # Draw detection results on patched image
    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]
    for i, (bbox, label, score) in enumerate(zip(boxes, labels, scores)):
        
        x_min, y_min, x_max, y_max = bbox
        
        # Get class name
        if class_names and 0 <= label < len(class_names):
            class_name = class_names[label]
        else:
            class_name = f"class_{label}"
        
        color = colors[label % len(colors)]
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        axe.add_patch(rect)
        
        # Add label with score
        if is_gt:
            axe.text(x_min, y_min - 5, f"{class_name} (GT)", 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10, color='white', weight='bold')
        else:
            axe.text(x_min, y_min - 5, f"{class_name} ({score:.2f})", 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                            fontsize=10, color='white', weight='bold')
        
# def visualize_attack_comparison(
#     processed_image: np.ndarray,
#     processed_annotations: np.ndarray,
#     patched_image: np.ndarray,
#     original_detections: np.ndarray,
#     patched_detections: np.ndarray,
#     class_names: List[str] = None,
#     save_dir: str = None,
#     score_thresh: float=0.3
# ) -> None:
#     """
#     Visualize attack comparison: original vs patched image with annotations and detection results.
    
#     Args:
#         processed_image: Processed image array in BGR format [H,W,C]
#         processed_annotations: Ground truth annotations
#         patched_image: Patched image array in BGR format [H,W,C]
#         original_detections: Detection results on original image
#         patched_detections: Detection results on patched image
#         class_names: List of class names
#         save_dir: Directory to save the comparison image
#     """
#     # Convert BGR to RGB for visualization
#     processed_image_rgb = processed_image[..., ::-1].copy()  # BGR to RGB
#     patched_image_rgb = patched_image[..., ::-1].copy()  # BGR to RGB
    
#     # Ensure image values are in valid range for matplotlib
#     processed_image_rgb = np.clip(processed_image_rgb, 0, 255).astype(np.uint8)
#     patched_image_rgb = np.clip(patched_image_rgb, 0, 255).astype(np.uint8)

#     img_height, img_width = processed_image_rgb.shape[:2]
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
#     # Define colors
#     colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
#     # Plot 1: Original image with ground truth annotations
#     axes[0, 0].imshow(processed_image_rgb)
#     axes[0, 0].set_title('Original Image (Ground Truth)', fontsize=14, weight='bold')
#     axes[0, 0].axis('off')
    
#     # Draw ground truth annotations
#     boxes = processed_annotations["boxes"]
#     labels = processed_annotations["labels"]
#     scores = processed_annotations["scores"]
#     for i, (bbox, label, score) in enumerate(zip(boxes, labels, scores)):
#         x_min, y_min, x_max, y_max = bbox
        
#         # Check bounds
#         x_min = max(0, min(x_min, img_width))
#         y_min = max(0, min(y_min, img_height))
#         x_max = max(0, min(x_max, img_width))
#         y_max = max(0, min(y_max, img_height))
        
#         # Get class name
#         if class_names and 0 <= label < len(class_names):
#             class_name = class_names[label]
#         else:
#             class_name = f"class_{label}"
        
#         color = colors[label % len(colors)]
        
#         # Draw bounding box
#         rect = patches.Rectangle(
#             (x_min, y_min), x_max - x_min, y_max - y_min,
#             linewidth=2, edgecolor=color, facecolor='none'
#         )
#         axes[0, 0].add_patch(rect)
        
#         # Add label
#         axes[0, 0].text(x_min, y_min - 5, f"{class_name} (GT)", 
#                         bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
#                         fontsize=10, color='white', weight='bold')
    
#     # Plot 2: Original image with detection results
#     axes[0, 1].imshow(processed_image_rgb)
#     axes[0, 1].set_title('Original Image (Detection Results)', fontsize=14, weight='bold')
#     axes[0, 1].axis('off')
    
#     # Draw detection results on original image
#     boxes = original_detections["boxes"]
#     labels = original_detections["labels"]
#     scores = original_detections["scores"]
#     for i, (bbox, label, score) in enumerate(zip(boxes, labels, scores)):
#         if score < score_thresh:
#             continue
#         x_min, y_min, x_max, y_max = bbox
        
#         # Get class name
#         if class_names and 0 <= label < len(class_names):
#             class_name = class_names[label]
#         else:
#             class_name = f"class_{label}"
        
#         color = colors[label % len(colors)]
        
#         # Draw bounding box
#         rect = patches.Rectangle(
#             (x_min, y_min), x_max - x_min, y_max - y_min,
#             linewidth=2, edgecolor=color, facecolor='none'
#         )
#         axes[0, 1].add_patch(rect)
        
#         # Add label with score
#         axes[0, 1].text(x_min, y_min - 5, f"{class_name} ({score:.2f})", 
#                         bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
#                         fontsize=10, color='white', weight='bold')
    
#     # Plot 3: Patched image with ground truth annotations
#     axes[1, 0].imshow(patched_image_rgb)
#     axes[1, 0].set_title('Patched Image (Ground Truth)', fontsize=14, weight='bold')
#     axes[1, 0].axis('off')
    
#     # Draw ground truth annotations on patched image
#     boxes = processed_annotations["boxes"]
#     labels = processed_annotations["labels"]
#     scores = processed_annotations["scores"]
#     for i, (bbox, label, score) in enumerate(zip(boxes, labels, scores)):
#         x_min, y_min, x_max, y_max = bbox
        
#         # Check bounds
#         x_min = max(0, min(x_min, img_width))
#         y_min = max(0, min(y_min, img_height))
#         x_max = max(0, min(x_max, img_width))
#         y_max = max(0, min(y_max, img_height))
        
#         # Get class name
#         if class_names and 0 <= label < len(class_names):
#             class_name = class_names[label]
#         else:
#             class_name = f"class_{label}"
        
#         color = colors[label % len(colors)]
        
#         # Draw bounding box
#         rect = patches.Rectangle(
#             (x_min, y_min), x_max - x_min, y_max - y_min,
#             linewidth=2, edgecolor=color, facecolor='none'
#         )
#         axes[1, 0].add_patch(rect)
        
#         # Add label
#         axes[1, 0].text(x_min, y_min - 5, f"{class_name} (GT)", 
#                         bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
#                         fontsize=10, color='white', weight='bold')
    
#     # Plot 4: Patched image with detection results
#     axes[1, 1].imshow(patched_image_rgb)
#     axes[1, 1].set_title('Patched Image (Detection Results)', fontsize=14, weight='bold')
#     axes[1, 1].axis('off')
    
#     # Draw detection results on patched image
#     boxes = patched_detections["boxes"]
#     labels = patched_detections["labels"]
#     scores = patched_detections["scores"]
#     for i, (bbox, label, score) in enumerate(zip(boxes, labels, scores)):
        
#         x_min, y_min, x_max, y_max = bbox
        
#         # Get class name
#         if class_names and 0 <= label < len(class_names):
#             class_name = class_names[label]
#         else:
#             class_name = f"class_{label}"
        
#         color = colors[label % len(colors)]
        
#         # Draw bounding box
#         rect = patches.Rectangle(
#             (x_min, y_min), x_max - x_min, y_max - y_min,
#             linewidth=2, edgecolor=color, facecolor='none'
#         )
#         axes[1, 1].add_patch(rect)
        
#         # Add label with score
#         axes[1, 1].text(x_min, y_min - 5, f"{class_name} ({score:.2f})", 
#                         bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
#                         fontsize=10, color='white', weight='bold')
    
#     # Print attack summary
#     print(f"\n=== Attack Summary ===")
#     print(f"Number of ground truth targets: {len(processed_annotations['boxes'])}")
#     print(f"Original image detections: {len(original_detections['boxes'])}")
#     print(f"Patched image detections: {len(patched_detections['boxes'])}")
    
#     # Calculate attack effectiveness
#     if len(original_detections['boxes']) > 0:
#         detection_change = len(patched_detections['boxes']) - len(original_detections['boxes'])
#         detection_reduction = (len(original_detections['boxes']) - len(patched_detections['boxes'])) / len(original_detections['boxes']) * 100
#         print(f"Detection count change: {detection_change:+d}")
#         print(f"Detection reduction: {detection_reduction:.1f}%")

#     plt.tight_layout()
    
#     # Use timestamp for unique filename
#     timestamp = int(time.time())
#     save_path = os.path.join(save_dir, f"attack_comparison_{timestamp}.png")
#     plt.savefig(save_path, dpi=150, bbox_inches='tight')
#     print(f"Attack comparison saved to: {save_path}")
  
#     plt.close()

def visualize_attack_comparison(
    processed_image: np.ndarray,
    processed_annotations: np.ndarray,
    patched_image: np.ndarray,
    original_detections: np.ndarray,
    patched_detections: np.ndarray,
    class_names: List[str] = None,
    save_dir: str = None
) -> None:
    """
    Visualize attack comparison: original vs patched image with annotations and detection results.
    
    Args:
        processed_image: Processed image array in BGR format [H,W,C]
        processed_annotations: Ground truth annotations
        patched_image: Patched image array in BGR format [H,W,C]
        original_detections: Detection results on original image
        patched_detections: Detection results on patched image
        class_names: List of class names
        save_dir: Directory to save the comparison image
    """
    # Convert BGR to RGB for visualization
    processed_image_rgb = processed_image[..., ::-1].copy()  # BGR to RGB
    patched_image_rgb = patched_image[..., ::-1].copy()  # BGR to RGB
    
    # Ensure image values are in valid range for matplotlib
    processed_image_rgb = np.clip(processed_image_rgb, 0, 255).astype(np.uint8)
    patched_image_rgb = np.clip(patched_image_rgb, 0, 255).astype(np.uint8)

    img_height, img_width = processed_image_rgb.shape[:2]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Plot 1: Original image with ground truth annotations
    axes[0, 0].imshow(processed_image_rgb)
    axes[0, 0].set_title('Original Image (Ground Truth)', fontsize=14, weight='bold')
    axes[0, 0].axis('off')
    
    # Draw ground truth annotations
    draw_detection_box(axes[0,0], 
                       processed_image_rgb,
                       'Original Image (Ground Truth)',
                       processed_annotations,
                       class_names,
                       colors,
                       is_gt=True)
                       
    # Plot 2: Original image with detection results
    draw_detection_box(axes[0,1], 
                       processed_image_rgb,
                       'Original Image (Detection Results)',
                       original_detections,
                       class_names,
                       colors,
                       is_gt=False)
    
    # Plot 3: Patched image with ground truth annotations
    draw_detection_box(axes[1,0], 
                       patched_image_rgb,
                       'Patched Image (Ground Truth)',
                       processed_annotations,
                       class_names,
                       colors,
                       is_gt=True)
    
    # Plot 4: Patched image with detection results
    draw_detection_box(axes[1,1], 
                       patched_image_rgb,
                       'Patched Image (Detection Results)',
                       patched_detections,
                       class_names,
                       colors,
                       is_gt=False)
    
    # Print attack summary
    print(f"\n=== Attack Summary ===")
    print(f"Number of ground truth targets: {len(processed_annotations['boxes'])}")
    print(f"Original image detections: {len(original_detections['boxes'])}")
    print(f"Patched image detections: {len(patched_detections['boxes'])}")
    
    # Calculate attack effectiveness
    if len(original_detections['boxes']) > 0:
        detection_change = len(patched_detections['boxes']) - len(original_detections['boxes'])
        detection_reduction = (len(original_detections['boxes']) - len(patched_detections['boxes'])) / len(original_detections['boxes']) * 100
        print(f"Detection count change: {detection_change:+d}")
        print(f"Detection reduction: {detection_reduction:.1f}%")

    plt.tight_layout()
    
    # Use timestamp for unique filename
    timestamp = int(time.time())
    save_path = os.path.join(save_dir, f"attack_comparison_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Attack comparison saved to: {save_path}")
  
    plt.close()

def visualize_original_annotations(
    image_path: str, 
    annotations: List[Dict],
    class_names: List[str], 
    save_path: str = None
) -> None:
    """
    Visualize original image annotations with bounding boxes and labels.
    This function works with original image coordinates, not scaled ones.
    
    Args:
        image_path: Path to the image file
        annotations: annotations on original data with format List of [x_min, y_min, x_max, y_max], label
        class_names: List of class names
        save_path: Save path, if None then display the image
    """
    """
    # Get original annotations
    annotations = get_original_annotations(image_path, file_to_annotations)
    
    """    
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
    print(f"Number of annotated objects: {len(annotations)}")
    
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
        print(f"Object {i+1}: {class_name} (ID: {label}) - BBox: [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}], BBox size: {x_max - x_min:.1f} x {y_max - y_min:.1f}")
    
    ax.set_title(f'Original Image Annotations: {os.path.basename(image_path)}', fontsize=14, weight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Original annotation visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# def visualize_training_images(
#     images_directory: str,
#     file_to_annotations: Dict[str, List[Dict]],
#     save_directory: str,
#     max_images: int = 5
# ) -> None:
#     """
#     Visualize training images with their annotations.
    
#     Args:
#         images_directory: Directory containing image files
#         file_to_annotations: Mapping from filename to annotations
#         max_images: Maximum number of images to visualize
#     """
#     print(f"\nVisualizing training images (up to {max_images})...")
    
#     # Create training images directory
#     os.makedirs(save_directory, exist_ok=True)
    
#     image_count = 0
#     filename_list = sorted(os.listdir(images_directory))
#     for filename in filename_list:
#         if image_count >= max_images:
#             break
            
#         if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
#             continue
        
#         image_path = os.path.join(images_directory, filename)
        
#         # Get annotations for this image
#         annotations = get_original_annotations(image_path, file_to_annotations)
        
#         if not annotations: # Only visualize images with annotations
#             print(f"Warning: image {filename} does not have annotations. Skip for visualizeation!")
#             continue
 
#         print(f"Visualizing training image {image_count+1}: {filename}")
#         save_path = os.path.join(save_directory, f'training_image_{image_count+1}.png')
#         visualize_original_annotations(
#             image_path=image_path,
#             file_to_annotations=file_to_annotations,
#             class_names=COCO80_NAMES,
#             save_path=save_path
#         )
#         image_count += 1