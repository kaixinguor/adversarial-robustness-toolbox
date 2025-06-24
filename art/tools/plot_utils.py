import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

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
