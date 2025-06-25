import os
import numpy as np
from typing import Dict, List, Tuple
from torchvision import transforms
from PIL import Image

from art.tools.coco_tools import coco_resize_bboxes, coco_label_mapping


SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
MAX_TRAINING_IMAGES = 10 # TODO: 修改迭代器
IMAGE_SIZE = (500, 500)

def load_training_images(
    images_dir: str, 
    file_to_annotations: Dict[str, List[Dict]],
    max_images: int = MAX_TRAINING_IMAGES
) -> Tuple[np.ndarray, List[List[Tuple]]]:
    """
    Load and process multiple images for training.
    
    Args:
        images_dir: Directory containing image files
        file_to_annotations: Mapping from filename to annotations
        max_images: Maximum number of images to load for training
        
    Returns:
        Tuple of (stacked images array, list of annotations for each image)
    """
    transform = transforms.Resize(IMAGE_SIZE)
    
    processed_images = []
    all_annotations = []
    image_count = 0
    
    for filename in sorted(os.listdir(images_dir)):
        if image_count >= max_images:
            break
            
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            image_array, annotations = process_image_file(
                filename, images_dir, transform, file_to_annotations
            )
            
            # Only include images with annotations for training
            if annotations:
                processed_images.append(image_array)
                all_annotations.append(annotations)
                image_count += 1
                print(f"Loaded training image {image_count}: {filename} ({len(annotations)} objects)")
    
    # Stack all images into a single array with shape [B,C,H,W]
    if processed_images:
        images_batch = np.stack(processed_images)
        print(f"Training batch created: {len(processed_images)} images")
    else:
        raise ValueError("No images with annotations found for training")
    
    return images_batch, all_annotations

def process_image_file(
    filename: str, 
    images_dir: str, 
    transform: transforms.Compose,
    file_to_annotations: Dict[str, List[Dict]]
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Process a single image file and extract its annotations.
    
    Args:
        filename: Name of the image file
        images_dir: Directory containing images
        transform: Image transformation pipeline
        file_to_annotations: Mapping from filename to annotations
        
    Returns:
        Tuple of (processed image array, list of (bbox, label) tuples)
    """
    image_path = os.path.join(images_dir, filename)
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    original_shape = image.size
    image = transform(image)  # Resize to align H and W dimensions
    resized_shape = image.size
    
    # Convert to numpy array and change to BGR format
    image_array = np.array(image).astype(np.float32)  # RGB [H,W,C]
    image_array = image_array[..., ::-1]  # Convert RGB to BGR
    image_array = np.transpose(image_array, (2, 0, 1))  # Shape: [C,H,W], range [0.0,255.0]
    
    # Process annotations for this image
    annotations = file_to_annotations.get(filename, [])
    bboxes = [annotation['bbox'] for annotation in annotations]
    bboxes = coco_resize_bboxes(bboxes, original_shape, resized_shape)
    
    labels = [annotation['category_id'] for annotation in annotations]
    labels = [coco_label_mapping(label) for label in labels]

    # processed_annotations = list(zip(bboxes, labels))
    processed_annotations = {'boxes': np.array(bboxes), 
                             'labels': np.array(labels),
                             'scores': np.ones(len(bboxes),dtype=np.float32)}
    
    return image_array, processed_annotations