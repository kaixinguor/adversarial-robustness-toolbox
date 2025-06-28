import os
import numpy as np
from typing import Dict, List, Tuple, Iterator, Optional
from torchvision import transforms
from PIL import Image
import random


from art.tools.coco_tools import coco_resize_bboxes, get_coco80_annotations
from art.tools.plot_utils import visualize_original_annotations
from art.tools.coco_categories80 import COCO_INSTANCE_CATEGORY_NAMES as COCO80_NAMES
import logging

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')
MAX_TRAINING_IMAGES = 10 # TODO: 修改迭代器
IMAGE_SIZE = (500, 500)

class BatchDataLoader:
    """
    Professional batch data loader for large-scale training.
    Supports shuffling, batching, and efficient memory usage.
    """
    
    def __init__(
        self,
        images_dir: str,
        file_to_annotations: Dict[str, List[Dict]],
        batch_size: int = 8,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        shuffle: bool = True,
        max_images: Optional[int] = None,
        seed: Optional[int] = None,
        channels_first: bool = True
    ):
        """
        Initialize the batch data loader.
        
        Args:
            images_dir: Directory containing image files
            file_to_annotations: Mapping from filename to annotations
            batch_size: Number of images per batch
            image_size: Target image size (height, width)
            shuffle: Whether to shuffle the dataset
            max_images: Maximum number of images to use (None for all)
            seed: Random seed for reproducibility
            channel_first: Whether to output images in channel-first format [C,H,W] (True) 
                          or channel-last format [H,W,C] (False)
        """
        self.images_dir = images_dir
        self.file_to_annotations = file_to_annotations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.max_images = max_images
        self.seed = seed
        self.channel_first = channels_first
        
        # Initialize transform
        self.transform = transforms.Resize(image_size)
        
        # Get all valid image files
        self.image_files = self._get_valid_image_files()
        
        if self.max_images:
            self.image_files = self.image_files[:self.max_images]
        
        self.num_images = len(self.image_files)
        self.num_batches = (self.num_images + batch_size - 1) // batch_size
        
        print(f"BatchDataLoader initialized:")
        print(f"  - Total images: {self.num_images}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Number of batches: {self.num_batches}")
        print(f"  - Image size: {image_size}")
        print(f"  - Channel format: {'First [C,H,W]' if channels_first else 'Last [H,W,C]'}")

        print(f"training file names: {self.image_files}")

    
    def _get_valid_image_files(self) -> List[str]:
        """Get list of image files that have annotations."""
        valid_files = []
        
        for filename in sorted(os.listdir(self.images_dir)):
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                # Check if image has annotations
                if filename in self.file_to_annotations and self.file_to_annotations[filename]:
                    valid_files.append(filename)
        
        return valid_files
    
    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
        """Create iterator for batches."""
        return self._batch_iterator()
    
    def _batch_iterator(self) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
        """Generate batches of images and annotations."""
        # Set random seed if specified
        if self.seed is not None:
            random.seed(self.seed)
        
        # Create copy of files for shuffling
        files_to_process = self.image_files.copy()
        
        if self.shuffle:
            random.shuffle(files_to_process)
        
        # Process batches
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_images)
            
            batch_files = files_to_process[start_idx:end_idx]
            
            # Load batch
            batch_images, batch_annotations = self._load_batch(batch_files)
            
            yield batch_images, batch_annotations
    
    def _load_batch(self, batch_files: List[str]) -> Tuple[np.ndarray, List[Dict]]:
        """Load a batch of images and their annotations."""
        batch_images = []
        batch_annotations = []
        
        for filename in batch_files:
            try:
                image_array, annotations = process_image_file(
                    filename, self.images_dir, self.transform, self.file_to_annotations, self.channel_first
                )
                
                if annotations and len(annotations['boxes']) > 0:
                    batch_images.append(image_array)
                    batch_annotations.append(annotations)
                    
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")
                continue
        
        if not batch_images:
            raise ValueError("No valid images found in batch")
        
        # Stack images into batch
        images_batch = np.stack(batch_images)
        
        return images_batch, batch_annotations
    
    def visualize_annotated_images(self, save_dir, max_images=5):
        """Visualize images with their annotations."""
            
        print(f"\nVisualizing images (up to {max_images})...")
        
        # Create visualization directory
        os.makedirs(save_dir, exist_ok=True)
 
        for image_count, filename in enumerate(self.image_files[:max_images]):

            image_path = os.path.join(self.images_dir, filename)
            original_annotations = self.file_to_annotations.get(filename,[])

            annotations = get_coco80_annotations(original_annotations)
            
            if not annotations:  # Only visualize images with annotations
                print(f"Warning: image {filename} does not have annotations. Skip for visualization!")
                continue
     
            print(f"Visualizing training image {image_count+1}: {filename}")
            save_path = os.path.join(save_dir, f'training_image_{image_count+1}_{os.path.splitext(filename)[0]}.png')
            self._visualize_single_image(
                image_path=image_path,
                annotations=annotations,
                save_path=save_path
            )

    def _visualize_single_image(self, image_path: str, annotations: List[Tuple], save_path: str):
        """
        Visualize a single image with its annotations.
        
        Args:
            image_path: Path to the image file
            annotations: List of (bbox, label) tuples
            save_path: Path to save the visualization
        """
        visualize_original_annotations(
            image_path=image_path,
            annotations=annotations,
            class_names=COCO80_NAMES,
            save_path=save_path
        )

def create_batch_loader(
    images_dir: str,
    file_to_annotations: Dict[str, List[Dict]],
    batch_size: int = 8,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    shuffle: bool = True,
    max_images: Optional[int] = None,
    seed: Optional[int] = None,
    channels_first: bool = True
) -> BatchDataLoader:
    """
    Create a batch data loader for training.
    
    Args:
        images_dir: Directory containing image files
        file_to_annotations: Mapping from filename to annotations
        batch_size: Number of images per batch
        image_size: Target image size (height, width)
        shuffle: Whether to shuffle the dataset
        max_images: Maximum number of images to use (None for all)
        seed: Random seed for reproducibility
        channels_first: Whether to output images in channel-first format [C,H,W] (True) 
                      or channel-last format [H,W,C] (False)
        
    Returns:
        BatchDataLoader instance
    """
    return BatchDataLoader(
        images_dir=images_dir,
        file_to_annotations=file_to_annotations,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        max_images=max_images,
        seed=seed,
        channels_first=channels_first
    )

# def load_training_images(
#     images_dir: str, 
#     file_to_annotations: Dict[str, List[Dict]],
#     max_images: int = MAX_TRAINING_IMAGES,
#     channel_first: bool = True
# ) -> Tuple[np.ndarray, List[List[Tuple]]]:
#     """
#     Load and process multiple images for training.
    
#     Args:
#         images_dir: Directory containing image files
#         file_to_annotations: Mapping from filename to annotations
#         max_images: Maximum number of images to load for training
#         channel_first: Whether to output images in channel-first format [C,H,W] (True) 
#                       or channel-last format [H,W,C] (False)
        
#     Returns:
#         Tuple of (stacked images array, list of annotations for each image)
#     """
#     transform = transforms.Resize(IMAGE_SIZE)
    
#     processed_images = []
#     all_annotations = []
#     image_count = 0

#     training_filenames = []
#     for filename in sorted(os.listdir(images_dir)):
#         if image_count >= max_images:
#             break
            
#         if filename.lower().endswith(SUPPORTED_EXTENSIONS):
#             image_array, annotations = process_image_file(
#                 filename, images_dir, transform, file_to_annotations, channel_first
#             )
            
#             # Only include images with annotations for training
#             if annotations:
#                 processed_images.append(image_array)
#                 all_annotations.append(annotations)
#                 image_count += 1
#                 print(f"Loaded training image {image_count}: {filename} ({len(annotations['boxes'])} objects)")
#                 training_filenames.append(filename)
    
#     # Stack all images into a single array with shape [B,C,H,W] or [B,H,W,C]
#     if processed_images:
#         images_batch = np.stack(processed_images)
#         print(f"Training batch created: {len(processed_images)} images")
#         print(f"Image format: {'[B,C,H,W]' if channel_first else '[B,H,W,C]'}")
#     else:
#         raise ValueError("No images with annotations found for training")
    
#     return images_batch, all_annotations, training_filenames

def process_image_file(
    filename: str, 
    images_dir: str, 
    transform: transforms.Compose,
    file_to_annotations: Dict[str, List[Dict]],
    channels_first: bool = True
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Process a single image file and extract its annotations.
    
    Args:
        filename: Name of the image file
        images_dir: Directory containing images
        transform: Image transformation pipeline
        file_to_annotations: Mapping from filename to annotations
        channel_first: Whether to output images in channel-first format [C,H,W] (True) 
                      or channel-last format [H,W,C] (False)
        
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
    
    # Apply channel format based on parameter
    if channels_first:
        image_array = np.transpose(image_array, (2, 0, 1))  # Shape: [C,H,W], range [0.0,255.0]
    # If channels_first=False, keep as [H,W,C] format
    
    # Process annotations for this image
    annotations = file_to_annotations.get(filename, [])
    bboxes = [annotation['bbox'] for annotation in annotations]
    bboxes = coco_resize_bboxes(bboxes, original_shape, resized_shape)
    
    labels = [annotation['category_id'] for annotation in annotations]
    # labels = [coco_label_mapping(label) for label in labels]

    # processed_annotations = list(zip(bboxes, labels))
    processed_annotations = {'boxes': np.array(bboxes), 
                             'labels': np.array(labels),
                             'scores': np.ones(len(bboxes),dtype=np.float32)}
    
    return image_array, processed_annotations