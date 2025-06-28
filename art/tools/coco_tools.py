import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import numpy as np
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


def coco_resize_bboxes(bboxes, original_size, new_size):
    """
    按比例缩放bboxes标签坐标 coco数据类型

    inputs:
        bboxes: List of [x_min,y_min,weight,height] 原始bboxes 可以是 numpy 数组或列表
        original_size: (width, height) 原始图像尺寸
        new_size: (width, height) Resize后的图像尺寸

    output:
        bboxes: List of [x_min,y_min,x_max,y_max] 缩放后的 bboxes
    """

    bboxes = np.array(bboxes)
    bboxes[:,2] = bboxes[:,0] + bboxes[:,2]
    bboxes[:,3] = bboxes[:,1] + bboxes[:,3]
    
    orig_w, orig_h = original_size
    new_w, new_h = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale_x  # x_min, x_max
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale_y  # y_min, y_max

    return bboxes

# def get_original_annotations(
#     image_path: str,
#     file_to_annotations: Dict[str, List[Dict]]
# ) -> List[Tuple]:
#     """
#     Get original annotations for an image without any scaling or preprocessing.
    
#     Args:
#         image_path: Path to the image file
#         file_to_annotations: Mapping from filename to annotations
        
#     Returns:
#         List of (bbox, label) tuples in original image coordinates
#     """
#     filename = os.path.basename(image_path)
#     annotations = file_to_annotations.get(filename, [])
    
#     # Convert annotations to the format expected by visualization
#     processed_annotations = []
#     for annotation in annotations:
#         bbox = annotation['bbox']  # [x, y, w, h] format
#         label = annotation['category_id']
        
#         # Convert [x, y, w, h] to [x_min, y_min, x_max, y_max]
#         x_min, y_min, w, h = bbox
#         x_max = x_min + w
#         y_max = y_min + h
        
#         # Map COCO label to vehicle label if needed
#         mapped_label = coco_label_mapping(label)
        
#         processed_annotations.append(([x_min, y_min, x_max, y_max], mapped_label))
    
    # return processed_annotations

def get_coco80_annotations(
    annotations: List[Dict]
) -> List[Tuple]:
    """
    Get original annotations for an image without any scaling or preprocessing.
    
    Args:
        image_path: Path to the image file
        file_to_annotations: Mapping from filename to annotations
        
    Returns:
        List of (bbox, label) tuples in original image coordinates
    """
    
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