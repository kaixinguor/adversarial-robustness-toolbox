# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import json
import yaml
import pprint
import matplotlib.patches as patches
import logging
import os.path as osp

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import DPatch
from art.tools.coco_categories90 import COCO_INSTANCE_CATEGORY_NAMES
from art.tools.plot_utils import plot_image_with_boxes
from art.tools.fasterrcnn_utils import extract_predictions, get_loss, append_loss_history
from art.tools.preprocess_utils import load_training_images
from art.tools.coco_tools import load_annotation_data, visualize_training_images
from art.tools.patch_utils import save_trained_patch, visualize_patch_only


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

def visualize_attack_comparison(clean_img, clean_boxes, clean_cls, adv_img, adv_boxes, adv_cls, class_names=None, clean_scores=None, adv_scores=None, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    # 原图
    axes[0].imshow(clean_img.astype(np.uint8))
    axes[0].set_title('Original Image (Detection)', fontsize=16, weight='bold')
    axes[0].axis('off')
    colors = [
        'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
        'cyan', 'magenta', 'yellow', 'lime', 'teal', 'gold', 'navy', 'maroon'
    ]
    for i in range(len(clean_boxes)):
        color = colors[i % len(colors)]
        box = clean_boxes[i]
        rect = patches.Rectangle(
            (int(box[0][0]), int(box[0][1])),
            int(box[1][0] - box[0][0]),
            int(box[1][1] - box[0][1]),
            linewidth=3, edgecolor=color, facecolor='none', alpha=0.8
        )
        axes[0].add_patch(rect)
        label = clean_cls[i] if class_names is None else class_names[clean_cls[i]] if isinstance(clean_cls[i], int) else clean_cls[i]
        if clean_scores is not None:
            label = f"{label} ({clean_scores[i]:.2f})"
        axes[0].text(
            int(box[0][0]), int(box[0][1]) - 10,
            label,
            fontsize=14, color='white', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7)
        )
    # 攻击后
    axes[1].imshow(adv_img.astype(np.uint8))
    axes[1].set_title('Patched Image (Detection)', fontsize=16, weight='bold')
    axes[1].axis('off')
    for i in range(len(adv_boxes)):
        color = colors[i % len(colors)]
        box = adv_boxes[i]
        rect = patches.Rectangle(
            (int(box[0][0]), int(box[0][1])),
            int(box[1][0] - box[0][0]),
            int(box[1][1] - box[0][1]),
            linewidth=3, edgecolor=color, facecolor='none', alpha=0.8
        )
        axes[1].add_patch(rect)
        label = adv_cls[i] if class_names is None else class_names[adv_cls[i]] if isinstance(adv_cls[i], int) else adv_cls[i]
        if adv_scores is not None:
            label = f"{label} ({adv_scores[i]:.2f})"
        axes[1].text(
            int(box[0][0]), int(box[0][1]) - 10,
            label,
            fontsize=14, color='white', weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7)
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default=None, help="Path of config yaml file")
    cmdline = parser.parse_args()

    if cmdline.config and os.path.exists(cmdline.config):
        with open(cmdline.config, "r") as cf:
            config = yaml.safe_load(cf.read())
    else:
        config = {
            "attack_losses": ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
            "cuda_visible_devices": "1",
            # "crop_range": [0, 0],
            # "brightness_range": [1.0, 1.0],
            # "rotation_weights": [1, 0, 0, 0],
            # "sample_size": 1,
            "max_iter": 1,
            "image_file": "dataset/vehicle_images_5/images/000000000471.jpg",
            "resume": False,
            
            "training_output_dir": "results/dpatch_fasterrcnn/training_output",
            "training_visualization_dir": "results/dpatch_fasterrcnn/training_output/visualization",
            "training_images_save_dir": "results/dpatch_fasterrcnn/training_output/images",
            "training_comparison_dir": "results/dpatch_fasterrcnn/training_output/comparison",
            "training_log_dir": "results/dpatch_fasterrcnn/training_output/log",
            "training_patch_dir": "results/dpatch_fasterrcnn/training_output/patch",
            
            "annotation_path": 'dataset/vehicle_coco2017/annotations/instances_vehicle_train2017.json',
            "image_directory": "dataset/vehicle_coco2017/images/train2017",
            "config_file": "mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_coco-dpatch.py",
            "checkpoint_file": "mmdetection/weights/faster_rcnn_r101_fpn_1x_coco.pth"
            
            
        }

    # Create output directories
    os.makedirs(config["training_visualization_dir"], exist_ok=True)
    os.makedirs(config["training_images_save_dir"], exist_ok=True)
    os.makedirs(config["training_comparison_dir"], exist_ok=True)
    os.makedirs(config["training_log_dir"], exist_ok=True)
    os.makedirs(config["training_patch_dir"], exist_ok=True)

    # Load annotation data
    _, file_to_annotations = load_annotation_data(config["annotation_path"])
    
    # Visualize training images
    visualize_training_images(config["image_directory"], file_to_annotations, config["training_images_save_dir"], max_images=5)
    
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(config)

    if config["cuda_visible_devices"] is None:
        device_type = "cpu"
    else:
        device_type = "gpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), channels_first=True, attack_losses=config["attack_losses"], device_type=device_type
    )

    attack = DPatch(
        frcnn,
        log_dir=config["training_log_dir"]
    )

    # 读取图片
    logging.info(f"Reading image from {config['image_directory']}")
    image_batch, annotations_batch = load_training_images(config["image_directory"], file_to_annotations)


    # image = np.stack([image_1], axis=0).astype(np.float32)
    logging.info(f"Training batch prepared: {image_batch.shape}")
    logging.info(f"Number of training images: {len(annotations_batch)}")
    
    # training procedure
    if config["resume"]:
        patch = np.load(os.path.join(config["training_patch_dir"], "patch.npy"))
        attack._patch = patch

        with open(os.path.join(config["training_log_dir"], "loss_history.json"), "r") as file:
            loss_history = json.load(file)
    else:
        loss_history = {"loss_classifier": [], "loss_box_reg": [], "loss_objectness": [], "loss_rpn_box_reg": []}
        
    for i in range(config["max_iter"]):
        print("Iteration:", i)
        patch = attack.generate(image_batch)
        x_patch = attack.apply_patch(image_batch)

        loss = get_loss(frcnn, x_patch, annotations_batch)
        logging.info(f"Loss for iteration {i}: {loss}")
        loss_history = append_loss_history(loss_history, loss)

        with open(os.path.join(config["training_log_dir"], "loss_history.json"), "w") as file:
            file.write(json.dumps(loss_history))

        # Save trained patch to file
        logging.info("Saving trained patch to file...")
        patch_file_path = save_trained_patch(attack._patch, config["training_patch_dir"], "trained_patch")
        
        # Save patch visualization
        print("Saving trained patch visualization...")
        patch_viz_path = os.path.join(config["training_patch_dir"], f'trained_patch.png')
        visualize_patch_only(attack._patch, save_path=patch_viz_path)
    
    # 训练后绘制loss曲线
    loss_history_path = os.path.join(config["path"], "loss_history.json")
    with open(loss_history_path, "r") as f:
        loss_history = json.load(f)

    plt.figure(figsize=(10, 6))
    for loss_name, loss_values in loss_history.items():
        plt.plot(loss_values, label=loss_name)
    plt.xlabel("Iteration")
    plt.ylabel("Loss Value")
    plt.title("DPatch Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config["path"], "loss_curve.png"))
    plt.show()

    # test procedure
    image_1 = cv2.imread(config["image_file"])
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_1 = cv2.resize(image_1, dsize=(image_1.shape[1], image_1.shape[0]), interpolation=cv2.INTER_CUBIC)

    image = np.stack([image_1], axis=0).astype(np.float32)
    x = image.copy()
    y = frcnn.predict(x=x)

    for i in range(image.shape[0]):
        print(f"\nPredictions clean image {i}:")
        predictions_class, predictions_boxes, predictions_score = extract_predictions(y[i], class_names=COCO_INSTANCE_CATEGORY_NAMES)
        plot_image_with_boxes(
            img=x[i].copy(),
            boxes=predictions_boxes,
            pred_cls=predictions_class,
            title="Original Image (Detection)",
            scores=list(y[i]["scores"]) if "scores" in y[i] else None
        )

    for i, y_i in enumerate(y):
        y[i]["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(frcnn._device)
        y[i]["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(frcnn._device)
        y[i]["scores"] = torch.from_numpy(y_i["scores"]).to(frcnn._device)

    x_patch = attack.apply_patch(x)  
    predictions_adv = frcnn.predict(x=x_patch)

    for i in range(image.shape[0]):
        print("\nPredictions adversarial image {}:".format(i))
        predictions_adv_class, predictions_adv_boxes, predictions_adv_score = extract_predictions(predictions_adv[i], class_names=COCO_INSTANCE_CATEGORY_NAMES)
        plot_image_with_boxes(
            img=x_patch[i].copy(),
            boxes=predictions_adv_boxes,
            pred_cls=predictions_adv_class,
            title="Patched Image (Detection)",
            scores=predictions_adv_score
        )
        # 保存对比图
        visualize_attack_comparison(
            clean_img=x[i].copy(),
            clean_boxes=predictions_boxes,
            clean_cls=predictions_class,
            adv_img=x_patch[i].copy(),
            adv_boxes=predictions_adv_boxes,
            adv_cls=predictions_adv_class,
            class_names=COCO_INSTANCE_CATEGORY_NAMES,
            clean_scores=list(y[i]["scores"]) if "scores" in y[i] else None,
            adv_scores=list(predictions_adv[i]["scores"]) if "scores" in predictions_adv[i] else None,
            save_path=os.path.join(config["path"], f"attack_comparison_{i}.png")
        )

    attack.close()