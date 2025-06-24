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
import torchvision
import argparse
import json
import yaml
import pprint
import matplotlib.patches as patches
import logging

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import DPatch

from art.tools.coco_categories90 import COCO_INSTANCE_CATEGORY_NAMES
from art.tools.plot_utils import plot_image_with_boxes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

def extract_predictions(predictions_):

    # Get the predicted class
    predictions_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions_["labels"])]
    print("\npredicted classes:", predictions_class)

    # Get the predicted bounding boxes
    predictions_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions_["boxes"])]

    # Get the predicted prediction score
    predictions_score = list(predictions_["scores"])
    print("predicted score:", predictions_score)

    # Get a list of index with score greater than threshold
    threshold = 0.5
    predictions_t = [predictions_score.index(x) for x in predictions_score if x > threshold][-1]

    predictions_boxes = predictions_boxes[: predictions_t + 1]
    predictions_class = predictions_class[: predictions_t + 1]
    predictions_score = predictions_score[: predictions_t + 1]

    return predictions_class, predictions_boxes, predictions_score
def get_loss(frcnn, x, y):
    frcnn._model.train()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor_list = list()

    for i in range(x.shape[0]):
        if frcnn.clip_values is not None:
            img = transform(x[i] / frcnn.clip_values[1]).to(frcnn._device)
        else:
            img = transform(x[i]).to(frcnn._device)
        image_tensor_list.append(img)

    loss = frcnn._model(image_tensor_list, y)
    for loss_type in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        loss[loss_type] = loss[loss_type].cpu().detach().numpy().item()
    return loss


def append_loss_history(loss_history, output):
    for loss in ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]:
        loss_history[loss] += [output[loss]]
    return loss_history


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
            "path": "results/dpatch_fasterrcnn",
        }

    os.makedirs(config["path"], exist_ok=True)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    if config["cuda_visible_devices"] is None:
        device_type = "cpu"
    else:
        device_type = "gpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), channels_first=False, attack_losses=config["attack_losses"], device_type=device_type
    )

    image_1 = cv2.imread(config["image_file"])
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_1 = cv2.resize(image_1, dsize=(image_1.shape[1], image_1.shape[0]), interpolation=cv2.INTER_CUBIC)

    image = np.stack([image_1], axis=0).astype(np.float32)

    attack = DPatch(
        frcnn,
        log_dir="results/dpatch_fasterrcnn/log"
    )

    x = image.copy()

    y = frcnn.predict(x=x)

    for i in range(image.shape[0]):
        print(f"\nPredictions clean image {i}:")
        predictions_class, predictions_boxes, predictions_score = extract_predictions(y[i])
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

    if config["resume"]:
        patch = np.load(os.path.join(config["path"], "patch.npy"))
        attack._patch = patch

        with open(os.path.join(config["path"], "loss_history.json"), "r") as file:
            loss_history = json.load(file)
    else:
        loss_history = {"loss_classifier": [], "loss_box_reg": [], "loss_objectness": [], "loss_rpn_box_reg": []}

    for i in range(config["max_iter"]):
        print("Iteration:", i)
        patch = attack.generate(x)
        x_patch = attack.apply_patch(x)

        loss = get_loss(frcnn, x_patch, y)
        print(loss)
        loss_history = append_loss_history(loss_history, loss)

        with open(os.path.join(config["path"], "loss_history.json"), "w") as file:
            file.write(json.dumps(loss_history))

        np.save(os.path.join(config["path"], "patch"), attack._patch)

    predictions_adv = frcnn.predict(x=x_patch)

    for i in range(image.shape[0]):
        print("\nPredictions adversarial image {}:".format(i))
        predictions_adv_class, predictions_adv_boxes, predictions_adv_score = extract_predictions(predictions_adv[i])
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

    # 在攻击前保存patch可视化
    patch_vis_path = os.path.join(config["path"], "patch_visualization.png")
    visualize_patch_only(attack._patch, save_path=patch_vis_path)

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

    attack.close()