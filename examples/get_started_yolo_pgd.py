"""
The script demonstrates a simple example of using ART with YOLO (versions 3 and 5).
The example loads a YOLO model pretrained on the COCO dataset
and creates an adversarial example using Projected Gradient Descent method.

- To use Yolov5, run:
        pip install yolov5

Note: If pytorchyolo throws an error in pytorchyolo/utils/loss.py, add before line 174 in that file, the following:
        gain = gain.to(torch.int64)
"""

import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch

from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.attacks.evasion import ProjectedGradientDescent

import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from art.tools.coco_categories80 import COCO_INSTANCE_CATEGORY_NAMES as COCO80_NAMES
from art.tools.plot_utils import plot_image_with_boxes
from art.tools.fasterrcnn_utils import extract_predictions


"""
#################        Model definition        #################
"""
MODEL = "yolov5"  # OR yolov5

import yolov5
from yolov5.utils.loss import ComputeLoss

matplotlib.use("TkAgg")

class Yolo(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.hyp = {
            "box": 0.05,
            "obj": 1.0,
            "cls": 0.5,
            "anchor_t": 4.0,
            "cls_pw": 1.0,
            "obj_pw": 1.0,
            "fl_gamma": 0.0,
        }
        self.compute_loss = ComputeLoss(self.model.model.model)

    def forward(self, x, targets=None):
        if self.training:
            outputs = self.model.model.model(x)
            loss, loss_items = self.compute_loss(outputs, targets)
            loss_components_dict = {"loss_total": loss}
            return loss_components_dict
        else:
            return self.model(x)

model = yolov5.load("models/yolov5/yolov5s.pt")

model = Yolo(model)

detector = PyTorchYolo(
    model=model, device_type="gpu", input_shape=(3, 640, 640), clip_values=(0, 255), attack_losses=("loss_total",)
)


"""
#################        Example image        #################
"""
# response = requests.get("https://ultralytics.com/images/zidane.jpg")
# img = np.asarray(Image.open(BytesIO(response.content)).resize((640, 640)))

local_img_path = "dataset/vehicle_images_5/images/000000000471.jpg"
img = np.asarray(Image.open(local_img_path).resize((640, 640)))
image = np.stack([img], axis=0).astype(np.float32)
image_chw = np.transpose(image, (0, 3, 1, 2))

"""
#################        Evasion attack        #################
"""

eps = 32
attack = ProjectedGradientDescent(estimator=detector, eps=eps, eps_step=2, max_iter=10)
image_adv_chw = attack.generate(x=image_chw, y=None)
image_adv = np.transpose(image_adv_chw, (0, 2, 3, 1))

print("\nThe attack budget eps is {}".format(eps))
print("The resulting maximal difference in pixel values is {}.".format(np.amax(np.abs(image_chw - image_adv_chw))))

plt.axis("off")
plt.title("adversarial image")
plt.imshow(image_adv[0].astype(np.uint8), interpolation="nearest")
plt.show()

predictions = detector.predict(x=image_chw)
predictions_class, predictions_boxes, predictions_scores = extract_predictions(predictions[0], class_names=COCO80_NAMES)
plot_image_with_boxes(
    img=image[0], boxes=predictions_boxes, pred_cls=predictions_class, 
    title="Predictions on original image", scores=predictions_scores
)
print("before attack:", predictions_class, predictions_boxes, predictions_scores)

predictions = detector.predict(image_adv_chw)
predictions_class, predictions_boxes, d = extract_predictions(predictions[0], class_names=COCO80_NAMES)
plot_image_with_boxes(
    img=image_adv[0],
    boxes=predictions_boxes,
    pred_cls=predictions_class,
    title="Predictions on adversarial image",
    scores=d
)
print("after attack:", predictions_class, predictions_boxes, d)
