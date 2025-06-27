import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import json
import logging
import yaml
from torchvision import transforms

from art.estimators.object_detection import PyTorchFasterRCNN
from art.attacks.evasion import RobustDPatch

from art.tools.coco_categories90 import COCO_INSTANCE_CATEGORY_NAMES as COCO90_NAMES
from art.tools.preprocess_utils import process_image_file, create_batch_loader
from art.tools.plot_utils import visualize_attack_comparison, visualize_training_images
from art.tools.fasterrcnn_utils import get_loss, append_loss_history
from art.tools.coco_tools import load_annotation_data
from art.tools.patch_utils import save_trained_patch, visualize_patch_only

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

def attack_train(config):
    # Create output directories
    os.makedirs(config["training_visualization_dir"], exist_ok=True)
    os.makedirs(config["training_comparison_dir"], exist_ok=True)
    os.makedirs(config["training_log_dir"], exist_ok=True)
    os.makedirs(config["training_patch_dir"], exist_ok=True)

    # Load annotation data
    _, file_to_annotations = load_annotation_data(config["annotation_path"])
    
    # Visualize training images
    visualize_training_images(config["image_directory"], file_to_annotations, config["training_images_save_dir"], max_images=5)

    if config["cuda_visible_devices"] is None:
        device_type = "cpu"
    else:
        device_type = "gpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), channels_first=False, attack_losses=config["attack_losses"], device_type=device_type
    )

    attack = RobustDPatch(
        frcnn,
        log_dir=config["training_log_dir"]
    )

    # Create professional batch data loader
    logging.info(f"Creating batch data loader for {config['image_directory']}")
    batch_loader = create_batch_loader(
        images_dir=config["image_directory"],
        file_to_annotations=file_to_annotations,
        batch_size=config.get("batch_size", 8),
        image_size=config["image_size"],
        shuffle=config.get("shuffle", True),
        max_images=config.get("max_training_images", None),
        seed=config.get("random_seed", 42)
    )
    
    logging.info(f"Training setup: {len(batch_loader)} batches, {batch_loader.num_images} total images")
    
    # training procedure
    if config["resume"]:
        patch = np.load(os.path.join(config["training_patch_dir"], "patch.npy"))
        attack._patch = patch

        with open(os.path.join(config["training_log_dir"], "loss_history.json"), "r") as file:
            loss_history = json.load(file)
    else:
        loss_history = {"loss_classifier": [], "loss_box_reg": [], "loss_objectness": [], "loss_rpn_box_reg": []}
        
    # Training loop with proper batch iteration
    for epoch in range(config["max_epochs"]):
        logging.info(f"Starting epoch {epoch + 1}/{config['max_epochs']}")
        
        epoch_losses = []
        
        for batch_idx, (image_batch, annotations_batch) in enumerate(batch_loader):
            logging.info(f"Processing batch {batch_idx + 1}/{len(batch_loader)} (epoch {epoch + 1})")
            
            # Generate patch for this batch
            patch = attack.generate(image_batch)
            x_patch = attack.apply_patch(image_batch)
            
            # Convert annotations_batch to tensor
            for i in range(len(annotations_batch)):
                annotations_batch[i]["boxes"] = torch.from_numpy(annotations_batch[i]["boxes"]).type(torch.float).to(frcnn._device)
                annotations_batch[i]["labels"] = torch.from_numpy(annotations_batch[i]["labels"]).type(torch.int64).to(frcnn._device)
                annotations_batch[i]["scores"] = torch.from_numpy(annotations_batch[i]["scores"]).type(torch.float).to(frcnn._device)

            # Calculate loss
            loss, loss_sum = get_loss(frcnn, x_patch, annotations_batch)
            epoch_losses.append(loss_sum)
            
            logging.info(f"Batch {batch_idx + 1} loss: {loss}")
            
            # Update loss history
            loss_history = append_loss_history(loss_history, loss)

            # Save loss history periodically
            if (batch_idx + 1) % config.get("save_interval", 10) == 0:
                with open(os.path.join(config["training_log_dir"], "loss_history.json"), "w") as file:
                    file.write(json.dumps(loss_history))

        # End of epoch processing
        avg_epoch_loss = np.mean(epoch_losses)
        logging.info(f"Epoch {epoch + 1} average loss: {avg_epoch_loss}")
        
        # Save trained patch after each epoch
        logging.info("Saving trained patch to file...")
        patch_file_path = save_trained_patch(attack._patch, config["training_patch_dir"], f"trained_patch_epoch_{epoch + 1}")
        patch_file_path = save_trained_patch(attack._patch, config["training_patch_dir"], f"patch")

        # Save patch visualization
        print("Saving trained patch visualization...")
        patch_viz_path = os.path.join(config["training_patch_dir"], f'trained_patch_epoch_{epoch + 1}.png')
        visualize_patch_only(attack._patch, save_path=patch_viz_path)
    
        
    # 训练后绘制loss曲线
    loss_history_path = os.path.join(config["training_log_dir"], "loss_history.json")
    with open(loss_history_path, "r") as f:
        loss_history = json.load(f)

    plt.figure(figsize=(12, 8))
    for loss_name, loss_values in loss_history.items():
        plt.plot(loss_values, label=loss_name, alpha=0.8)
    plt.xlabel("Training Step")
    plt.ylabel("Loss Value")
    plt.title("RobustDPatch Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config["training_log_dir"], "loss_curve.png"), dpi=150, bbox_inches='tight')
    # plt.show()
    
    attack.close()

def attack_test(config):
    
    os.makedirs(config["test_output_dir"], exist_ok=True)
    
    # Load annotation data
    _, file_to_annotations = load_annotation_data(config["annotation_path"])

    if config["cuda_visible_devices"] is None:
        device_type = "cpu"
    else:
        device_type = "gpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    frcnn = PyTorchFasterRCNN(
        clip_values=(0, 255), channels_first=False, attack_losses=config["attack_losses"], device_type=device_type
    )

    attack = RobustDPatch(
        frcnn
    )
        
    # Get image filename from path
    image_path = config["image_file"]
    filename = os.path.basename(image_path)
    images_dir = os.path.dirname(image_path)
    
    # Use the same transform as training process
    transform = transforms.Resize(config["image_size"])
    processed_image_array, processed_annotations = process_image_file(
        filename, images_dir, transform, file_to_annotations
    )
    
    # processed_image_array is [H,W,C] in BGR format from process_image_file
    logging.info(f"Processed image shape: {processed_image_array.shape}")
    logging.info(f"Processed image dtype: {processed_image_array.dtype}")
    logging.info(f"Processed image range: [{processed_image_array.min():.2f}, {processed_image_array.max():.2f}]")

    logging.info(f"\nPredict on original image {image_path}")
    image_batch = np.stack([processed_image_array], axis=0).astype(np.float32)
    x = image_batch.copy()
    predictions = frcnn.predict(x=x)

    # Load patch
    patch_path = os.path.join(config["training_patch_dir"], "patch.npy")
    logging.info(f"Load trained patch from path: {patch_path}")
    patch = np.load(patch_path)
    attack._patch = patch

    # predict on adversarial image
    logging.info(f"\nPredict on adversarial image {image_path}")
    x_patch = attack.apply_patch(x, random_location=True)  
    predictions_adv = frcnn.predict(x=x_patch)

    # Save comparison visualization
    logging.info("Generating attack comparison visualization...")
    visualize_attack_comparison(
        processed_image=image_batch[0],  # [H,W,C] BGR format
        processed_annotations=processed_annotations,
        patched_image=x_patch[0],  # [H,W,C] BGR format
        original_detections=predictions[0],
        patched_detections=predictions_adv[0],
        class_names=COCO90_NAMES,
        save_dir=config["test_output_dir"]
    )

    attack.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RobustDPatch training for FasterRCNN")
    parser.add_argument("--config", required=False, default=None, help="Path of config yaml file")
    parser.add_argument("--mode", choices=["train", "test"], default="train", help="Training or testing mode")
    cmdline = parser.parse_args()

    if cmdline.config and os.path.exists(cmdline.config):
        with open(cmdline.config, "r") as cf:
            config = yaml.safe_load(cf.read())
        logging.info(f"Loaded configuration from {cmdline.config}")
    else:
        logging.info("Using default configuration")
        config = {
            "attack_losses": ["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
            "cuda_visible_devices": "1",
            
            # Professional training parameters
            "max_epochs": 1,  # Number of training epochs
            "batch_size": 1,  # Batch size for training
            "shuffle": False,  # Whether to shuffle the dataset
            "max_training_images": None,  # None for all images, or specify a number
            "random_seed": 42,  # Random seed for reproducibility
            "save_interval": 10,  # Save loss history every N batches

            # training input parameters
            "annotation_path": 'dataset/coco2017/annotations/instances_train2017.json',
            "image_directory": "dataset/vehicle_coco2017/images/train2017",
            "config_file": "mmdetection/configs/faster_rcnn/faster-rcnn_r101_fpn_1x_coco-dpatch.py",
            "checkpoint_file": "mmdetection/weights/faster_rcnn_r101_fpn_1x_coco.pth",
            
            # training output parameters
            "training_output_dir": "results/robustdpatch_fasterrcnn/training_output",
            "training_visualization_dir": "results/robustdpatch_fasterrcnn/training_output/visualization",
            "training_images_save_dir": "results/robustdpatch_fasterrcnn/training_output/images",
            "training_comparison_dir": "results/robustdpatch_fasterrcnn/training_output/comparison",
            "training_log_dir": "results/robustdpatch_fasterrcnn/training_output/log",
            "training_patch_dir": "results/robustdpatch_fasterrcnn/training_output/patch",
            
            # test parameters
            "image_file": "dataset/vehicle_images_5/images/000000000471.jpg",
            "resume": False,
            "image_size": (500, 500),
            "test_output_dir": "results/robustdpatch_fasterrcnn/test_output",
    }

    if cmdline.mode == "train":
        print("Starting RobustDPatch training...")
        attack_train(config)
    elif cmdline.mode == "test":
        print("Starting RobustDPatch testing...")
        attack_test(config)
