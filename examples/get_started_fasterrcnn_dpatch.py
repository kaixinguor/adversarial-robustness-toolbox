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
from art.attacks.evasion import DPatch

from art.tools.coco_categories90 import COCO_INSTANCE_CATEGORY_NAMES
from art.tools.preprocess_utils import process_image_file, create_batch_loader
from art.tools.plot_utils import visualize_attack_comparison, visualize_training_images
from art.tools.fasterrcnn_utils import extract_predictions, get_loss, append_loss_history
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

    attack = DPatch(
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
    plt.title("DPatch Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config["training_log_dir"], "loss_curve.png"), dpi=150, bbox_inches='tight')
    # plt.show()
    
    attack.close()

def attack_test(config, frcnn, attack):
    
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

    attack = DPatch(
        frcnn
    )
        
    # Get image filename from path
    image_path = config["image_file"]
    filename = os.path.basename(image_path)
    images_dir = os.path.dirname(image_path)
    # Use the same transform as training process
    transform = transforms.Resize(config["image_size"])  # Same as IMAGE_SIZE in train_dpatch.py
    processed_image_array, processed_annotations = process_image_file(
        filename, images_dir, transform, file_to_annotations
    )
    # Convert processed image back to RGB format for visualization
    # processed_image_array is [C,H,W] in BGR format, convert to RGB [H,W,C]
    processed_image = np.transpose(processed_image_array, (1, 2, 0))  # [H,W,C]
    processed_image = processed_image[..., ::-1]  # BGR to RGB
    processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

    image = np.stack([processed_image], axis=0).astype(np.float32)
    x = image.copy()
    y = frcnn.predict(x=x)
        
    print(f"\nPredictions clean image {i}:")
    predictions_class, predictions_boxes, predictions_score = extract_predictions(y[i], class_names=COCO_INSTANCE_CATEGORY_NAMES)

    if frcnn._device == "cpu":
        logging.warning("FasterRCNN is running on CPU, which is not recommended for training.")

        for i, y_i in enumerate(y):
            y[i]["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(frcnn._device)
            y[i]["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(frcnn._device)
            y[i]["scores"] = torch.from_numpy(y_i["scores"]).to(frcnn._device)

    x_patch = attack.apply_patch(x)  
    predictions_adv = frcnn.predict(x=x_patch)

    print("\nPredictions adversarial image {}:".format(i))
    predictions_adv_class, predictions_adv_boxes, predictions_adv_score = extract_predictions(predictions_adv[i], class_names=COCO_INSTANCE_CATEGORY_NAMES)


    # 保存对比图
    # Get image dimensions
    img_height, img_width = processed_image.shape[:2]
    visualize_attack_comparison(
        img_width=img_width,
        img_height=img_height,
        processed_image=processed_image,
        patched_image=x_patch[i].copy(),
        processed_annotations=processed_annotations,
        original_detections=y[i],
        patched_detections=predictions_adv[i],
        clean_img=x[i].copy(),
        clean_boxes=predictions_boxes,
        clean_cls=predictions_class,
        adv_img=x_patch[i].copy(),
        adv_boxes=predictions_adv_boxes,
        adv_cls=predictions_adv_class,
        class_names=COCO_INSTANCE_CATEGORY_NAMES,
        clean_scores=list(y[i]["scores"]) if "scores" in y[i] else None,
        adv_scores=list(predictions_adv[i]["scores"]) if "scores" in predictions_adv[i] else None,
        save_path=os.path.join(config["training_comparison_dir"], f"attack_comparison_{i}.png")
    )

    attack.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional DPatch training for FasterRCNN")
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
            "training_output_dir": "results/dpatch_fasterrcnn/training_output",
            "training_visualization_dir": "results/dpatch_fasterrcnn/training_output/visualization",
            "training_images_save_dir": "results/dpatch_fasterrcnn/training_output/images",
            "training_comparison_dir": "results/dpatch_fasterrcnn/training_output/comparison",
            "training_log_dir": "results/dpatch_fasterrcnn/training_output/log",
            "training_patch_dir": "results/dpatch_fasterrcnn/training_output/patch",
            
            # test parameters
            "image_file": "dataset/vehicle_images_5/images/000000000471.jpg",
            "resume": False,
            "image_size": (500, 500), # Same as IMAGE_SIZE in train_dpatch.py
    }

    if cmdline.mode == "train":
        print("Starting professional DPatch training...")
        attack_train(config)
    elif cmdline.mode == "test":
        print("Starting DPatch testing...")
        # Note: attack_test function needs to be updated to work with the new batch system
        # For now, we'll just call attack_train as a placeholder
        attack_test(config)
