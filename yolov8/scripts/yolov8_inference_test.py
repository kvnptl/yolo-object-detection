"""
YOLOv8 Training Script

Source: https://github.com/ultralytics/ultralytics

Tunable parameters:

- multi-gpu training
- pretrained model
- batch size
- epochs
- image size
- model name
- cuda devices
- freeze layers
- results directory
- data augmentation

"""

from ultralytics import YOLO

# Inputs and Hyperparameters for training
pretrained_model = "/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/yolov8/yolov8_robocup_2023/train/yolov8s_epoch3000_dataset_ver12/weights/best.pt" # default is yolov8n.pt
test_images_dir = "/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/datasets/att2"
image_size = 640
confidence_threshold = 0.45
iou_threshold = 0.45
half_precision = True # use FP16 half-precision inference (faster, less accurate)
project_name = "yolov8_robocup_2023/predictions"
model_name = "test_" # results are saved in runs/train
cuda_devices = '0' # GPU devices ids

# Load a model
model = YOLO(pretrained_model)  # load a pretrained model (recommended for training)

# Prediction on test images
model.predict(source=test_images_dir,
              conf=confidence_threshold,
              iou=iou_threshold,
              half=half_precision,
              device=cuda_devices,
              save=True,
              project=project_name,
              name=model_name)

# To use it on CLI