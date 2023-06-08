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
import cv2

# Inputs and Hyperparameters for training
pretrained_model = "/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/yolov8/yolov8_robocup_2023/train/yolov8s_308_461_epoch1000_epoch3000_3/weights/best.pt" # default is yolov8n.pt
config_file = "/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/yolov8/yolov8_config_robocup_2023.yaml"
data_yaml = "/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/robocup_2023_dataset_308_461/dataset.yaml"
test_images_dir = "/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/test_images_robocup_2023"
image_size = 640
confidence_threshold = 0.45
iou_threshold = 0.45
half_precision = True # use FP16 half-precision inference (faster, less accurate)
cuda_devices = 'cpu' # GPU devices ids

img = cv2.imread("/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/datasets/test_images_robocup_2023/set2_frame00000.png")

# Load a model
model = YOLO(pretrained_model)  # load a pretrained model (recommended for training)

# Prediction on test images
results = model.predict(source=img,
              conf=confidence_threshold,
              iou=iou_threshold,
              half=half_precision,
              device=cuda_devices,
              save=False)

# convert results to numpy array
new_results = results[0].boxes.numpy()
class_ids = new_results.cls
class_names = results[0].names
class_names = [class_names[i] for i in class_ids]
class_scores = new_results.conf
class_bboxes = new_results.xyxy

print(f"Class names are: {class_names}")
print(f"Class scores are: {class_scores}")
print(f"Class bboxes are: {class_bboxes}")

# save the image with bounding boxes
img_bbox = results[0].plot()
cv2.imwrite("test_image_bbox.png", img_bbox)

# To use it on CLI