"""
YOLOv8 Training Script

Source: https://github.com/ultralytics/ultralytics

"""

from ultralytics import YOLO

# Inputs and Hyperparameters for training
pretrained_model = "/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/yolov8/yolov8_robocup_2023/train/yolov8s_epoch3000_dataset_ver12/weights/best.pt" # default is yolov8n.pt

# Load a model
model = YOLO(pretrained_model)  # load a pretrained model (recommended for training)

# Export model to ONNX
path = model.export(format="onnx", opset=12)
print("Exported model to: ", path)

# To use it on CLI
# yolo export model=path/to/best.pt format=onnx opset=12 # export custom trained model