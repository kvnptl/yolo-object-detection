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

# Inputs
pretrained_model = 'yolov8s.pt' # default is yolov8n.pt
config_file = "/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/yolov8/yolov8_config_robocup_2023.yaml"
data_yaml = "/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/robocup_2023_dataset_308_461/dataset.yaml"

# Hyperparameters
epochs = 10
image_size = 640
batch_size = 64 # (-1 for AutoBatch, works only for single GPU)
project_name = "yolov8_robocup_2023/train" # save training results to <project-name>/train
file_name = os.path.basename(os.path.splitext(pretrained_model)[0])
model_name = f"{file_name}_epoch{epochs}_" if file_name != 'best' else f"{pretrained_model.split('/')[-3]}_epoch{epochs}_"
cuda_devices = '0' # GPU devices ids 
freeze_layers = 10 # number of layers to freeze (from the beginning)

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = freeze_layers
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

# Load a model
model = YOLO(pretrained_model)  # load a pretrained model (recommended for training)
# model.add_callback("on_train_start", freeze_layer) # DOESN"T IMPROVE PERFORMANCE

# Train a model using custom config yaml file
model.train(cfg=config_file, # custom config file
            data=data_yaml,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project=project_name,
            name=model_name,
            device=cuda_devices,
            pretrained=True,
            )

# TO use it on CLI
# yolo train cfg=/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/yolov8/yolov8_config_ss22.yaml data=/home/kpatel2s/kpatel2s/b_it_bots/2d_object_detection/yolo-object-detection/dataset_ss22_v4.yaml epochs=5 imgsz=640 batch=64 project=yolov8_ss22_v4 name=test_ device=0,1 pretrained=True

# NOTE: To free up GPU memory, follow these steps:
# https://askubuntu.com/a/1118325/922137