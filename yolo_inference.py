# %%
from ultralytics import YOLO
import torch

print(f'Torch version: {torch.version}')
print(f'CUDA version: {torch.version.cuda}')
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")
print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
print(f'Torch version: {torch.__version__}')
print(f'cudnn: {torch.backends.cudnn.enabled}')

#device = torch.device("cpu")

#model = YOLO("yolov8x")
model = YOLO("yolo11n")

#model = model.to(device)

file = 'c:/coding/yolo/media/celtics.mp4'
#file = 'c:/coding/yolo/media/test.mp4'

results = model.track(file, save=True)


#yolo task=detect mode=train model=yolo516u.y