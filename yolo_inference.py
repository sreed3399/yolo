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


model = YOLO("yolov8x")

results = model.track('c:/coding/yolo/media/celtics.mp4', save=True)


#yolo task=detect mode=train model=yolo516u.y