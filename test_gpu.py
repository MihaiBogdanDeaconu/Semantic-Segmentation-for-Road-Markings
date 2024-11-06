import torch
import os

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    print("CUDA is available! PyTorch can use the GPU.")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU device: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")


# Define the path you want to check
path = '/shares/CC_v_Val_FV_Gen3_all/SemsegCrosswalkRestrictedArea/train_rgb/images/DS-CN_13R7C_20180508_142445_f000550_fc00191778_4d87dc.png'

# print(os.listdir(path))
# Check if the path exists
if os.path.exists(path):
    print("The path exists.")
else:
    print("The path does not exist.")