import torch

if torch.cuda.is_available():
    print(f"GPU 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    print("GPU 사용 불가능")