import torch

# CUDA 사용 가능 여부 확인
cuda_available = torch.cuda.is_available()
print("CUDA 사용 가능 여부:", cuda_available)

# 사용 가능한 GPU 이름 확인
if cuda_available:
    print("GPU 이름:", torch.cuda.get_device_name(0))
else:
    print("GPU를 사용할 수 없습니다.")
