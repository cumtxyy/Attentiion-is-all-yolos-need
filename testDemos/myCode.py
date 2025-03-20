import torch
import numpy as np
print(np.__version__)  # 应输出 1.23.5
print(f"PyTorch版本: {torch.__version__}")          # 应输出 2.0.1
print(f"CUDA是否可用: {torch.cuda.is_available()}") # 必须为 True
print(f"CUDA版本: {torch.version.cuda}")           # 应与安装命令中的 CUDA 版本一致（如 11.8）

print('Hello World')


