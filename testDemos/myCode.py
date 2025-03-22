# import torch
# import numpy as np
# print(np.__version__)  # 应输出 1.23.5
# print(f"PyTorch版本: {torch.__version__}")          # 应输出 2.0.1
# print(f"CUDA是否可用: {torch.cuda.is_available()}") # 必须为 True
# print(f"CUDA版本: {torch.version.cuda}")           # 应与安装命令中的 CUDA 版本一致（如 11.8）
#
# print('Hello World')



from ultralytics import YOLO
# 加载修改后的配置文件
model = YOLO('yolov8_4fca.yaml')

# 打印关键层结构（验证是否成功添加）
print(model.model.model[19])  # 应显示nn.Upsample
print(model.model.model[20])  # 应显示CustomAttention(512)
print(model.model.model[21])  # 应显示Concat


