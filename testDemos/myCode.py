import torch
import numpy as np
from ultralytics import YOLO


# print(np.__version__)  # 应输出 1.23.5
# print(f"PyTorch版本: {torch.__version__}")          # 应输出 2.0.1
# print(f"CUDA是否可用: {torch.cuda.is_available()}") # 必须为 True
# print(f"CUDA版本: {torch.version.cuda}")           # 应与安装命令中的 CUDA 版本一致（如 11.8）
# print('Hello World')


# 加载修改后的配置文件
model = YOLO('yolov8_4fca.yaml')
# model = YOLO('yolov8.yaml')

# 打印关键层结构（验证是否成功添加）
# print(model.model.model[28])  # 应显示nn.Upsample
# print(model.model.model[27])  # 应显示CustomAttention(512)
# print(model.model.model[28])  # 应显示Concat
# print(model.model.model[26])  # 应显示Concat

# model.info(verbose=True)  # 打印详细结构


# 打印模块结构和参数
# 假设 FCAAttention 是模型的第5层
# fca_module = model.model.model[11]  # 索引从0开始
# print(fca_module)
# print("\n参数列表：")
# for name, param in fca_module.named_parameters():
#     print(f"参数名: {name}")
#     print(f"形状: {param.shape}")
#     print(f"是否可训练: {param.requires_grad}\n")


# YOLOv8_4fca summary: 253 layers, 3,260,198 parameters, 3,260,182 gradients, 8.9 GFLOPs
#YOLOv8 summary: 225 layers, 3,157,200 parameters, 3,157,184 gradients, 8.9 GFLOPs