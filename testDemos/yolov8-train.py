# from ultralytics import YOLO
#
# # model = YOLO('../models/yolov8m.pt')
#
# model = YOLO('../ultralytics/cfg/models/v8/yolov8_4fca.yaml').load('../models/yolov8m.pt')
#
# #官方默认配置
# # model.train(data='myBddDatasets.yaml',workers=0,epochs=120,batch=16)
#
# #服务器 训练配置
# model.train(
#     data='../myBddDatasets.yaml',
#     epochs=150,           # 增大训练轮次（大数据集需要更充分训练）
#     batch=64,             # 根据自动检测结果调整
#     imgsz=640,
#     workers=8,            # 数据加载优化
#     optimizer='AdamW',     # 替代默认 SGD（对大数据集更友好）
#     lr0=1e-3,             # 初始学习率（配合 AdamW 使用）
#     lrf=0.01,            # 最终学习率 = lr0 * lrf
#     patience=20,         # 早停
#     device=1,            # 明确指定 GPU
# )

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import FCAAttention

# --- 步骤1：定义层号映射 ---
layer_mapping = {
    11: 12,    # 示例：原第11层 -> 新第12层
    12: 13,
    13: 14,
    14: 16,
    15: 17,
    16: 18,
    17: 20,
    18: 21,
    19: 22,
    20: 24,
    21: 25,
    22: 26,     #别忘了检测头的mapping
}

# --- 步骤2：重命名权重键 ---
pretrained = torch.load('../models/yolov8m.pt')['model'].state_dict()
new_state_dict = {}

for key, value in pretrained.items():
    parts = key.split('.')
    if parts[0] == 'model' and parts[1].isdigit():
        old_layer = int(parts[1])
        if old_layer in layer_mapping:
            parts[1] = str(layer_mapping[old_layer])
    new_key = '.'.join(parts)
    new_state_dict[new_key] = value

# --- 步骤3：加载并初始化 ---
model = YOLO('../ultralytics/cfg/models/v8/yolov8_4fca.yaml')
load_result = model.model.load_state_dict(new_state_dict, strict=False)



#
# #服务器 训练配置
# # model.train(
# #     data='../myBddDatasets.yaml',
# #     epochs=150,           # 增大训练轮次（大数据集需要更充分训练）
# #     batch=64,             # 根据自动检测结果调整
# #     imgsz=640,
# #     workers=8,            # 数据加载优化
# #     optimizer='AdamW',     # 替代默认 SGD（对大数据集更友好）
# #     lr0=1e-3,             # 初始学习率（配合 AdamW 使用）
# #     lrf=0.01,            # 最终学习率 = lr0 * lrf
# #     patience=20,         # 早停
# #     device=1,            # 明确指定 GPU
# # )

# model = YOLO('../ultralytics/cfg/models/v8/yolov8_4fca.yaml')
