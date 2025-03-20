from ultralytics import YOLO
model = YOLO('../models/yolov8m.pt')

#官方默认配置
# model.train(data='myBddDatasets.yaml',workers=0,epochs=120,batch=16)

#本地PC 检查配置
model.train(
    data='myBddDatasets.yaml',
    epochs=80,             # 小数据集(2000) 总轮次设置为 80-100
    batch=32,              # 增大batch（需验证显存）
    workers=10,            # 稍低于物理核数(性能核 + 能效核)
    optimizer='SGD',
    lr0=0.003,             # 小数据集 初始学习率0.001 - 0.005
    momentum=0.9,          # 大数据集用0.937 但小数据集可微调至0.9以降低噪声敏感性
    cos_lr=True,           # 启用余弦退火[5](@ref)
    patience=20,           # 早停阈值
    warmup_epochs=5,       # 学习率预热
    device=0,              # 指定GPU索引
    val=False              # 每2-3轮验证一次（节省时间）
)

#服务器 训练配置
model.train(
    data='../myBddDatasets.yaml',
    epochs=150,           # 增大训练轮次（大数据集需要更充分训练）
    batch=64,             # 根据自动检测结果调整
    imgsz=640,
    workers=8,            # 数据加载优化
    amp=True,             # 开启混合精度
    optimizer='AdamW',     # 替代默认 SGD（对大数据集更友好）
    lr0=1e-3,             # 初始学习率（配合 AdamW 使用）
    lrf=0.01,            # 最终学习率 = lr0 * lrf
    weight_decay=0.05,   # 正则化防过拟合
    dropout=0.2,          # 仅在 YOLOv8m 及以上版本生效 训练过程中，以 0.2 的概率随机丢弃网络层中的神经元，迫使模型学习更鲁棒的特征
    translate=0.2,       # 数据增强强度 默认0.1 提高图像随机平移幅度以模拟复杂场景
    mixup=0.2,           # 开启 MixUp 通过混合两张图像提升小目标检测能力
    erasing=0.4,         # 随机擦除 通过随机遮挡局部区域增强模型鲁棒性
    close_mosaic=10,     # 最后 10 轮关闭  默认否 mosaic 在训练最后阶段禁用马赛克增强，避免极端增强干扰收敛
    device=0,            # 明确指定 GPU
    pretrained=True      # 确保使用预训练权重
)
