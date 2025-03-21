from ultralytics import YOLO
model = YOLO('../models/yolov8m.pt')

#官方默认配置
# model.train(data='myBddDatasets.yaml',workers=0,epochs=120,batch=16)

#服务器 训练配置
model.train(
    data='../myBddDatasets.yaml',
    epochs=150,           # 增大训练轮次（大数据集需要更充分训练）
    batch=64,             # 根据自动检测结果调整
    imgsz=640,
    workers=8,            # 数据加载优化
    optimizer='AdamW',     # 替代默认 SGD（对大数据集更友好）
    lr0=1e-3,             # 初始学习率（配合 AdamW 使用）
    lrf=0.01,            # 最终学习率 = lr0 * lrf
    patience=20,         # 早停
    device=1,            # 明确指定 GPU
)
