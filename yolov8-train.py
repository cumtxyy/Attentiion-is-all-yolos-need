# from ultralytics import YOLO
#
# model = YOLO('models/yolov8m.pt')
# model.train(data='myBddDatasets.yaml',workers=0,epochs=120,batch=16)


from ultralytics import YOLO

model = YOLO('models/yolov8m.pt')
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
