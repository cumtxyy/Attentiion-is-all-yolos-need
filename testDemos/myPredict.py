from ultralytics import YOLO

# 加载预训练模型
model = YOLO('../models/yolov8m.pt')

# 执行预测
results = model.predict(
    source='/data/AIfusion/liuzhongyun/Datasets/BDD100K/bdd100k_images/bdd100k/images/100k/test/cabc30fc-e7726578.jpg',
    project="runs",  # 自定义根目录
    show=False,
    save=True,
    conf=0.25,
    save_txt=False,
    save_conf=False,
    line_width=1  # 设置为1-3的整数，数值越小线条越细
)

# from ultralytics import YOLO
#
# model = YOLO('../runs/detect/train/weights/best.pt')
# results = model.predict(
#     source='D://testData',
#     stream=True,  # <-- Mandatory for large datasets
#     line_width=1,  # 设置为1-3的整数，数值越小线条越细
#     save=True,
#     conf=0.25,
#     save_txt=False,
#     save_conf=False,
#     show=False,
# )
#
# # 启用流式推理模式 节约内存占用
# # 添加stream=True参数，将推理结果转换为生成器（generator），逐帧释放内存
# for result in results:  # 逐帧处理，内存占用降低60-80%
#     boxes = result.boxes  # 实时处理并释放内存
#     masks = result.masks  # Segmentation masks (also affected)
#     probs = result.probs  # Classification outputs