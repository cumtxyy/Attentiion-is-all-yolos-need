from ultralytics import YOLO

# 加载预训练模型
model = YOLO('../best.pt')

# 执行预测
results = model.predict(
    source='../imageSets',  # 输入源路径
    show=True,         # 实时显示检测结果
    save=True,         # 自动保存带标注的图片
    conf=0.25,         # 置信度阈值(默认值)
    save_txt=False,    # 是否保存检测结果为txt文件
    save_conf=False    # 在保存的txt中是否包含置信度
)

