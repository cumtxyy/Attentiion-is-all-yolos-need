from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.train(data='myBddDatasets.yaml',workers=0,epochs=100,batch=16)