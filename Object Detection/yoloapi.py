from ultralytics import YOLO
import cv2

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
#model.train(data="coco128.yaml", epochs=3)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
results = model("demo2.jpg")  # predict on an image
res_plotted = results[0].plot()
cv2.imwrite("yolo_2.jpg", res_plotted)