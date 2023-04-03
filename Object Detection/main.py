from ultralytics import YOLO
import cv2
model=YOLO("yolov8n.pt")
result=model("demo.jpg")
cv2.imwrite("yolo_demo.jpg",result[0].plot())