from ultralytics import YOLO
import torch
from roboflow import Roboflow


if __name__ == '__main__':

    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # print("Using:", device)
    # torch.cuda.set_device(device)

    rf = Roboflow(api_key="BS8zZk5oG44IuXWegBlr")
    project = rf.workspace("agrobotics").project("yolo_new-4z2qz")
    version = project.version(1)
    dataset = version.download("yolov8")

    model = YOLO("yolov8n.pt")
    # model.class_weights = [3, 3, 1] # adjust weights to emphasize classes with less representation
    model.train(data='yolo_new-1/data.yaml', epochs=300, patience=50)
