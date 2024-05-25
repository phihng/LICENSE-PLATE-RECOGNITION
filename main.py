from ultralytics import YOLO
# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8x.pt')
# Train the model using the 'mydata.yaml' dataset for 3 epochs
results = model.train(data='mydata.yaml', epochs=50, device='0')
