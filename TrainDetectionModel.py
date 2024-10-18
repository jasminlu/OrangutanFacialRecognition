from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8s.pt')

# Train the model
model.train(
    data='/Users/jasminlu/Documents/2024/Semester2/Topics/Orangutan/Data3/data.yaml', 
    epochs=25,        
    imgsz=640,         
    batch=16,          
    name='test'        
)
