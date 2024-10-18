from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('/Users/jasminlu/Documents/2024/Semester2/Topics/Orangutan/runs/detect/test6/weights/best.pt')

# Set the source video path
source_video = '/Users/jasminlu/Documents/2024/Semester2/Topics/Orangutan/Videos/new/IMG_5601.MOV'

# Open the video file
cap = cv2.VideoCapture(source_video)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
output_path = 'output_detection.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Perform detection on the frame
    results = model.track(frame, conf=0.5, show=True, save=False, tracker='/Users/jasminlu/Documents/2024/Semester2/Topics/Orangutan/botsort.yaml')
    
    # Get the plotted frame with detections
    annotated_frame = results[0].plot()
    
    # Write the frame with detections to the output video
    out.write(annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()