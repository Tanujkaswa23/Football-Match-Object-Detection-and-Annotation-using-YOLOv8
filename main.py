import cv2
import os
from ultralytics import YOLO
import pandas as pd

# Load YOLO model
model = YOLO("yolov8n.pt")

# Path to the uploaded video
video_path = "example_video.mp4"
output_video_path = "annotated_video.mp4"

# Open the input video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Store all detections for CSV export
all_detections = []
frame_count = 0
confidence_threshold = 0.5  # Only annotate detections with confidence >= 0.5

print("Processing video frames...")
print(f"Using confidence threshold: {confidence_threshold}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the frame
    results = model(frame, verbose=False)

    # Process detections
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Only process detections with confidence >= 0.5
            if conf >= confidence_threshold:
                # Store detection data
                all_detections.append({
                    "frame": frame_count,
                    "class": cls_name,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

                # Draw bounding box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label with class name and confidence
                label = f"{cls_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Write the annotated frame to output video
    out.write(frame)

    frame_count += 1

    # Print progress every 30 frames
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

# Save detections to CSV
if all_detections:
    df = pd.DataFrame(all_detections)
    df.to_csv("video_detections.csv", index=False)
    print(f"Saved {len(all_detections)} detections to video_detections.csv")
    print(f"All detections have confidence >= {confidence_threshold}")
else:
    print(f"No detections found with confidence >= {confidence_threshold}")

print(f"Annotated video saved as: {output_video_path}")
print(f"Total frames processed: {frame_count}")