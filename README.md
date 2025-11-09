# Football Match Object Detection and Annotation using YOLOv8  

This project demonstrates **real-time object detection and video annotation** using the **YOLOv8** deep learning model. It automatically detects and annotates **players (persons)** and **sports balls** in a football match video. Each detection is labeled with its **class name** and **confidence score**, and the annotated video is saved along with a **CSV file** containing all detections.  

---

## Project Overview  

The aim of this project is to analyze football match footage and visualize object detections such as **players and balls** using the **Ultralytics YOLOv8 model**. The script processes each frame of the video, draws bounding boxes around detected objects, and exports both the **annotated video** and **detection metadata** for further analysis.  

This can be extended for:
- Player tracking  
- Ball movement analysis  
- Sports analytics  
- Automation of video tagging  

---

## Features  

- Detects **persons** and **sports balls** in football videos using YOLOv8  
- Annotates each frame with **bounding boxes** and **confidence scores**  
- Filters detections based on a configurable **confidence threshold**  
- Exports all detection data (class, confidence, coordinates, frame number) to a **CSV file**  
- Displays real-time progress for long videos  

---

## Tech Stack  

| Category | Tools & Libraries |
|-----------|------------------|
| Programming Language | Python |
| Deep Learning Model | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| Data Handling | Pandas |
| Model Framework | PyTorch (via Ultralytics) |

---

## Project Structure  

```
Football-Object-Detection-YOLOv8
├── main.py                      # Main script for video annotation
├── example_video.py             # Example video to annote
├── video_detections.csv         # Exported detections data (generated after run)
├── annotated_video.mp4          # Annotated output video
├── requirements.txt             # Required dependencies
└── README.md                    # Project documentation
```

---

## How It Works  

1. **Load the YOLOv8 model** (`yolov8n.pt`) from Ultralytics.  
2. **Read the input video** using OpenCV.  
3. For each frame:
   - Perform object detection with YOLOv8.  
   - Extract class name, confidence score, and bounding box coordinates.  
   - Filter detections by confidence (≥ 0.5).  
   - Draw bounding boxes and labels on the frame.  
4. **Save**:
   - The annotated video (`annotated_video_confidence.mp4`).  
   - A CSV file (`video_detections.csv`) with all detections.  

---

## 🧾 Example Output  

**Annotations include:**  
- Green bounding boxes for detected objects.  
- Labels showing class (e.g., *person*, *sports ball*) and confidence (e.g., *0.82*).  

*Sample Output Frame:*  
```
person: 0.91
sports ball: 0.87
```

**CSV Output Example:**
| frame | class       | confidence | x1  | y1  | x2  | y2  |
|-------|-------------|-------------|-----|-----|-----|-----|
| 45    | person      | 0.91        | 120 | 210 | 320 | 640 |
| 45    | sports ball | 0.87        | 540 | 380 | 580 | 420 |

---

## 💾 Installation  

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Football-Object-Detection-YOLOv8.git
   cd Football-Object-Detection-YOLOv8
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python pandas
   ```

3. Place your video file inside the project folder and update the `video_path` in the script.

4. Run the project:
   ```bash
   python main.py
   ```

---

## Results  

- **Annotated video** → `annotated_video.mp4`  
- **Detections CSV** → `video_detections.csv`  
- **Confidence threshold** → Adjustable in the script (`confidence_threshold = 0.5`)  

---

## Future Enhancements  

- Add **object tracking** (e.g., using DeepSORT or ByteTrack) to track players and ball movement.  
- Integrate **heatmaps** for player positioning.  
- Build a **dashboard (Streamlit/Power BI)** for interactive analytics.  

---

## Author  

**Tanuj Kaswa**  
AI & ML Engineer | Automation Developer | Data Science Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/tanuj-kaswa-a99aa7212/)  
