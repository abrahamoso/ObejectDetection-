
# Object Detection and Hand Signal Recognition System  

This project implements an **object detection and hand signal recognition system** using **YOLOv5** and **OpenCV**. It detects and classifies objects in real time while also recognizing **hand signals** such as **left, right, up, down, and stop**. The system is designed for integration into larger projects, including **drone control systems**, to enable gesture-based commands.  

## Features  
- **Real-Time Object Detection:** Uses YOLOv5 for fast and accurate identification of objects.  
- **Hand Signal Recognition:** Detects predefined gestures (left, right, up, down, stop) for interaction and control.  
- **Performance Optimization:** Supports GPU acceleration for faster processing.  
- **Modular Design:** Easily extendable for hardware integration and advanced functionalities.  
- **Drone Integration Ready:** Compatible with drone control systems (details available in the [Drone Project Repository](#)).  

## Applications  
- **Gesture-Controlled Systems** – Use hand signals to control robots, drones, or other devices.  
- **Surveillance Systems** – Detect unauthorized movements and hand gestures for security monitoring.  
- **Interactive Interfaces** – Create systems that respond to human gestures for automation and remote operations.  

## Getting Started  
1. **Clone the Repository:**  
   ```bash  
   git clone https://github.com/abrahamoso/ObjectDetection.git  
   cd ObjectDetection  
   ```  
2. **Install Dependencies:**  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. **Run the System:**  
   ```bash  
   python detect.py --source 0 --weights yolov5s.pt --conf 0.4  
   ```  
