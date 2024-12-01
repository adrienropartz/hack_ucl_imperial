import cv2
import numpy as np
import logging
from datetime import datetime
import time

class VideoProcessor:
    def __init__(self):
        # Setup logging
        logging.basicConfig(
            filename=f'video_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def process_video(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        
        try:
            print("Starting video processing... Press 'q' to quit")
            
            while cap.isOpened():
                # Read a frame from the video
                ret, frame = cap.read()
                
                if not ret:
                    print("\nEnd of video file...")
                    break
                
                # Display the frame
                cv2.imshow('Video Processing', frame)
                
                # Add FPS counter
                current_time = time.time()
                fps = cap.get(cv2.CAP_PROP_FPS)
                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopping video processing...")
                    break
                
        except KeyboardInterrupt:
            print("\nStopping video processing...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Processing complete. Windows closed.")

if __name__ == "__main__":
    processor = VideoProcessor()
    # Replace with your video file path
    video_path = "path/to/your/video.mp4"
    processor.process_video(video_path) 