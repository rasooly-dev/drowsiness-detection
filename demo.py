import cv2
from cv2.data import haarcascades
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import threading
from queue import Queue
import logging
from collections import deque
import numpy as np

class DrowsinessDetector:
    def __init__(self):
        # Initialize mixer for alerts
        mixer.init()
        self.sound = mixer.Sound('assets/alert.wav')
        
        # Load models and cascades
        self.face_cascade = cv2.CascadeClassifier(haarcascades + 'haarcascade_frontalface_default.xml')
        self.left_eye_cascade = cv2.CascadeClassifier(haarcascades + 'haarcascade_lefteye_2splits.xml')
        self.right_eye_cascade = cv2.CascadeClassifier(haarcascades + 'haarcascade_righteye_2splits.xml')
        self.eye_model = load_model('models/CNN_eye_normal.keras')
        self.yawn_model = load_model('models/yawn_detection_model.keras')
        
        # Initialize queues and shared variables
        self.frame_queue = Queue(maxsize=30)
        self.result_queue = Queue(maxsize=30)
        self.score_deque = deque(maxlen=10)  # Store last 10 scores for smoothing
        self.running = False
        self.score = 0
        self.score_threshold = 15
        
        # Threading locks
        self.score_lock = threading.Lock()
        self.alert_lock = threading.Lock()
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('DrowsinessDetector')

    def get_eye_position(self, eyes, leftmost=True):
        """Helper function to get leftmost or rightmost eye"""
        if not len(eyes):
            return None
        compare = min if leftmost else max
        return compare(eyes, key=lambda x: x[0])

    def process_frame(self, frame):
        """Process a single frame for drowsiness detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            local_score = 0
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_gray = gray[y:y+h, x:x+w]
                
                # Process yawn
                face_gray_resized = cv2.resize(face_gray, (150, 150))
                face_gray_resized = face_gray_resized.reshape(1, 150, 150, 1) / 255.0
                yawn_pred = self.yawn_model.predict(face_gray_resized, verbose=0)
                if yawn_pred[0] > 0.5:
                    local_score += 2
                
                # Detect and process eyes
                left_eye = self.left_eye_cascade.detectMultiScale(face_gray, 1.3, 5)
                right_eye = self.right_eye_cascade.detectMultiScale(face_gray, 1.3, 5)
                
                for eye in [self.get_eye_position(left_eye, True), 
                           self.get_eye_position(right_eye, False)]:
                    if eye is None:
                        continue
                    
                    ex, ey, ew, eh = eye
                    eye_roi = face[ey:ey+eh, ex:ex+ew]
                    eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                    eye_gray = cv2.resize(eye_gray, (250, 250))
                    eye_gray = eye_gray.reshape(1, 250, 250, 1) / 255.0
                    
                    eye_pred = self.eye_model.predict(eye_gray, verbose=0)
                    if eye_pred[0] < 0.5:
                        local_score += 1.5
                
                # Add drowsiness warning to frame if needed
                if local_score > self.score_threshold:
                    cv2.putText(frame, 'DROWSINESS ALERT!', (x, y-50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
            return frame, local_score
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return frame, 0

    def capture_frames(self):
        """Thread function for capturing frames"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Failed to capture frame")
                break
                
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            
        cap.release()

    def process_frames(self):
        """Thread function for processing frames"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed_frame, local_score = self.process_frame(frame)
                
                if not self.result_queue.full():
                    self.result_queue.put((processed_frame, local_score))

    def update_score_and_alert(self):
        """Thread function for managing score and alerts"""
        while self.running:
            if not self.result_queue.empty():
                _, local_score = self.result_queue.get()
                
                with self.score_lock:
                    self.score_deque.append(local_score)
                    self.score = np.mean(self.score_deque)
                    
                    if self.score > self.score_threshold:
                        with self.alert_lock:
                            if not mixer.get_busy():
                                self.sound.play()
                    else:
                        with self.alert_lock:
                            self.sound.stop()
            
            time.sleep(0.01)  # Small sleep to prevent CPU overuse

    def display_frames(self):
        """Thread function for displaying processed frames"""
        while self.running:
            if not self.result_queue.empty():
                frame, _ = self.result_queue.get()
                
                # Add score display
                with self.score_lock:
                    cv2.putText(frame, f'Score: {self.score:.1f}', (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Drowsiness Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

    def run(self):
        """Main method to run the drowsiness detection system"""
        self.running = True
        
        # Create and start threads
        threads = [
            threading.Thread(target=self.capture_frames),
            threading.Thread(target=self.process_frames),
            threading.Thread(target=self.update_score_and_alert),
            threading.Thread(target=self.display_frames)
        ]
        
        for thread in threads:
            thread.start()
        
        # Wait for threads to complete
        for thread in threads:
            thread.join()
        
        # Cleanup
        cv2.destroyAllWindows()
        self.sound.stop()

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()
