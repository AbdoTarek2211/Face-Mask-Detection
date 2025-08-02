# Real-time Face Mask Detection using OpenCV
# This script uses the trained model to detect face masks in real-time through webcam

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from datetime import datetime
import os
import argparse

class FaceMaskDetector:
    def __init__(self, model_path='face_mask_detection_final.h5'):
        """
        Initialize the Face Mask Detector
        
        Args:
            model_path (str): Path to the trained model file
        """
        print("Initializing Face Mask Detector...")
        
        # Load the trained model
        try:
            self.model = load_model(model_path)
            print(f"✓ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            exit(1)
        
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Model parameters
        self.IMG_SIZE = (224, 224)
        self.class_names = ['WithMask', 'WithoutMask']  # Adjust based on your model
        
        # Colors for bounding boxes (BGR format)
        self.colors = {
            'WithMask': (0, 255, 0),      # Green
            'WithoutMask': (0, 0, 255),   # Red
        }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        print("✓ Face Mask Detector initialized successfully!")
    
    def preprocess_face(self, face):
        """
        Preprocess face image for model prediction
        
        Args:
            face (numpy.ndarray): Face image from OpenCV
            
        Returns:
            numpy.ndarray: Preprocessed face image
        """
        # Resize to model input size
        face_resized = cv2.resize(face, self.IMG_SIZE)
        
        # Convert BGR to RGB (OpenCV uses BGR, model expects RGB)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def predict_mask(self, face):
        """
        Predict if face has mask or not
        
        Args:
            face (numpy.ndarray): Preprocessed face image
            
        Returns:
            tuple: (predicted_class, confidence_score)
        """
        # Make prediction
        predictions = self.model.predict(face, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        
        return predicted_class, confidence
    
    def draw_prediction(self, frame, x, y, w, h, prediction, confidence):
        """
        Draw bounding box and prediction on frame
        
        Args:
            frame (numpy.ndarray): Video frame
            x, y, w, h (int): Bounding box coordinates
            prediction (str): Predicted class
            confidence (float): Prediction confidence
        """
        # Choose color based on prediction
        color = self.colors[prediction]
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Prepare label text
        label = f"{prediction}: {confidence:.1%}"
        
        # Get text size for background rectangle
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame, 
            (x, y - text_height - 10), 
            (x + text_width, y), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            frame, 
            label, 
            (x, y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update FPS every 30 frames
            end_time = time.time()
            self.current_fps = self.fps_counter / (end_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_info(self, frame):
        """
        Draw information overlay on frame
        
        Args:
            frame (numpy.ndarray): Video frame
        """
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Info text
        info_text = [
            f"FPS: {self.current_fps:.1f}",
            f"Time: {current_time}",
            "Press 'q' to quit, 's' to screenshot"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw info text
        for i, text in enumerate(info_text):
            cv2.putText(
                frame, 
                text, 
                (15, 30 + i * 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        return frame
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_mask_detection_{timestamp}.jpg"
        
        # Create screenshots directory if it doesn't exist
        os.makedirs("screenshots", exist_ok=True)
        filepath = os.path.join("screenshots", filename)
        
        cv2.imwrite(filepath, frame)
        print(f"Screenshot saved: {filepath}")
    
    def run_realtime_detection(self, camera_id=0, min_confidence=0.5):
        """
        Run real-time face mask detection
        
        Args:
            camera_id (int): Camera device ID (usually 0 for default camera)
            min_confidence (float): Minimum confidence threshold for predictions
        """
        print(f"Starting real-time detection with camera {camera_id}...")
        print(f"Minimum confidence threshold: {min_confidence}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(camera_id)
        
        # Check if camera is opened
        if not cap.isOpened():
            print(f"✗ Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Camera initialized successfully!")
        print("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("✗ Error: Could not read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    # Preprocess face for prediction
                    preprocessed_face = self.preprocess_face(face_roi)
                    
                    # Make prediction
                    prediction, confidence = self.predict_mask(preprocessed_face)
                    
                    # Only draw if confidence is above threshold
                    if confidence >= min_confidence:
                        self.draw_prediction(frame, x, y, w, h, prediction, confidence)
                    else:
                        # Draw gray box for low confidence
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 2)
                        cv2.putText(
                            frame, 
                            f"Low Confidence: {confidence:.1%}", 
                            (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (128, 128, 128), 
                            2
                        )
                
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
            
            # Calculate FPS
            self.calculate_fps()
            
            # Draw information overlay
            frame = self.draw_info(frame)
            
            # Display frame
            cv2.imshow('Face Mask Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_screenshot(frame)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Camera released and windows closed")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Real-time Face Mask Detection')
    parser.add_argument('--model', type=str, default='face_mask_detection_final.h5',
                       help='Path to the trained model file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Minimum confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"✗ Error: Model file '{args.model}' not found!")
        print("Please make sure you have trained the model first.")
        return
    
    try:
        # Initialize detector
        detector = FaceMaskDetector(model_path=args.model)
        
        # Run real-time detection
        detector.run_realtime_detection(
            camera_id=args.camera,
            min_confidence=args.confidence
        )
        
    except KeyboardInterrupt:
        print("\n✓ Detection stopped by user")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    main()