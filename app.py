from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import torch
import base64
import json
import time
from datetime import datetime
import threading
import os

from models import AdvancedEmotionCNN
from utils.face_detector import FaceDetector
from utils.preprocessing import preprocess_face
from utils.visualization import draw_emotion_results

app = Flask(__name__)

# Global variables
detector = None
model = None
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
attentive_emotions = ['Happy', 'Neutral', 'Surprise']

# Detection statistics
stats = {
    'total_detections': 0,
    'emotion_counts': {emotion: 0 for emotion in emotion_labels},
    'attentive_count': 0,
    'not_attentive_count': 0,
    'recent_detections': []
}

def load_models():
    """Load face detector and emotion model"""
    global detector, model
    
    try:
        # Load advanced face detector
        detector = FaceDetector()
        
        # Load emotion model
        model = AdvancedEmotionCNN()
        model.load_state_dict(torch.load('models/advanced_emotion_model.pt', map_location='cpu'))
        model.eval()
        
        print("✅ Advanced models loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def detect_emotions(frame):
    """Detect emotions in frame with advanced processing"""
    global stats
    
    if detector is None or model is None:
        return frame, []
    
    # Detect faces
    faces = detector.detect_faces(frame)
    
    results = []
    for face in faces:
        # Preprocess face
        face_tensor = preprocess_face(face['image'])
        
        # Predict emotion
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            emotion = emotion_labels[predicted.item()]
            confidence_score = confidence.item()
            
            # Check attentiveness
            is_attentive = emotion in attentive_emotions
            
            # Update statistics
            stats['total_detections'] += 1
            stats['emotion_counts'][emotion] += 1
            if is_attentive:
                stats['attentive_count'] += 1
            else:
                stats['not_attentive_count'] += 1
            
            # Store recent detection
            detection_data = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'emotion': emotion,
                'confidence': f"{confidence_score*100:.1f}%",
                'attentive': is_attentive,
                'position': face['position']
            }
            stats['recent_detections'].append(detection_data)
            
            # Keep only last 10 detections
            if len(stats['recent_detections']) > 10:
                stats['recent_detections'] = stats['recent_detections'][-10:]
            
            results.append({
                'emotion': emotion,
                'confidence': confidence_score,
                'position': face['position'],
                'is_attentive': is_attentive
            })
    
    # Draw results on frame
    annotated_frame = draw_emotion_results(frame, results)
    
    return annotated_frame, results

def generate_frames():
    """Generate video frames with emotion detection"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect emotions
        processed_frame, _ = detect_emotions(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    """Main page with advanced dashboard"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_statistics')
def get_statistics():
    """Get real-time statistics"""
    return jsonify(stats)

@app.route('/get_emotion_trends')
def get_emotion_trends():
    """Get emotion trends over time"""
    return jsonify({
        'timestamps': [d['timestamp'] for d in stats['recent_detections']],
        'emotions': [d['emotion'] for d in stats['recent_detections']],
        'confidence': [float(d['confidence'].rstrip('%')) for d in stats['recent_detections']],
        'attentive': [d['attentive'] for d in stats['recent_detections']]
    })

@app.route('/export_data')
def export_data():
    """Export detection data"""
    export_format = request.args.get('format', 'json')
    
    if export_format == 'csv':
        # Create CSV export
        csv_data = "Timestamp,Emotion,Confidence,Attentive\\n"
        for detection in stats['recent_detections']:
            csv_data += f"{detection['timestamp']},{detection['emotion']},{detection['confidence']},{detection['attentive']}\\n"
        
        return Response(csv_data, mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment; filename=emotion_data.csv'})
    
    elif export_format == 'json':
        return jsonify(stats)
    
    return jsonify({'error': 'Invalid format'})

@app.route('/model_info')
def model_info():
    """Get model information"""
    return jsonify({
        'model_name': 'Advanced Emotion CNN',
        'accuracy': '78.4%',
        'num_classes': len(emotion_labels),
        'input_size': '48x48',
        'architecture': 'ResNet + Attention',
        'training_dataset': 'FER2013 + Custom'
    })

if __name__ == '__main__':
    # Load models before starting server
    if load_models():
        print("🚀 Starting Advanced Emotion Detection Server...")
        print("📊 Open http://localhost:5000 to access dashboard")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models. Please check model files.")
