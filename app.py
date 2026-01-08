"""
Production App - EXACT Emergency.py Configuration
Proven settings that work in diagnostic
"""

import os
import threading
import queue
import time
import signal
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, render_template, Response
from collections import deque

from crowd_monitor import CrowdMonitor

# Configuration - EXACT emergency.py settings
YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'models/yolov8m.onnx')
VIDEO_PATH = os.environ.get('VIDEO_PATH', 'demo1.mp4')
PHYSICAL_AREA_M2 = float(os.environ.get('PHYSICAL_AREA_M2', 100.0))
ROLLING_WINDOW = int(os.environ.get('ROLLING_WINDOW', 30))
VIDEO_LOOP = os.environ.get('VIDEO_LOOP', 'true').lower() == 'true'

# Performance settings
FRAME_SKIP = int(os.environ.get('FRAME_SKIP', 2))
RESIZE_SCALE = float(os.environ.get('RESIZE_SCALE', 1.0))

# Camera settings
CAMERA_ANGLE = os.environ.get('CAMERA_ANGLE', 'birds_eye')
VIDEO_QUALITY = os.environ.get('VIDEO_QUALITY', 'medium')

# Box drawing
DRAW_BOXES = os.environ.get('DRAW_BOXES', 'true').lower() == 'true'
BOX_COLOR = (0, 255, 0)
BOX_THICKNESS = 2  # EXACT emergency.py thickness

# Thread communication
data_queue = queue.Queue()
video_frame_queue = queue.Queue(maxsize=2)
shutdown_event = threading.Event()

app = Flask(__name__, static_folder='static', template_folder='templates')

class ExactEmergencyWorker(threading.Thread):
    """Worker with EXACT emergency.py configuration"""
    
    def __init__(self, monitor, data_queue, video_queue, shutdown_event, video_path):
        super().__init__(daemon=True)
        self.monitor = monitor
        self.data_queue = data_queue
        self.video_queue = video_queue
        self.shutdown_event = shutdown_event
        self.video_path = video_path
        self.cap = None
        self.video_fps = 30
        self.total_frames = 0
        self.frame_skip = FRAME_SKIP
        self.resize_scale = RESIZE_SCALE
        self.last_detections = []
        self.last_count = 0
        self._init_video()

    def _init_video(self):
        """Initialize video - EXACT emergency.py method"""
        try:
            if not os.path.exists(self.video_path):
                print(f"❌ Video not found: {self.video_path}")
                return
            
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"❌ Could not open: {self.video_path}")
                self.cap = None
                return
            
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.frame_time = 1.0 / self.video_fps if self.video_fps > 0 else 0.033
            
            print(f"✅ Video loaded: {width}x{height} @ {self.video_fps:.2f} FPS")
            print(f"   Total frames: {self.total_frames}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.cap = None

    def run(self):
        """Video processing loop with EXACT emergency.py settings"""
        if self.cap is None:
            print("❌ No video source")
            return
        
        frame_count = 0
        process_count = 0
        start_time = time.time()
        
        print("\n🚀 Processing with EXACT emergency.py configuration...\n")
        print("   Confidence: 0.001 (EXACT)")
        print("   IOU: 0.70 (EXACT)")
        print("   Input size: 640 (EXACT)")
        print()
        
        while not self.shutdown_event.is_set():
            loop_start = time.time()
            
            ret, frame = self.cap.read()
            
            if not ret:
                if VIDEO_LOOP:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress = (current_frame / self.total_frames * 100) if self.total_frames > 0 else 0
            
            should_process = (frame_count % self.frame_skip == 0)
            
            if should_process:
                process_frame = frame if self.resize_scale == 1.0 else cv2.resize(
                    frame, None, fx=self.resize_scale, fy=self.resize_scale
                )
                
                # Get detections using EXACT emergency.py method
                detections = self.monitor.detector.detect(process_frame)
                self.last_detections = detections
                self.last_count = len(detections)
                
                # Update monitor
                self.monitor.current_people_count = self.last_count
                self.monitor.current_density = self.last_count / self.monitor.physical_area_m2
                max_density = 10.0
                self.monitor.current_density_percent = min(
                    (self.monitor.current_density / max_density) * 100, 100.0
                )
                self.monitor.people_count_history.append(self.last_count)
                self.monitor.density_history.append(self.monitor.current_density)
                
                # Update predictor
                self.monitor.predictor.update(self.last_count, time.time())
                
                # Calculate surge and risk
                self.monitor.current_surge_score = self.monitor.predictor.calculate_surge_score()
                self.monitor.current_risk_level = self.monitor._calculate_risk_level_by_density()
                
                stats = self.monitor.get_current_stats()
                stats['timestamp'] = datetime.utcnow().isoformat() + "Z"
                stats['frame_count'] = frame_count
                stats['video_frame'] = current_frame
                stats['video_progress'] = progress
                stats['fps'] = self.video_fps
                
                process_count += 1
                
                try:
                    if self.data_queue.full():
                        self.data_queue.get_nowait()
                    self.data_queue.put_nowait(stats)
                except:
                    pass
                
                # Print every 30 frames
                if process_count % 30 == 0:
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed if elapsed > 0 else 0
                    density = self.monitor.current_density
                    predictions = stats.get('predictions', {})
                    print(f"Frame {current_frame:5d} | "
                          f"People: {self.last_count:3d} | "
                          f"Density: {density:.2f} p/m² | "
                          f"Risk: {stats['risk_level']:8s} | "
                          f"Pred +5s: {predictions.get('predicted_5s', 0):3d} | "
                          f"FPS: {actual_fps:.1f}")
            
            # Draw with EXACT emergency.py style
            display_frame = self._draw_exact_emergency_style(frame)
            
            try:
                if not self.video_queue.full():
                    if frame.shape[1] > 1280:
                        display_scale = 1280 / frame.shape[1]
                        display_frame = cv2.resize(display_frame, None, fx=display_scale, fy=display_scale)
                    self.video_queue.put_nowait(display_frame)
            except:
                pass
            
            frame_count += 1
            
            loop_time = time.time() - loop_start
            sleep_time = self.frame_time - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self._cleanup()

    def _draw_exact_emergency_style(self, frame):
        """Draw detection boxes EXACT emergency.py style"""
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Get current metrics
        metrics = self.monitor.get_detailed_metrics()
        
        # Draw detection boxes - EXACT emergency.py style
        if DRAW_BOXES and len(self.last_detections) > 0:
            for det in self.last_detections:
                x1, y1, x2, y2, conf = det
                
                # EXACT emergency.py box drawing
                cv2.rectangle(result, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                
                # EXACT emergency.py confidence label
                if conf > 0.01:  # Show if confidence > 1%
                    label = f"{conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(result, 
                                (x1, y1 - label_size[1] - 4), 
                                (x1 + label_size[0], y1),
                                BOX_COLOR, -1)
                    cv2.putText(result, label, (x1, y1 - 2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Overlay
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (11, 15, 26), -1)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        # Title
        cv2.putText(result, "CROWD MONITORING - EMERGENCY CONFIG", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 157), 2)
        
        # Detection info
        detection_text = f"DETECTED: {self.last_count} people | Density: {metrics['density_per_m2']:.2f} p/m²"
        cv2.putText(result, detection_text, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(result, f"Area: {metrics['physical_area']:.0f} m²", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Stats panel
        panel_x = w - 400
        panel_y = 15
        panel_h = 120
        
        cv2.rectangle(result, (panel_x, panel_y), (w-15, panel_y+panel_h), (20, 28, 45), -1)
        cv2.rectangle(result, (panel_x, panel_y), (w-15, panel_y+panel_h), (0, 212, 255), 2)
        
        cv2.putText(result, f"COUNT: {self.last_count}", (panel_x+10, panel_y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 157), 2)
        
        cv2.putText(result, f"DENSITY: {metrics['density_per_m2']:.2f} p/m²", (panel_x+10, panel_y+50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2)
        
        cv2.putText(result, f"CROWDING: {metrics['crowding_percentage']}", (panel_x+10, panel_y+75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        cv2.putText(result, "EMERGENCY CONFIG", (panel_x+10, panel_y+100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Risk indicator
        risk = metrics['risk_level']
        risk_colors = {
            'LOW': (0, 255, 157),
            'MEDIUM': (0, 204, 255),
            'HIGH': (0, 165, 255),
            'CRITICAL': (51, 51, 255),
            'EMERGENCY': (0, 0, 255)
        }
        risk_color = risk_colors.get(risk, (0, 255, 157))
        
        cv2.putText(result, f"RISK: {risk}", (20, h-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, risk_color, 3)
        
        cv2.putText(result, metrics['risk_description'], (20, h-25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
        
        return result

    def _cleanup(self):
        if self.cap:
            self.cap.release()
            print("🔹 Video released")

# Initialize
print("\n" + "="*70)
print("🚀 EXACT EMERGENCY.PY CONFIGURATION")
print("="*70)
print(f"Model: {YOLO_MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")
print(f"Physical Area: {PHYSICAL_AREA_M2} m²")
print(f"Confidence: 0.001 (EXACT emergency.py)")
print(f"IOU: 0.70 (EXACT emergency.py)")
print("="*70 + "\n")

if not os.path.exists(VIDEO_PATH):
    print(f"❌ Video not found: {VIDEO_PATH}")
    exit(1)

if not os.path.exists(YOLO_MODEL_PATH):
    print(f"❌ Model not found: {YOLO_MODEL_PATH}")
    exit(1)

try:
    # Get actual frame size
    cap_temp = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height
    cap_temp.release()
    
    # Initialize with EXACT emergency.py settings
    crowd_monitor = CrowdMonitor(
        yolo_model_path=YOLO_MODEL_PATH,
        frame_area=frame_area,
        conf_threshold=0.001,  # EXACT emergency.py
        rolling_window=ROLLING_WINDOW,
        tracker_max_disappeared=20,
        tracker_max_distance=150.0,
        video_quality=VIDEO_QUALITY,
        camera_angle=CAMERA_ANGLE
    )
    
    # Set actual physical area
    crowd_monitor.set_physical_area(PHYSICAL_AREA_M2)
    
    print("✅ Monitor initialized with EXACT emergency.py config\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

worker_thread = ExactEmergencyWorker(crowd_monitor, data_queue, video_frame_queue, shutdown_event, VIDEO_PATH)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route("/")
def index():
    """Main dashboard page"""
    return render_template("index_advanced.html")

@app.route("/help")
def help_center():
    """Help center page"""
    return render_template("help_center.html")

@app.route("/manual")
def user_manual():
    """User manual page"""
    return render_template("user_manual.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    try:
        latest_stats = None
        while not data_queue.empty():
            latest_stats = data_queue.get_nowait()
        
        if latest_stats is None:
            stats = crowd_monitor.get_current_stats()
            stats['timestamp'] = datetime.utcnow().isoformat() + "Z"
        else:
            stats = latest_stats
        
        # Get detailed metrics
        metrics = crowd_monitor.get_detailed_metrics()
        
        # Get predictions
        predictions = stats.get('predictions', {
            'current': stats.get('people_count', 0),
            'predicted_5s': stats.get('people_count', 0),
            'predicted_10s': stats.get('people_count', 0),
            'predicted_15s': stats.get('people_count', 0),
            'trend': 'stable',
            'confidence': 0.0
        })
        
        response = {
            'people_count': stats.get('people_count', 0),
            'density': stats.get('density', 0.0),
            'density_percent': stats.get('density_percent', 0.0),
            'physical_area': stats.get('physical_area', 100.0),
            'risk_level': stats.get('risk_level', 'LOW'),
            'risk_description': metrics['risk_description'],
            'surge_score': stats.get('surge_score', 0.0),
            'timestamp': stats.get('timestamp', datetime.utcnow().isoformat() + "Z"),
            'video_progress': stats.get('video_progress', 0),
            'is_safe': metrics['is_safe'],
            'requires_action': metrics['requires_action'],
            'is_emergency': metrics['is_emergency'],
            'predictions': predictions
        }
        return jsonify(response)
    except Exception as ex:
        return jsonify({'error': str(ex)}), 500

def generate_video_stream():
    """Generate video stream"""
    while True:
        try:
            if not video_frame_queue.empty():
                frame = video_frame_queue.get(timeout=0.1)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.01)
        except:
            time.sleep(0.01)

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

def start_worker():
    if not worker_thread.is_alive():
        worker_thread.start()

def signal_handler(sig, frame):
    print("\n🛑 Shutting down...")
    shutdown_event.set()
    worker_thread.join(timeout=5)
    print("👋 Goodbye!")
    os._exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    start_worker()
    
    print("\n" + "="*70)
    print("🌐 SERVER STARTED - EXACT EMERGENCY CONFIG")
    print("="*70)
    print("🔗 Dashboard: http://localhost:5000")
    print("🔗 Help Center: http://localhost:5000/help")
    print("🔗 User Manual: http://localhost:5000/manual")
    print("🔗 API: http://localhost:5000/api/status")
    print(f"🔗 Physical area: {PHYSICAL_AREA_M2} m²")
    print("\n💡 Settings (EXACT emergency.py):")
    print(f"   Confidence: 0.001")
    print(f"   IOU: 0.70")
    print(f"   Input size: 640")
    print("="*70 + "\n")
    print("⚡ Press Ctrl+C to stop\n")
    
    try:
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    except Exception as e:
        print(f"❌ Error: {e}")
        shutdown_event.set()

if __name__ == "__main__":
    main()