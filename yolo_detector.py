"""
EXACT YOLO Detector - Based on Proven Emergency.py Configuration
Uses EXACT same settings that work in emergency diagnostic
"""
import cv2
import numpy as np
from typing import List, Tuple
import threading

class YOLODetector:
    """
    Detector using EXACT emergency.py proven configuration
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.001):
        # EXACT emergency.py settings
        self.conf_threshold = 0.001  # EXACT - proven in emergency
        self.iou_threshold = 0.70    # EXACT - proven in emergency
        self.person_class_id = 0
        
        # EXACT emergency.py input size
        self.input_size = 640
        
        print(f"🚀 EXACT Emergency Config Detector")
        print(f"   Model: {model_path}")
        print(f"   Confidence: {self.conf_threshold} (EXACT emergency.py)")
        print(f"   IOU: {self.iou_threshold} (EXACT emergency.py)")
        print(f"   Input size: {self.input_size} (EXACT emergency.py)")
        
        self._load_model(model_path)
        self.lock = threading.RLock()

    def _load_model(self, model_path: str):
        """Load model - EXACT emergency.py method"""
        try:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            layer_names = self.net.getLayerNames()
            
            # EXACT emergency.py layer extraction
            try:
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            except:
                self.output_layers = self.net.getUnconnectedOutLayersNames()
            
            if not self.output_layers:
                self.output_layers = ['output0']
            
            print(f"✅ Model loaded: {len(layer_names)} layers")
            print(f"   Output layers: {self.output_layers}")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise

    def set_angle_mode(self, angle: str):
        """Set camera angle - keeps emergency.py settings"""
        print(f"📐 Camera angle: {angle}")

    def set_quality_mode(self, quality: str):
        """Set quality mode - keeps emergency.py settings"""
        print(f"📹 Video quality: {quality}")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect people - EXACT emergency.py method
        """
        with self.lock:
            return self._detect_emergency_exact(frame)

    def _detect_emergency_exact(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        EXACT copy of emergency.py detection logic
        """
        orig_h, orig_w = frame.shape[:2]
        
        # EXACT emergency.py preprocessing
        img = cv2.resize(frame, (self.input_size, self.input_size))
        blob = cv2.dnn.blobFromImage(
            img, 
            1/255.0, 
            (self.input_size, self.input_size),
            swapRB=True, 
            crop=False
        )
        
        # EXACT emergency.py forward pass
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # EXACT emergency.py output handling
        if len(outputs) == 0:
            return []
        
        preds = outputs[0]
        
        # EXACT emergency.py shape handling
        if len(preds.shape) == 3:
            preds = preds[0]
        
        if len(preds.shape) != 2:
            return []
        
        # EXACT emergency.py transpose logic
        if preds.shape[0] == 84 or (preds.shape[0] < preds.shape[1]):
            preds = preds.T
        
        # EXACT emergency.py parsing
        boxes = []
        confidences = []
        
        for pred in preds:
            if len(pred) < 84:
                continue
            
            # EXACT emergency.py coordinate extraction
            x_center, y_center, width, height = pred[0:4]
            class_scores = pred[4:84]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            
            # EXACT emergency.py filtering
            if class_id != 0:  # Only person
                continue
            
            if confidence < self.conf_threshold:
                continue
            
            # EXACT emergency.py coordinate conversion
            x1 = int((x_center - width / 2) / self.input_size * orig_w)
            y1 = int((y_center - height / 2) / self.input_size * orig_h)
            x2 = int((x_center + width / 2) / self.input_size * orig_w)
            y2 = int((y_center + height / 2) / self.input_size * orig_h)
            
            # EXACT emergency.py clipping
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            # EXACT emergency.py validation
            if x2 <= x1 or y2 <= y1:
                continue
            
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(confidence)
        
        if len(boxes) == 0:
            return []
        
        # EXACT emergency.py NMS
        try:
            indices = cv2.dnn.NMSBoxes(
                boxes, 
                confidences,
                self.conf_threshold,  # 0.001
                self.iou_threshold     # 0.70
            )
        except:
            return []
        
        # EXACT emergency.py result formatting
        results = []
        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0]
            
            for i in (indices.flatten() if hasattr(indices, 'flatten') else indices):
                x, y, w, h = boxes[i]
                results.append((x, y, x+w, y+h, confidences[i]))
        
        return results