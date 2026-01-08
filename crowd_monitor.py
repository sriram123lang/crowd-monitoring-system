"""
Crowd Monitor - EXACT Emergency.py Configuration
Matches proven working settings
"""

import numpy as np
from collections import deque
import time
from typing import Dict, List, Tuple
from yolo_detector import YOLODetector
from crowd_predictor import CrowdPredictor


class CrowdMonitor:
    """
    Crowd monitor with EXACT emergency.py configuration
    """
    
    def __init__(
        self,
        yolo_model_path: str,
        frame_area: float = 921600.0,
        conf_threshold: float = 0.001,  # EXACT emergency.py
        rolling_window: int = 30,
        tracker_max_disappeared: int = 20,
        tracker_max_distance: float = 100.0,
        video_quality: str = 'medium',
        camera_angle: str = 'birds_eye'
    ):
        """
        Initialize with EXACT emergency.py settings
        """
        
        print("🚀 EXACT Emergency Config Crowd Monitor")
        
        # Initialize detector with EXACT emergency.py settings
        self.detector = YOLODetector(yolo_model_path, conf_threshold=0.001)
        self.detector.set_angle_mode(camera_angle)
        self.detector.set_quality_mode(video_quality)
        
        # Initialize predictor
        self.predictor = CrowdPredictor(history_size=100)
        
        # Configuration
        self.frame_area = frame_area
        self.rolling_window = rolling_window
        
        # Physical area
        self.physical_area_m2 = 100.0
        
        # Auto-detect based on frame size
        if frame_area > 1920 * 1080:
            self.physical_area_m2 = 150.0
        elif frame_area > 1280 * 720:
            self.physical_area_m2 = 100.0
        else:
            self.physical_area_m2 = 75.0
        
        print(f"   Frame area: {frame_area:.0f} pixels")
        print(f"   Physical area: {self.physical_area_m2:.0f} m²")
        print(f"   Confidence: 0.001 (EXACT emergency.py)")
        print(f"   IOU: 0.70 (EXACT emergency.py)")
        
        # Statistics
        self.people_count_history = deque(maxlen=rolling_window)
        self.density_history = deque(maxlen=rolling_window)
        
        # Current state
        self.current_people_count = 0
        self.current_density = 0.0
        self.current_density_percent = 0.0
        self.current_risk_level = 'LOW'
        self.current_surge_score = 0.0
        self.confidence = 1.0
        
        # Risk thresholds (people per m²)
        self.DENSITY_LOW = 1.0
        self.DENSITY_MEDIUM = 2.5
        self.DENSITY_HIGH = 4.0
        self.DENSITY_CRITICAL = 5.5
        
        print("✅ Monitor initialized with EXACT emergency.py config")
    
    def set_physical_area(self, area_m2: float):
        """Set the actual physical area covered by the camera"""
        self.physical_area_m2 = area_m2
        print(f"📏 Physical area set to: {area_m2} m²")
    
    def update(self, frame: np.ndarray) -> Dict:
        """
        Process frame with EXACT emergency.py detection
        """
        # Detect people using EXACT emergency.py method
        detections = self.detector.detect(frame)
        
        # Count people
        self.current_people_count = len(detections)
        
        # Update predictor
        current_time = time.time()
        self.predictor.update(self.current_people_count, current_time)
        
        # Calculate density
        self.current_density = self.current_people_count / self.physical_area_m2
        
        # Calculate density percentage
        max_density = 10.0
        self.current_density_percent = min((self.current_density / max_density) * 100, 100.0)
        
        # Update history
        self.people_count_history.append(self.current_people_count)
        self.density_history.append(self.current_density)
        
        # Calculate surge score
        self.current_surge_score = self.predictor.calculate_surge_score()
        
        # Determine risk level
        self.current_risk_level = self._calculate_risk_level_by_density()
        
        # High confidence
        self.confidence = 0.95
        
        return self.get_current_stats()
    
    def get_current_stats(self) -> Dict:
        """Get current statistics with predictions"""
        predictions = self.predictor.get_prediction_display()
        
        return {
            'people_count': self.current_people_count,
            'density': self.current_density,
            'density_percent': self.current_density_percent,
            'physical_area': self.physical_area_m2,
            'risk_level': self.current_risk_level,
            'surge_score': self.current_surge_score,
            'confidence': self.confidence,
            'rolling_avg': self._get_rolling_average(),
            'rolling_max': self._get_rolling_max(),
            'rolling_min': self._get_rolling_min(),
            'avg_density': self._get_average_density(),
            'predictions': predictions
        }
    
    def _calculate_risk_level_by_density(self) -> str:
        """Calculate risk level based on crowd density"""
        density = self.current_density
        
        if density < self.DENSITY_LOW:
            return 'LOW'
        elif density < self.DENSITY_MEDIUM:
            return 'MEDIUM'
        elif density < self.DENSITY_HIGH:
            return 'HIGH'
        elif density < self.DENSITY_CRITICAL:
            return 'CRITICAL'
        else:
            return 'EMERGENCY'
    
    def get_risk_description(self) -> str:
        """Get human-readable risk description"""
        risk_descriptions = {
            'LOW': 'Safe - Normal crowd levels',
            'MEDIUM': 'Crowded - Monitor situation',
            'HIGH': 'Very Crowded - Take action',
            'CRITICAL': 'Dangerous - Immediate action required',
            'EMERGENCY': 'EMERGENCY - Evacuate immediately'
        }
        return risk_descriptions.get(self.current_risk_level, 'Unknown')
    
    def _get_rolling_average(self) -> float:
        """Get rolling average of people count"""
        if not self.people_count_history:
            return 0.0
        return float(np.mean(list(self.people_count_history)))
    
    def _get_rolling_max(self) -> int:
        """Get rolling maximum of people count"""
        if not self.people_count_history:
            return 0
        return int(np.max(list(self.people_count_history)))
    
    def _get_rolling_min(self) -> int:
        """Get rolling minimum of people count"""
        if not self.people_count_history:
            return 0
        return int(np.min(list(self.people_count_history)))
    
    def _get_average_density(self) -> float:
        """Get average density over time"""
        if not self.density_history:
            return 0.0
        return float(np.mean(list(self.density_history)))
    
    def reset_statistics(self):
        """Reset statistics"""
        self.people_count_history.clear()
        self.density_history.clear()
        self.current_people_count = 0
        self.current_density = 0.0
        self.current_density_percent = 0.0
        self.current_risk_level = 'LOW'
        self.current_surge_score = 0.0
        print("📊 Statistics reset")
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_frames_processed': len(self.people_count_history),
            'current_count': self.current_people_count,
            'average_count': self._get_rolling_average(),
            'max_count': self._get_rolling_max(),
            'min_count': self._get_rolling_min(),
            'current_density': self.current_density,
            'current_density_percent': self.current_density_percent,
            'average_density': self._get_average_density(),
            'physical_area_m2': self.physical_area_m2,
            'current_risk': self.current_risk_level,
            'risk_description': self.get_risk_description(),
            'current_surge': self.current_surge_score,
            'confidence': self.confidence
        }
    
    def get_detailed_metrics(self) -> Dict:
        """Get detailed crowd metrics for display"""
        density = self.current_density
        count = self.current_people_count
        
        return {
            'people_count': count,
            'density_per_m2': round(density, 2),
            'density_percent': round(self.current_density_percent, 1),
            'physical_area': self.physical_area_m2,
            'risk_level': self.current_risk_level,
            'risk_description': self.get_risk_description(),
            'is_safe': density < self.DENSITY_MEDIUM,
            'requires_action': density >= self.DENSITY_HIGH,
            'is_emergency': density >= self.DENSITY_CRITICAL,
            'people_per_sqm': f"{density:.2f} people/m²",
            'crowding_percentage': f"{self.current_density_percent:.1f}%",
            'status_color': self._get_status_color()
        }
    
    def _get_status_color(self) -> str:
        """Get color code for current status"""
        colors = {
            'LOW': 'green',
            'MEDIUM': 'yellow',
            'HIGH': 'orange',
            'CRITICAL': 'red',
            'EMERGENCY': 'darkred'
        }
        return colors.get(self.current_risk_level, 'gray')