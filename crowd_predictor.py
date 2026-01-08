"""
Crowd Predictor - Time Series Prediction for Upcoming Crowd Counts
Uses exponential smoothing and trend analysis for short-term predictions
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class CrowdPredictor:
    """
    Predicts future crowd counts using exponential smoothing and trend analysis
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize predictor
        
        Args:
            history_size: Number of historical data points to maintain
        """
        self.history_size = history_size
        self.count_history = deque(maxlen=history_size)
        self.timestamp_history = deque(maxlen=history_size)
        
        # Smoothing parameters
        self.alpha = 0.3  # Level smoothing
        self.beta = 0.1   # Trend smoothing
        
        # State variables
        self.level = 0.0
        self.trend = 0.0
        self.last_prediction_time = None
        
        print("✅ Crowd Predictor initialized")
    
    def update(self, count: int, timestamp: float = None):
        """
        Update predictor with new observation
        
        Args:
            count: Current people count
            timestamp: Time of observation (defaults to current time)
        """
        import time
        if timestamp is None:
            timestamp = time.time()
        
        self.count_history.append(count)
        self.timestamp_history.append(timestamp)
        
        # Initialize level and trend
        if len(self.count_history) == 1:
            self.level = float(count)
            self.trend = 0.0
        elif len(self.count_history) >= 2:
            # Update using double exponential smoothing
            prev_level = self.level
            self.level = self.alpha * count + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
    
    def predict(self, steps_ahead: List[int] = [5, 10, 15]) -> Dict:
        """
        Predict future crowd counts
        
        Args:
            steps_ahead: List of steps ahead to predict (in frames/seconds)
        
        Returns:
            Dictionary with predictions and confidence
        """
        if len(self.count_history) < 5:
            # Not enough data for prediction
            current = self.count_history[-1] if self.count_history else 0
            return {
                'current': current,
                'predictions': {step: current for step in steps_ahead},
                'trend': 'stable',
                'confidence': 0.0,
                'trend_strength': 0.0
            }
        
        current_count = self.count_history[-1]
        
        # Make predictions using exponential smoothing
        predictions = {}
        for step in steps_ahead:
            forecast = self.level + step * self.trend
            # Ensure non-negative
            forecast = max(0, int(round(forecast)))
            predictions[step] = forecast
        
        # Calculate trend direction
        recent_data = list(self.count_history)[-20:]
        if len(recent_data) >= 10:
            first_half = np.mean(recent_data[:len(recent_data)//2])
            second_half = np.mean(recent_data[len(recent_data)//2:])
            trend_diff = second_half - first_half
            
            if trend_diff > 5:
                trend = 'increasing'
            elif trend_diff < -5:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        # Calculate confidence based on variance
        if len(recent_data) >= 10:
            variance = np.var(recent_data)
            mean_val = np.mean(recent_data)
            cv = np.sqrt(variance) / (mean_val + 1)  # Coefficient of variation
            confidence = max(0, min(100, 100 * (1 - cv)))
        else:
            confidence = 50.0
        
        # Calculate trend strength
        trend_strength = abs(self.trend) / (self.level + 1) * 100
        trend_strength = min(trend_strength, 100.0)
        
        return {
            'current': current_count,
            'predictions': predictions,
            'trend': trend,
            'confidence': confidence,
            'trend_strength': trend_strength,
            'level': self.level,
            'trend_value': self.trend
        }
    
    def get_prediction_display(self) -> Dict:
        """
        Get formatted predictions for UI display
        """
        pred_data = self.predict(steps_ahead=[5, 10, 15])
        
        return {
            'current': pred_data['current'],
            'predicted_5s': pred_data['predictions'].get(5, pred_data['current']),
            'predicted_10s': pred_data['predictions'].get(10, pred_data['current']),
            'predicted_15s': pred_data['predictions'].get(15, pred_data['current']),
            'trend': pred_data['trend'],
            'confidence': round(pred_data['confidence'], 1),
            'trend_strength': round(pred_data['trend_strength'], 1)
        }
    
    def calculate_surge_score(self) -> float:
        """
        Calculate surge probability based on recent changes
        More accurate than the basic implementation
        
        Returns:
            Surge score between 0-100
        """
        if len(self.count_history) < 10:
            return 0.0
        
        recent = list(self.count_history)[-20:]
        
        # Calculate rate of change
        changes = []
        for i in range(1, len(recent)):
            change = recent[i] - recent[i-1]
            changes.append(change)
        
        if not changes:
            return 0.0
        
        # Metrics for surge detection
        avg_change = np.mean(changes)
        max_change = np.max(np.abs(changes))
        std_change = np.std(changes)
        
        # Surge score components
        # 1. Magnitude of average change (40% weight)
        magnitude_score = min(abs(avg_change) / 30.0, 1.0) * 40
        
        # 2. Peak changes (30% weight)
        peak_score = min(max_change / 50.0, 1.0) * 30
        
        # 3. Volatility/instability (30% weight)
        volatility_score = min(std_change / 20.0, 1.0) * 30
        
        total_surge = magnitude_score + peak_score + volatility_score
        
        return min(total_surge, 100.0)
    
    def get_detailed_metrics(self) -> Dict:
        """
        Get comprehensive prediction and trend metrics
        """
        predictions = self.predict()
        surge = self.calculate_surge_score()
        
        return {
            'predictions': predictions,
            'surge_score': surge,
            'data_points': len(self.count_history),
            'is_predicting': len(self.count_history) >= 5
        }