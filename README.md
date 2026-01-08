# 🎮 AI-Powered Crowd Monitoring System

Real-time crowd detection, counting, and risk assessment using AI computer vision.

## ✨ Features

- 🎯 **Real-time People Detection** - YOLOv8 AI model
- 📊 **Crowd Density Analysis** - People per square meter calculation
- 🔮 **AI Predictions** - Forecasts crowd size 5-15 seconds ahead
- ⚠️ **Risk Assessment** - 5-level risk system (LOW → EMERGENCY)
- 🌊 **Surge Detection** - Identifies dangerous crowd movements
- 📈 **Live Dashboard** - Beautiful cyber-themed interface
- 📚 **Complete Documentation** - Help center, FAQ, user manual

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/crowd-monitoring-system.git
cd crowd-monitoring-system

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model
# Place yolov8m.onnx in models/ folder
```

### Configuration
```bash
# Set physical area (in square meters)
set PHYSICAL_AREA_M2=100

# Windows
set VIDEO_PATH=video.mp4

# Linux/Mac
export PHYSICAL_AREA_M2=100
export VIDEO_PATH=video.mp4
```

### Run
```bash
python app.py
```

Open browser: **http://localhost:5000**

## 📊 Screenshots

[Add screenshots of your dashboard here]

## 🎯 Use Cases

- Event management (concerts, festivals)
- Public safety monitoring
- Stadium/venue management
- Train stations, airports
- Shopping malls

## 🏆 Hackathon Project

Built for [Your Hackathon Name] - [Date]

## 📄 License

MIT License

## 👥 Team

- [Your Name] - [Your Role]

## 🙏 Acknowledgments

- YOLOv8 for object detection
- Flask for web framework
- Claude AI for development assistance