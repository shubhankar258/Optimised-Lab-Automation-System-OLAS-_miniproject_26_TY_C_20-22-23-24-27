# 🚀 OLAS — Optimised Lab Automation System

## 📌 Overview
OLAS is an IoT + Machine Learning based smart lab lighting system. It uses an ESP32 with a relay module to control lights via mobile app, Google Home, and automatic ML-based scheduling — without requiring additional hardware.

---

## ✨ Features
- Manual control using physical switches  
- Remote control via ESP RainMaker mobile app  
- Voice control using Google Home  
- ML-based automatic lighting (Random Forest ~95% accuracy)  
- Works without modifying existing wiring  

---

## 🏗️ System Architecture
- **Hardware:** ESP32 + 4-Channel Relay + Lab Lights  
- **Cloud:** ESP RainMaker (MQTT + REST API)  
- **ML Layer:** Flask app on Render  
- **Control Paths:** Switch | App | Voice | ML  

---

## 🔧 Tech Stack
- ESP32 (Arduino / C++)  
- Python (Flask)  
- scikit-learn (Random Forest)  
- ESP RainMaker  
- Render (Deployment)  

---

## ⚙️ How It Works
1. ESP32 controls relays connected to lab lights  
2. Users control via switch, app, or voice  
3. Flask server runs ML model every 30 minutes  
4. If prediction differs from actual state → system auto-corrects  

---

## 📊 ML Details
- Dataset: ~3360 rows (time-based)  
- Features: hour, minute, day, weekend, cyclical encoding  
- Accuracy: ~95.5%  

---

## 📸 Screenshots
_Add your images in `/images` folder and link here_

- Hardware Setup  
- Dashboard  
- RainMaker App  
- Google Home  

---
