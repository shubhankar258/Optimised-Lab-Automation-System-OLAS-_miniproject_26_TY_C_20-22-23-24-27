# OLAS — Optimised Lab Automation System

## Overview  
OLAS is an IoT + Machine Learning based smart lab lighting system. It uses an ESP32 with a relay module to control lights via mobile app, Google Home, and manual switches. The system analyzes real-time device states and compares them with ML predictions to identify energy-saving opportunities.

---

## Features  
- Manual control using physical switches  
- Remote control via ESP RainMaker mobile app  
- Voice control using Google Home  
- ML-based prediction using Random Forest (~95.5% accuracy)  
- Real-time comparison of actual vs predicted states  
- Suggests optimal ON/OFF actions for energy efficiency  
- Works without modifying existing wiring  

---

## System Architecture  
- **Hardware:** ESP32 + 4-Channel Relay + Lab Lights  
- **Cloud:** ESP RainMaker (MQTT + REST API)  
- **ML Layer:** Flask application on Render  
- **Control Paths:** Switch | Mobile App | Voice  
- **ML Role:** Prediction + Comparison (No direct automation)  

---

## Tech Stack  
- ESP32 (Arduino / C++)  
- Python (Flask)  
- scikit-learn (Random Forest)  
- ESP RainMaker  
- Render (Deployment)  

---

## How It Works  
1. ESP32 controls relays connected to lab lights  
2. Users operate lights via switch, mobile app, or voice  
3. Flask server fetches real-time device states from RainMaker  
4. ML model predicts optimal ON/OFF states using time-based features  
5. System compares **actual vs predicted states**  
6. If mismatch → suggests better state (energy-saving recommendation)  

---

## Machine Learning Details  
- Dataset: ~3360 samples (synthetic timetable-based data)  
- Features: hour, minute, day of week, weekend flag, cyclical encoding  
- Model: Random Forest Classifier (4 independent models)  
- Accuracy: ~95.5%  
- Output: Match / Mismatch + Suggested action  

---

## Screenshots  

- Hardware Setup
- <img width="487" height="523" alt="Screenshot 2026-04-15 173311" src="https://github.com/user-attachments/assets/ee268813-fff0-4c3f-9da9-99c6145d0670" />

- ML Comparison Dashboard (Actual vs Predicted)
- <img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/8588ae95-003e-4a7b-9cf2-2df94caf826c" />
 
- RainMaker App
- <img width="575" height="1280" alt="WhatsApp Image 2026-04-15 at 12 13 58 PM" src="https://github.com/user-attachments/assets/af7c6ea4-c236-499f-8056-cdee819251be" />
 
- Google Home Integration
- <img width="575" height="1280" alt="WhatsApp Image 2026-04-15 at 12 13 55 PM" src="https://github.com/user-attachments/assets/e96bed71-9fa8-46e6-a4d7-7a7ecb8c84a6" />


---



## ⭐ Star this repo if you like it!
