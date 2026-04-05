# AURA — Adaptive Usage-based Relay Automation (2026)

## Files needed in this folder
- app.py           ← Flask scheduler (this file)
- lab_model.pkl    ← trained model (from Step 2)
- requirements.txt ← dependencies
- render.yaml      ← Render.com config

## Deploy on Render.com (free)

1. Create account at https://render.com
2. New → Web Service → connect your GitHub repo
3. Set environment variables:
   - RAINMAKER_NODE_ID  = your node id
   - RAINMAKER_TOKEN    = your access token
4. Build command  : pip install -r requirements.txt
5. Start command  : gunicorn app:app --workers 1 --threads 2 --bind 0.0.0.0:$PORT
6. Click Deploy

## Endpoints
- GET  /            → live dashboard (auto-refresh 60s)
- GET  /status      → JSON health check
- POST /trigger     → force prediction now (demo use)
- GET  /predict_time/9/30/0  → test any time (hour/minute/dow)
