# ============================================================
#  AURA — Adaptive Usage-based Relay Automation
#  Flask Scheduler  |  Deploy on Render.com (free tier)
#  Year: 2026
# ============================================================

from flask import Flask, jsonify, render_template_string
import pickle
import pandas as pd
import numpy as np
import requests
import datetime
import schedule
import threading
import time
import os
import logging

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("AURA")

app = Flask(__name__)

# ── Config — set these as Environment Variables on Render ────
NODE_ID = os.environ.get("RAINMAKER_NODE_ID", "YOUR_NODE_ID_HERE")
TOKEN   = os.environ.get("RAINMAKER_TOKEN",   "YOUR_TOKEN_HERE")

SWITCHES  = ["Switch1", "Switch2", "Switch3", "Switch4"]
FEATURES  = [
    "hour", "minute", "day_of_week", "is_weekend", "time_block",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "dow_sin", "dow_cos"
]
API_URL   = "https://api.rainmaker.espressif.com/v1/user/nodes/params"
HEADERS   = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type" : "application/json"
}

# ── Load model ───────────────────────────────────────────────
log.info("Loading lab_model.pkl ...")
with open("lab_model.pkl", "rb") as f:
    models = pickle.load(f)
log.info(f"Model loaded — classifiers: {list(models.keys())}")

# ── In-memory log of last 50 predictions ────────────────────
prediction_log = []

# ── Feature builder ─────────────────────────────────────────
def build_features(dt: datetime.datetime) -> pd.DataFrame:
    row = {
        "hour"       : dt.hour,
        "minute"     : dt.minute,
        "day_of_week": dt.weekday(),
        "is_weekend" : 1 if dt.weekday() == 6 else 0,
        "time_block" : dt.hour // 6,
        "minute_sin" : np.sin(2 * np.pi * dt.minute / 60),
        "minute_cos" : np.cos(2 * np.pi * dt.minute / 60),
        "hour_sin"   : np.sin(2 * np.pi * dt.hour   / 24),
        "hour_cos"   : np.cos(2 * np.pi * dt.hour   / 24),
        "dow_sin"    : np.sin(2 * np.pi * dt.weekday() / 7),
        "dow_cos"    : np.cos(2 * np.pi * dt.weekday() / 7),
    }
    return pd.DataFrame([row])[FEATURES]

# ── Which session is active right now ───────────────────────
def current_session(dt: datetime.datetime) -> str:
    t = dt.time()
    if datetime.time(9, 15) <= t < datetime.time(11, 15):
        return "Session 1  (9:15 – 11:15)"
    if datetime.time(11, 30) <= t < datetime.time(13, 30):
        return "Session 2  (11:30 – 13:30)"
    if datetime.time(14, 15) <= t < datetime.time(16, 15):
        return "Session 3  (14:15 – 16:15)"
    if dt.weekday() == 6:
        return "Sunday — no college"
    return "Outside session hours"

# ── Core: predict → send to RainMaker → write_callback fires ─
def predict_and_control(source: str = "scheduler"):
    now     = datetime.datetime.now()
    feats   = build_features(now)
    session = current_session(now)

    predictions = {}
    for sw in SWITCHES:
        pred = int(models[sw].predict(feats)[0])
        # Get probability for confidence score
        prob = models[sw].predict_proba(feats)[0][pred]
        predictions[sw] = {"state": bool(pred), "confidence": round(float(prob), 3)}

    # Build RainMaker payload — "Power" matches write_callback
    payload = {
        "node_id": NODE_ID,
        "payload": {
            sw: {"Power": predictions[sw]["state"]}
            for sw in SWITCHES
        }
    }

    status_code = None
    api_status  = "not_sent"

    # Only call API if token is configured
    if TOKEN != "YOUR_TOKEN_HERE":
        try:
            r = requests.put(API_URL, headers=HEADERS, json=payload, timeout=10)
            status_code = r.status_code
            api_status  = "ok" if r.status_code == 200 else f"error_{r.status_code}"
            if r.status_code == 200:
                log.info(f"Commands sent OK  |  {session}")
            else:
                log.warning(f"API returned {r.status_code}: {r.text[:120]}")
        except Exception as e:
            api_status = "connection_error"
            log.error(f"API call failed: {e}")
    else:
        api_status = "token_not_configured"
        log.warning("TOKEN not set — prediction run but not sent to ESP32")

    # Store in log
    entry = {
        "timestamp"  : now.strftime("%Y-%m-%d %H:%M:%S"),
        "session"    : session,
        "source"     : source,
        "predictions": predictions,
        "api_status" : api_status,
    }
    prediction_log.insert(0, entry)
    if len(prediction_log) > 50:
        prediction_log.pop()

    states_str = "  ".join([
        f"{sw}: {'ON ' if predictions[sw]['state'] else 'OFF'}"
        for sw in SWITCHES
    ])
    log.info(f"[{source}]  {states_str}  |  API: {api_status}")
    return entry

# ── Scheduler thread (runs every 30 min) ────────────────────
schedule.every(30).minutes.do(lambda: predict_and_control("scheduler"))

def run_scheduler():
    log.info("Scheduler started — firing every 30 minutes")
    # Run once immediately on startup
    predict_and_control("startup")
    while True:
        schedule.run_pending()
        time.sleep(30)

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# ── Dashboard HTML ───────────────────────────────────────────
DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
  <title>AURA Dashboard — 2026</title>
  <meta http-equiv="refresh" content="60">
  <style>
    body { font-family: monospace; background: #0f1117; color: #e0e0e0;
           max-width: 860px; margin: 40px auto; padding: 0 20px; }
    h1   { color: #7f77dd; letter-spacing: 2px; }
    h2   { color: #9FE1CB; font-size: 14px; margin-top: 32px; }
    .card { background: #1a1d27; border: 1px solid #2a2d3a;
            border-radius: 8px; padding: 16px; margin: 12px 0; }
    .on  { color: #5DCAA5; font-weight: bold; }
    .off { color: #888780; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    td, th { padding: 8px 12px; text-align: left;
             border-bottom: 1px solid #2a2d3a; }
    th { color: #7f77dd; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
             font-size: 11px; font-weight: bold; }
    .ok   { background: #085041; color: #9FE1CB; }
    .err  { background: #501313; color: #F09595; }
    .warn { background: #412402; color: #FAC775; }
  </style>
</head>
<body>
  <h1>AURA</h1>
  <p style="color:#5F5E5A">Adaptive Usage-based Relay Automation &nbsp;|&nbsp; 2026</p>

  <div class="card">
    <h2>LAST PREDICTION</h2>
    {% if logs %}
    {% set last = logs[0] %}
    <p>Time &nbsp;&nbsp;&nbsp;: {{ last.timestamp }}</p>
    <p>Session : {{ last.session }}</p>
    <p>Source  : {{ last.source }}</p>
    <p>
      {% for sw, info in last.predictions.items() %}
        <span class="{{ 'on' if info.state else 'off' }}">
          {{ sw }}: {{ 'ON ' if info.state else 'OFF' }}
          ({{ (info.confidence * 100)|int }}%)
        </span> &nbsp;&nbsp;
      {% endfor %}
    </p>
    <p>API &nbsp;&nbsp;&nbsp;:
      <span class="badge {{ 'ok' if last.api_status == 'ok' else 'warn' if 'not' in last.api_status else 'err' }}">
        {{ last.api_status }}
      </span>
    </p>
    {% endif %}
  </div>

  <h2>PREDICTION LOG (last 50)</h2>
  <table>
    <tr><th>Timestamp</th><th>Session</th>
        <th>S1</th><th>S2</th><th>S3</th><th>S4</th><th>API</th></tr>
    {% for entry in logs %}
    <tr>
      <td>{{ entry.timestamp }}</td>
      <td style="color:#888">{{ entry.session[:16] }}</td>
      {% for sw in ['Switch1','Switch2','Switch3','Switch4'] %}
      <td class="{{ 'on' if entry.predictions[sw].state else 'off' }}">
        {{ 'ON' if entry.predictions[sw].state else 'OFF' }}
      </td>
      {% endfor %}
      <td><span class="badge {{ 'ok' if entry.api_status=='ok' else 'warn' }}">
        {{ entry.api_status }}</span></td>
    </tr>
    {% endfor %}
  </table>
  <p style="color:#444; font-size:11px; margin-top:20px">
    Auto-refreshes every 60 s &nbsp;|&nbsp;
    POST /trigger to force a prediction now
  </p>
</body>
</html>
"""

# ── Flask routes ─────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD, logs=prediction_log)

@app.route("/status")
def status():
    """JSON status — useful for health checks."""
    last = prediction_log[0] if prediction_log else None
    return jsonify({
        "project"    : "AURA 2026",
        "status"     : "running",
        "scheduler"  : "active — every 30 min",
        "node_id"    : NODE_ID[:8] + "..." if len(NODE_ID) > 8 else NODE_ID,
        "last_run"   : last["timestamp"] if last else None,
        "last_session": last["session"]  if last else None,
    })

@app.route("/trigger", methods=["GET", "POST"])
def trigger():
    """Manually trigger a prediction — useful for demo."""
    entry = predict_and_control("manual_trigger")
    return jsonify({
        "message"    : "Prediction triggered",
        "timestamp"  : entry["timestamp"],
        "session"    : entry["session"],
        "predictions": entry["predictions"],
        "api_status" : entry["api_status"],
    })

@app.route("/predict_time/<int:hour>/<int:minute>/<int:dow>")
def predict_time(hour, minute, dow):
    """
    Test predictions for any time.
    /predict_time/9/30/0  → Monday 9:30
    /predict_time/14/15/2 → Wednesday 14:15
    dow: 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri 5=Sat 6=Sun
    """
    dt    = datetime.datetime.now().replace(hour=hour, minute=minute)
    dt    = dt.replace(day=dt.day)
    feats = build_features(dt)
    preds = {}
    for sw in SWITCHES:
        state = int(models[sw].predict(feats)[0])
        prob  = models[sw].predict_proba(feats)[0][state]
        preds[sw] = {"state": bool(state), "confidence": round(float(prob), 3)}
    return jsonify({
        "query"      : f"{hour:02d}:{minute:02d}  day={dow}",
        "predictions": preds
    })

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
