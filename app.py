# ============================================================
#  OLAS — Optimised Lab Automation System
#  Flask Scheduler + Live State Fetch | Deploy on Render.com
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
import pytz

# Set timezone to IST
IST = pytz.timezone('Asia/Kolkata')

# ── Logging setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("OLAS")

app = Flask(__name__)

# ── Config — set these as Environment Variables on Render ────
NODE_ID  = os.environ.get("RAINMAKER_NODE_ID",  "aHjSGbCmWDjvmETWDMrupL")
RM_EMAIL = os.environ.get("RAINMAKER_EMAIL",    "sdhumal197@gmail.com")
RM_PASS  = os.environ.get("RAINMAKER_PASSWORD", "Pass@123")

SWITCHES  = ["Switch1", "Switch2", "Switch3", "Switch4"]
FEATURES  = [
    "hour", "minute", "day_of_week", "is_weekend", "time_block",
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "dow_sin",  "dow_cos"
]
API_URL   = "https://api.rainmaker.espressif.com/v1/user/nodes/params"
LOGIN_URL = "https://api.rainmaker.espressif.com/v1/login2"

# ── Token cache ───────────────────────────────────────────────
_token_cache = {
    "access_token" : None,
    "refresh_token": None,
    "fetched_at"   : 0
}

# Global variable to store actual states fetched from RainMaker
actual_states = {sw: False for sw in SWITCHES}

def login_and_get_tokens():
    """
    Full login using email + password via /v1/login2.
    Stores both access_token and refresh_token in cache.
    """
    resp = requests.post(
        LOGIN_URL,
        json={"user_name": RM_EMAIL, "password": RM_PASS},
        timeout=10
    )
    if resp.status_code == 200:
        data = resp.json()
        _token_cache["access_token"]  = data["accesstoken"]
        _token_cache["refresh_token"] = data["refreshtoken"]
        _token_cache["fetched_at"]    = time.time()
        log.info("RainMaker login successful — tokens obtained")
    else:
        raise RuntimeError(
            f"RainMaker login failed: {resp.status_code}  {resp.text[:200]}"
        )

def get_access_token() -> str:
    """
    Returns a valid access token.
    """
    now = time.time()
    age = now - _token_cache["fetched_at"]

    if _token_cache["access_token"] is None or age > 3000:
        if _token_cache["refresh_token"]:
            try:
                resp = requests.post(
                    "https://cognito-idp.us-east-1.amazonaws.com/",
                    headers={
                        "Content-Type": "application/x-amz-json-1.1",
                        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth"
                    },
                    json={
                        "AuthFlow"      : "REFRESH_TOKEN_AUTH",
                        "ClientId"      : "1p3enpe49h9v0lqd7i4s5bub",
                        "AuthParameters": {
                            "REFRESH_TOKEN": _token_cache["refresh_token"]
                        }
                    },
                    timeout=10
                )
                new_token = resp.json()["AuthenticationResult"]["AccessToken"]
                _token_cache["access_token"] = new_token
                _token_cache["fetched_at"]   = now
                log.info("Access token refreshed silently via refresh_token")
            except Exception as e:
                log.warning(f"Silent refresh failed ({e}) — doing full re-login")
                login_and_get_tokens()
        else:
            login_and_get_tokens()

    return _token_cache["access_token"]

# ── Fetch actual states from RainMaker ───────────────────────
def fetch_actual_states():
    """Fetch current switch states from RainMaker cloud"""
    global actual_states
    try:
        headers = {"Authorization": f"Bearer {get_access_token()}"}
        resp = requests.get(
            f"{API_URL}?node_id={NODE_ID}",
            headers=headers,
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            for sw in SWITCHES:
                if sw in data:
                    state = data[sw].get("Power", data[sw].get("output", False))
                    actual_states[sw] = bool(state)
            log.info(f"Actual states fetched: {actual_states}")
            return True
        else:
            log.warning(f"Failed to fetch states: {resp.status_code}")
            return False
    except Exception as e:
        log.error(f"Error fetching states: {e}")
        return False

# ── Login on startup ──────────────────────────────────────────
if RM_EMAIL != "your@email.com":
    try:
        login_and_get_tokens()
        fetch_actual_states()
    except Exception as e:
        log.error(f"Startup login failed: {e}")

# ── Load model ───────────────────────────────────────────────
log.info("Loading lab_model.pkl ...")
with open("lab_model.pkl", "rb") as f:
    models = pickle.load(f)
log.info(f"Model loaded — classifiers: {list(models.keys())}")

# ── In-memory log of last 50 predictions ─────────────────────
prediction_log = []

# ── Feature builder ──────────────────────────────────────────
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

# ── Session detector ─────────────────────────────────────────
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

# ── Core: predict and compare (NO automation commands) ───────
def predict_and_compare(source: str = "scheduler"):
    """Run ML prediction and compare with actual states (no commands sent)"""
    now = datetime.datetime.now(IST)
    feats = build_features(now)
    session = current_session(now)

    # Fetch latest actual states
    fetch_success = fetch_actual_states()

    # Run all 4 classifiers
    predictions = {}
    for sw in SWITCHES:
        pred = int(models[sw].predict(feats)[0])
        prob = models[sw].predict_proba(feats)[0][pred]
        predictions[sw] = {
            "state": bool(pred),
            "confidence": round(float(prob), 3)
        }

    # Compare actual vs predicted
    comparison = {}
    for sw in SWITCHES:
        actual = actual_states.get(sw, False)
        predicted = predictions[sw]["state"]
        comparison[sw] = {
            "actual": actual,
            "predicted": predicted,
            "match": actual == predicted,
            "confidence": predictions[sw]["confidence"]
        }

    api_status = "states_fetched" if fetch_success else "fetch_failed"

    # Store entry in memory log
    entry = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "session": session,
        "source": source,
        "predictions": predictions,
        "actual_states": actual_states.copy(),
        "comparison": comparison,
        "api_status": api_status,
    }
    prediction_log.insert(0, entry)
    if len(prediction_log) > 50:
        prediction_log.pop()

    # Log summary
    matches = sum(1 for sw in SWITCHES if comparison[sw]["match"])
    log.info(f"[{source}] Match: {matches}/4 | Fetch: {api_status}")
    return entry

# ── Scheduler thread (only fetches and compares, no commands) ─
schedule.every(30).minutes.do(lambda: predict_and_compare("scheduler"))

def run_scheduler():
    log.info("Scheduler started — fetching states & comparing every 30 minutes")
    predict_and_compare("startup")
    while True:
        schedule.run_pending()
        time.sleep(30)

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# ── Dashboard HTML (Updated with Actual vs Predicted) ────────
DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OLAS · Smart Lab</title>
  <meta http-equiv="refresh" content="60">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }

    :root {
      --bg-page: #f5f7fa;
      --card-bg: #ffffff;
      --text-primary: #1a1f2e;
      --text-secondary: #5e6a7e;
      --text-muted: #8a94a6;
      --border-light: #e9ecf0;
      --accent: #0066ff;
      --accent-soft: #e0ebff;
      --success: #00b87a;
      --warning: #f59e0b;
      --danger: #ef4444;
      --shadow-md: 0 8px 20px rgba(0,0,0,0.06);
    }

    [data-theme="dark"] {
      --bg-page: #0b0e14;
      --card-bg: #141a24;
      --text-primary: #eef2f6;
      --text-secondary: #9aabbf;
      --text-muted: #6a7b8c;
      --border-light: #242c38;
      --accent: #3399ff;
      --accent-soft: #1a2a3a;
      --success: #00cc88;
    }

    body {
      background: var(--bg-page);
      color: var(--text-primary);
      padding: 20px 24px;
      transition: background 0.2s;
    }

    .dashboard { max-width: 1200px; margin: 0 auto; }

    /* Top Nav */
    .top-nav {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 28px;
      flex-wrap: wrap;
      gap: 16px;
    }

    .project-name {
      font-size: 2rem;
      font-weight: 700;
      background: linear-gradient(135deg, #0066ff, #00c2ff);
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
    }

    .nav-controls {
      display: flex;
      align-items: center;
      gap: 16px;
    }

    .theme-toggle {
      display: flex;
      background: var(--card-bg);
      border: 1px solid var(--border-light);
      border-radius: 40px;
      padding: 4px;
    }

    .theme-btn {
      background: transparent;
      border: none;
      padding: 6px 16px;
      border-radius: 40px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 500;
      color: var(--text-secondary);
    }

    .theme-btn.active {
      background: var(--accent);
      color: white;
    }

    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--accent-soft);
      padding: 8px 18px;
      border-radius: 40px;
    }

    .pulse-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--success);
      animation: pulse 1.8s infinite;
    }

    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

    /* Comparison Grid */
    .section-title {
      font-size: 1.2rem;
      font-weight: 600;
      margin: 24px 0 16px;
    }

    .comparison-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 20px;
    }

    .comparison-card {
      background: var(--card-bg);
      border-radius: 24px;
      padding: 20px 16px;
      box-shadow: var(--shadow-md);
      border: 2px solid var(--border-light);
      text-align: center;
    }

    .comparison-card.match {
      border-color: var(--success);
      background: linear-gradient(145deg, var(--card-bg), var(--accent-soft));
    }

    .comparison-card.mismatch {
      border-color: var(--warning);
    }

    .device-name {
      font-weight: 600;
      font-size: 1.1rem;
      margin-bottom: 16px;
    }

    .state-row {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid var(--border-light);
    }

    .state-row .label {
      color: var(--text-muted);
      font-size: 0.85rem;
    }

    .state-row .value {
      font-weight: 600;
    }

    .value.on { color: var(--success); }
    .value.off { color: var(--text-muted); }

    .match-status {
      margin-top: 12px;
      padding: 6px;
      border-radius: 20px;
      font-weight: 600;
      font-size: 0.9rem;
    }

    .match .match-status {
      background: var(--success);
      color: white;
    }

    .mismatch .match-status {
      background: var(--warning);
      color: white;
    }

    .action-needed {
      margin-top: 8px;
      font-size: 0.8rem;
      color: var(--text-muted);
      font-style: italic;
    }

    .confidence {
      margin-top: 8px;
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    /* Cards */
    .automation-card {
      background: var(--card-bg);
      border-radius: 24px;
      padding: 24px;
      margin: 28px 0;
      border: 1px solid var(--border-light);
      box-shadow: var(--shadow-md);
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 20px;
    }

    .btn {
      background: var(--accent);
      color: white;
      border: none;
      padding: 12px 24px;
      border-radius: 40px;
      font-weight: 600;
      cursor: pointer;
    }

    .btn-outline {
      background: transparent;
      border: 1px solid var(--border-light);
      color: var(--text-primary);
    }

    /* Manual Section */
    .manual-section {
      background: var(--card-bg);
      border-radius: 24px;
      padding: 24px;
      margin-bottom: 28px;
      border: 1px solid var(--border-light);
    }

    .manual-form {
      display: flex;
      gap: 16px;
      align-items: flex-end;
      flex-wrap: wrap;
      margin-top: 16px;
    }

    .input-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .input-group label {
      font-size: 0.8rem;
      color: var(--text-muted);
      font-weight: 500;
    }

    .input-group input, .input-group select {
      background: var(--bg-page);
      border: 1px solid var(--border-light);
      border-radius: 16px;
      padding: 12px 16px;
      color: var(--text-primary);
      font-size: 1rem;
      width: 120px;
    }

    .prediction-preview {
      margin-top: 24px;
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
    }

    .preview-item {
      background: var(--bg-page);
      padding: 16px 24px;
      border-radius: 20px;
    }

    /* Log Feed */
    .log-feed {
      background: var(--card-bg);
      border-radius: 24px;
      padding: 20px;
      border: 1px solid var(--border-light);
    }

    .log-item {
      display: flex;
      align-items: center;
      padding: 12px 0;
      border-bottom: 1px solid var(--border-light);
    }

    .log-time { min-width: 140px; color: var(--text-muted); font-size: 0.85rem; }

    @media (max-width: 800px) {
      .comparison-grid { grid-template-columns: repeat(2, 1fr); }
    }

    @media (max-width: 500px) {
      .comparison-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<div class="dashboard">
  <!-- Top Nav -->
  <div class="top-nav">
    <div class="project-name">OLAS</div>
    <div class="nav-controls">
      <div class="theme-toggle">
        <button class="theme-btn active" data-theme="light" onclick="setTheme('light')">☀️ Light</button>
        <button class="theme-btn" data-theme="dark" onclick="setTheme('dark')">🌙 Dark</button>
      </div>
      <div class="status-badge">
        <span class="pulse-dot"></span>
        <span class="status-text">Live · RainMaker</span>
      </div>
    </div>
  </div>

  <!-- COMPARISON GRID: Actual vs Predicted -->
  <div class="section-title">📊 Actual vs Predicted Comparison</div>
  <div class="comparison-grid" id="comparisonGrid">
    <!-- Populated by JS -->
  </div>

  <!-- Action Card -->
  <div class="automation-card">
    <div>
      <h3>🤖 ML Intelligence</h3>
      <p style="color: var(--text-muted);">Predictions compared with live states · Automation documented as future scope</p>
    </div>
    <button class="btn" id="refreshBtn">⟳ Fetch & Predict</button>
  </div>

  <!-- Manual Model Check -->
  <div class="manual-section">
    <h3>🧪 Manual Model Check</h3>
    <div class="manual-form">
      <div class="input-group">
        <label>Hour</label>
        <input type="number" id="predHour" min="0" max="23" value="9">
      </div>
      <div class="input-group">
        <label>Minute</label>
        <input type="number" id="predMinute" min="0" max="59" value="30">
      </div>
      <div class="input-group">
        <label>Day</label>
        <select id="predDay">
          <option value="0">Mon</option><option value="1">Tue</option><option value="2">Wed</option>
          <option value="3">Thu</option><option value="4">Fri</option><option value="5">Sat</option><option value="6">Sun</option>
        </select>
      </div>
      <button class="btn" id="predictBtn">Predict</button>
    </div>
    <div id="manualPredictionOutput" class="prediction-preview"></div>
  </div>

  <!-- Activity Log -->
  <div class="log-feed">
    <h3 style="margin-bottom: 16px;">📋 Recent Activity</h3>
    <div id="logFeedContainer"></div>
  </div>
</div>

<script>
  function setTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-theme="${t}"]`).classList.add('active');
    localStorage.setItem('olas-theme', t);
  }
  setTheme(localStorage.getItem('olas-theme') || 'light');

  const switches = ['Switch1','Switch2','Switch3','Switch4'];
  const initialLogs = {{ logs | tojson if logs else [] }};

  function renderComparison(logs) {
    const grid = document.getElementById('comparisonGrid');
    if (!logs.length) {
      grid.innerHTML = '<div style="grid-column:span 4; text-align:center; padding:40px;">No data yet</div>';
      return;
    }
    const last = logs[0];
    let html = '';
    switches.forEach(sw => {
      const comp = last.comparison?.[sw] || {
        actual: false,
        predicted: last.predictions[sw].state,
        match: false,
        confidence: last.predictions[sw].confidence
      };
      const actual = comp.actual;
      const predicted = comp.predicted;
      const match = comp.match;
      const conf = Math.round(comp.confidence * 100);
      
      html += `<div class="comparison-card ${match ? 'match' : 'mismatch'}">
        <div class="device-name">${sw}</div>
        <div class="state-row">
          <span class="label">Actual:</span>
          <span class="value ${actual ? 'on' : 'off'}">${actual ? 'ON' : 'OFF'}</span>
        </div>
        <div class="state-row">
          <span class="label">Predicted:</span>
          <span class="value ${predicted ? 'on' : 'off'}">${predicted ? 'ON' : 'OFF'}</span>
        </div>
        <div class="match-status">${match ? '✅ Match' : '⚠️ Mismatch'}</div>`;
      
      if (!match) {
        html += `<div class="action-needed">Would ${predicted ? 'turn ON' : 'turn OFF'}</div>`;
      }
      
      html += `<div class="confidence">${conf}% confidence</div>
      </div>`;
    });
    grid.innerHTML = html;
  }

  function renderLogs(logs) {
    const container = document.getElementById('logFeedContainer');
    if (!logs.length) {
      container.innerHTML = '<div style="padding:20px; text-align:center;">No logs</div>';
      return;
    }
    let items = '';
    logs.slice(0,8).forEach(entry => {
      const matches = entry.comparison ? 
        Object.values(entry.comparison).filter(c => c.match).length : 0;
      items += `<div class="log-item">
        <span class="log-time">${entry.timestamp}</span>
        <span>Match: ${matches}/4 · ${entry.session}</span>
        <span style="margin-left: auto;">${entry.api_status}</span>
      </div>`;
    });
    container.innerHTML = items;
  }

  // Manual prediction
  document.getElementById('predictBtn').addEventListener('click', async () => {
    const h = document.getElementById('predHour').value;
    const m = document.getElementById('predMinute').value;
    const d = document.getElementById('predDay').value;
    const out = document.getElementById('manualPredictionOutput');
    out.innerHTML = '<p>Loading...</p>';
    try {
      const r = await fetch(`/predict_time/${h}/${m}/${d}`);
      const data = await r.json();
      let html = `<div><strong>${data.query}</strong> · ${data.session}</div><div style="display:flex; gap:20px; margin-top:12px;">`;
      switches.forEach(sw => {
        const p = data.predictions[sw];
        html += `<div class="preview-item"><span>${sw}</span><div style="font-size:1.5rem; font-weight:700;">${p.state?'ON':'OFF'}</div><small>${Math.round(p.confidence*100)}%</small></div>`;
      });
      html += '</div>';
      out.innerHTML = html;
    } catch(e) {
      out.innerHTML = '<p style="color:red;">Failed</p>';
    }
  });

  // Refresh button
  document.getElementById('refreshBtn').addEventListener('click', async () => {
    const btn = document.getElementById('refreshBtn');
    btn.disabled = true;
    btn.textContent = 'Fetching...';
    try {
      await fetch('/trigger');
      setTimeout(() => location.reload(), 500);
    } catch {
      alert('Failed');
      btn.disabled = false;
      btn.textContent = '⟳ Fetch & Predict';
    }
  });

  // Initialize
  renderComparison(initialLogs);
  renderLogs(initialLogs);
</script>
</body>
</html>
"""

# ── Flask routes ──────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD, logs=prediction_log)

@app.route("/status")
def status():
    last = prediction_log[0] if prediction_log else None
    return jsonify({
        "project": "OLAS 2026",
        "status": "running",
        "scheduler": "active — every 30 min (fetch & compare only)",
        "node_id": NODE_ID[:8] + "..." if len(NODE_ID) > 8 else NODE_ID,
        "actual_states": actual_states,
        "last_run": last["timestamp"] if last else None,
        "last_session": last["session"] if last else None,
    })

@app.route("/trigger", methods=["GET", "POST"])
def trigger():
    entry = predict_and_compare("manual_trigger")
    return jsonify({
        "message": "States fetched & prediction completed",
        "timestamp": entry["timestamp"],
        "session": entry["session"],
        "actual_states": entry["actual_states"],
        "predictions": entry["predictions"],
        "comparison": entry["comparison"],
    })

@app.route("/predict_time/<int:hour>/<int:minute>/<int:dow>")
def predict_time(hour, minute, dow):
    dt = datetime.datetime.now(IST).replace(hour=hour, minute=minute)
    feats = build_features(dt)
    preds = {}
    for sw in SWITCHES:
        state = int(models[sw].predict(feats)[0])
        prob = models[sw].predict_proba(feats)[0][state]
        preds[sw] = {"state": bool(state), "confidence": round(float(prob), 3)}
    return jsonify({
        "query": f"{hour:02d}:{minute:02d}  day_of_week={dow}",
        "session": current_session(dt),
        "predictions": preds
    })

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)