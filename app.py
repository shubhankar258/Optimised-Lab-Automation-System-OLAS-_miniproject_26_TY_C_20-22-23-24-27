# ============================================================
#  OLAS — Optimised Lab Automation System
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
    - If token is fresh (< 50 min old) → returns cached token
    - If token is stale → tries silent refresh via Cognito
    - If refresh fails → does full re-login with email + password
    Access tokens last ~60 min. We refresh at 50 min to stay safe.
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

# ── Login on startup ──────────────────────────────────────────
if RM_EMAIL != "your@email.com":
    try:
        login_and_get_tokens()
    except Exception as e:
        log.error(f"Startup login failed: {e}")
        log.warning("Scheduler will retry login on first prediction attempt")

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

# ── Core: predict → compare → send to RainMaker ──────────────
def predict_and_control(source: str = "scheduler"):
    now     = datetime.datetime.now()
    feats   = build_features(now)
    session = current_session(now)

    # Run all 4 classifiers
    predictions = {}
    for sw in SWITCHES:
        pred = int(models[sw].predict(feats)[0])
        prob = models[sw].predict_proba(feats)[0][pred]
        predictions[sw] = {
            "state"     : bool(pred),
            "confidence": round(float(prob), 3)
        }

    # Build RainMaker payload — "Power" matches write_callback in ESP32 firmware
    payload = [
    {
        "node_id": NODE_ID,
        "payload": {
            sw: {
                "output": predictions[sw]["state"]
            }
            for sw in SWITCHES
        }
    }
]

    api_status = "not_sent"

    if RM_EMAIL != "your@email.com":
        try:
            # Always fetch a fresh (or cached) token — never hardcoded
            fresh_headers = {
                "Authorization": f"Bearer {get_access_token()}",
                "Content-Type" : "application/json"
            }
            r = requests.put(
                API_URL,
                headers=fresh_headers,
                json=payload,
                timeout=10
            )
            api_status = "ok" if r.status_code == 200 else f"error_{r.status_code}"
            if r.status_code == 200:
                log.info(f"Commands sent OK  |  {session}")
            else:
                log.warning(f"API returned {r.status_code}: {r.text[:120]}")
        except Exception as e:
            api_status = "connection_error"
            log.error(f"API call failed: {e}")
    else:
        api_status = "credentials_not_set"
        log.warning("RAINMAKER_EMAIL not configured — running in preview mode")

    # Store entry in memory log
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

# ── Scheduler thread ─────────────────────────────────────────
schedule.every(30).minutes.do(lambda: predict_and_control("scheduler"))

def run_scheduler():
    log.info("Scheduler started — firing every 30 minutes")
    predict_and_control("startup")
    while True:
        schedule.run_pending()
        time.sleep(30)

scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

# ── Dashboard HTML (OLAS Iron Man theme) ─────────────────────
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
      --shadow-sm: 0 2px 8px rgba(0,0,0,0.04);
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
      --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
      --shadow-md: 0 8px 24px rgba(0,0,0,0.4);
    }

    body {
      background: var(--bg-page);
      color: var(--text-primary);
      padding: 20px 24px;
      transition: background 0.2s;
    }

    .dashboard { max-width: 1200px; margin: 0 auto; }

    /* ----- TOP NAVIGATION BAR (clean, minimal) ----- */
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
      letter-spacing: -0.5px;
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
      border: 1px solid var(--border-light);
    }

    .pulse-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--success);
      box-shadow: 0 0 6px var(--success);
      animation: pulse 1.8s infinite;
    }

    .status-text {
      font-weight: 500;
      font-size: 0.9rem;
      color: var(--accent);
    }

    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

    /* ----- Device Grid ----- */
    .section-title {
      font-size: 1.2rem;
      font-weight: 600;
      margin: 24px 0 16px;
      color: var(--text-primary);
    }

    .device-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 20px;
    }

    .device-card {
      background: var(--card-bg);
      border-radius: 24px;
      padding: 24px 16px 20px;
      box-shadow: var(--shadow-md);
      border: 1px solid var(--border-light);
      text-align: center;
      transition: transform 0.2s;
    }

    .device-card:hover { transform: translateY(-4px); }

    .device-icon {
      font-size: 36px;
      margin-bottom: 12px;
    }

    .device-name {
      font-weight: 600;
      font-size: 1.1rem;
      margin-bottom: 16px;
    }

    .device-state {
      display: inline-block;
      padding: 6px 20px;
      border-radius: 40px;
      font-weight: 600;
      font-size: 0.9rem;
      margin-bottom: 16px;
    }

    .state-on {
      background: var(--success);
      color: white;
    }

    .state-off {
      background: var(--border-light);
      color: var(--text-secondary);
    }

    .confidence-meter {
      width: 100%;
      height: 4px;
      background: var(--border-light);
      border-radius: 4px;
      overflow: hidden;
    }

    .confidence-fill {
      height: 100%;
      background: var(--accent);
      border-radius: 4px;
    }

    .confidence-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 8px;
    }

    /* ----- Automation Card ----- */
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

    .automation-left h3 {
      font-size: 1.2rem;
      margin-bottom: 6px;
    }

    .automation-left p {
      color: var(--text-muted);
      font-size: 0.9rem;
    }

    .automation-controls {
      display: flex;
      align-items: center;
      gap: 20px;
    }

    .toggle-switch {
      position: relative;
      width: 56px;
      height: 30px;
      background: var(--border-light);
      border-radius: 40px;
      cursor: pointer;
      transition: 0.2s;
    }

    .toggle-switch.active {
      background: var(--accent);
    }

    .toggle-slider {
      position: absolute;
      top: 3px;
      left: 3px;
      width: 24px;
      height: 24px;
      background: white;
      border-radius: 50%;
      transition: 0.2s;
    }

    .active .toggle-slider { left: 29px; }

    /* ----- Manual Check Section ----- */
    .manual-section {
      background: var(--card-bg);
      border-radius: 24px;
      padding: 24px;
      margin-bottom: 28px;
      border: 1px solid var(--border-light);
      box-shadow: var(--shadow-md);
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

    .btn {
      background: var(--accent);
      color: white;
      border: none;
      padding: 12px 24px;
      border-radius: 40px;
      font-weight: 600;
      cursor: pointer;
      transition: 0.2s;
    }

    .btn-outline {
      background: transparent;
      border: 1px solid var(--border-light);
      color: var(--text-primary);
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

    .preview-item .label { color: var(--text-muted); font-size: 0.85rem; }
    .preview-item .value { font-size: 1.8rem; font-weight: 700; }

    /* ----- Activity Log Feed ----- */
    .log-feed {
      background: var(--card-bg);
      border-radius: 24px;
      padding: 20px;
      border: 1px solid var(--border-light);
      box-shadow: var(--shadow-md);
    }

    .log-item {
      display: flex;
      align-items: center;
      padding: 12px 0;
      border-bottom: 1px solid var(--border-light);
    }

    .log-time {
      min-width: 140px;
      color: var(--text-muted);
      font-size: 0.85rem;
    }

    .log-switches {
      display: flex;
      gap: 16px;
      margin: 0 20px;
    }

    .log-switch {
      font-weight: 600;
      font-size: 0.9rem;
    }

    .state-on { color: var(--success); }
    .state-off { color: var(--text-muted); }

    .log-session {
      color: var(--text-secondary);
      font-size: 0.85rem;
      margin-left: auto;
    }

    .status-tag {
      padding: 4px 10px;
      border-radius: 40px;
      font-size: 0.75rem;
      font-weight: 600;
      background: var(--accent-soft);
      color: var(--accent);
      margin-left: 16px;
    }

    @media (max-width: 800px) {
      .device-grid { grid-template-columns: repeat(2, 1fr); }
      .top-nav { flex-direction: column; align-items: flex-start; }
    }

    @media (max-width: 500px) {
      .device-grid { grid-template-columns: 1fr; }
      .automation-card { flex-direction: column; align-items: flex-start; }
      .nav-controls { flex-wrap: wrap; }
    }
  </style>
</head>
<body>
<div class="dashboard">
  <!-- TOP NAVIGATION BAR: OLAS | Theme Toggle | System Status -->
  <div class="top-nav">
    <div class="project-name">OLAS</div>
    <div class="nav-controls">
      <div class="theme-toggle">
        <button class="theme-btn active" data-theme="light" onclick="setTheme('light')">☀️ Light</button>
        <button class="theme-btn" data-theme="dark" onclick="setTheme('dark')">🌙 Dark</button>
      </div>
      <div class="status-badge">
        <span class="pulse-dot"></span>
        <span class="status-text">System Online · RainMaker</span>
      </div>
    </div>
  </div>

  <!-- DEVICE GRID -->
  <div class="section-title">🔌 Connected Devices</div>
  <div class="device-grid" id="deviceGrid">
    <!-- Populated by JS -->
  </div>

  <!-- ML AUTOMATION CARD -->
  <div class="automation-card">
    <div class="automation-left">
      <h3>🤖 ML Automation</h3>
      <p>Predictive control based on lab schedule</p>
    </div>
    <div class="automation-controls">
      <div class="toggle-switch active" id="automationToggle">
        <div class="toggle-slider"></div>
      </div>
      <button class="btn btn-outline" id="runNowBtn">▶ Run Now</button>
    </div>
  </div>

  <!-- MANUAL MODEL CHECK -->
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

  <!-- ACTIVITY LOG FEED -->
  <div class="log-feed">
    <h3 style="margin-bottom: 16px;">📋 Recent Activity</h3>
    <div id="logFeedContainer"></div>
  </div>
</div>

<script>
  // ---------- Theme ----------
  function setTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    document.querySelectorAll('.theme-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-theme="${t}"]`).classList.add('active');
    localStorage.setItem('olas-theme', t);
  }
  const saved = localStorage.getItem('olas-theme') || 'light';
  setTheme(saved);

  const switches = ['Switch1','Switch2','Switch3','Switch4'];
  const initialLogs = {{ logs | tojson if logs else [] }};
  let automationEnabled = true;

  function renderDevices(logs) {
    const grid = document.getElementById('deviceGrid');
    if (!logs.length) {
      grid.innerHTML = '<div style="grid-column:span 4; text-align:center; padding:40px; color:var(--text-muted);">No data yet</div>';
      return;
    }
    const last = logs[0];
    let html = '';
    switches.forEach(sw => {
      const pred = last.predictions[sw];
      const on = pred.state;
      const conf = Math.round(pred.confidence*100);
      html += `<div class="device-card">
        <div class="device-icon">💡</div>
        <div class="device-name">${sw}</div>
        <div class="device-state ${on ? 'state-on' : 'state-off'}">${on ? 'ON' : 'OFF'}</div>
        <div class="confidence-meter"><div class="confidence-fill" style="width:${conf}%"></div></div>
        <div class="confidence-label">${conf}% confidence</div>
      </div>`;
    });
    grid.innerHTML = html;
  }

  function renderLogs(logs) {
    const container = document.getElementById('logFeedContainer');
    if (!logs.length) {
      container.innerHTML = '<div style="padding:20px; text-align:center; color:var(--text-muted);">No activity yet</div>';
      return;
    }
    let items = '';
    logs.slice(0,8).forEach(entry => {
      const states = switches.map(sw => `<span class="log-switch ${entry.predictions[sw].state ? 'state-on' : 'state-off'}">${sw.slice(-1)}:${entry.predictions[sw].state?'ON':'OFF'}</span>`).join('');
      items += `<div class="log-item">
        <span class="log-time">${entry.timestamp}</span>
        <div class="log-switches">${states}</div>
        <span class="log-session">${entry.session}</span>
        <span class="status-tag">${entry.api_status}</span>
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
        html += `<div class="preview-item"><span class="label">${sw}</span><div class="value">${p.state?'ON':'OFF'}</div><small>${Math.round(p.confidence*100)}%</small></div>`;
      });
      html += '</div>';
      out.innerHTML = html;
    } catch(e) {
      out.innerHTML = '<p style="color:red;">Prediction failed</p>';
    }
  });

  // Run now
  document.getElementById('runNowBtn').addEventListener('click', async (e) => {
    const btn = e.target;
    btn.disabled = true;
    btn.textContent = 'Running...';
    try {
      await fetch('/trigger');
      setTimeout(() => location.reload(), 500);
    } catch {
      alert('Failed');
      btn.disabled = false;
      btn.textContent = '▶ Run Now';
    }
  });

  // Automation toggle
  document.getElementById('automationToggle').addEventListener('click', function() {
    automationEnabled = !automationEnabled;
    this.classList.toggle('active', automationEnabled);
  });

  // Initialize
  renderDevices(initialLogs);
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
        "project"     : "OLAS 2026",
        "status"      : "running",
        "scheduler"   : "active — every 30 min",
        "node_id"     : NODE_ID[:8] + "..." if len(NODE_ID) > 8 else NODE_ID,
        "last_run"    : last["timestamp"] if last else None,
        "last_session": last["session"]   if last else None,
        "token_age_s" : round(time.time() - _token_cache["fetched_at"]) if _token_cache["fetched_at"] else None,
    })

@app.route("/trigger", methods=["GET", "POST"])
def trigger():
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
    Test prediction for any time without sending to ESP32.
    /predict_time/9/30/0   → Monday 9:30   (Session 1)
    /predict_time/12/0/1   → Tuesday 12:00 (Session 2)
    /predict_time/14/15/2  → Wednesday 14:15 (Session 3)
    /predict_time/17/0/0   → Monday 17:00  (Outside)
    dow: 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri 5=Sat 6=Sun
    """
    dt    = datetime.datetime.now().replace(hour=hour, minute=minute)
    feats = build_features(dt)
    preds = {}
    for sw in SWITCHES:
        state = int(models[sw].predict(feats)[0])
        prob  = models[sw].predict_proba(feats)[0][state]
        preds[sw] = {"state": bool(state), "confidence": round(float(prob), 3)}
    return jsonify({
        "query"      : f"{hour:02d}:{minute:02d}  day_of_week={dow}",
        "session"    : current_session(dt),
        "predictions": preds
    })

# Add near the top of app.py
automation_enabled = True

@app.route('/toggle_automation', methods=['POST'])
def toggle_automation():
    global automation_enabled
    data = requests.get_json()
    automation_enabled = data.get('enabled', True)
    log.info(f"ML Automation toggled: {'ON' if automation_enabled else 'OFF'}")
    return jsonify({"automation_enabled": automation_enabled})

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)