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
<title>OLAS Dashboard</title>
<meta http-equiv="refresh" content="60">

<style>

/* ---------- GLOBAL ---------- */
body{
  font-family: 'Segoe UI', Arial, sans-serif;
  background:#0b1a24;
  margin:0;
  padding:20px;
  color:#e6f1f7;
  -webkit-font-smoothing: antialiased;
}

.container{
  max-width:1100px;
  margin:auto;
}

/* ---------- HEADER ---------- */
.header{
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:25px;
}

.title{
  font-size:30px;
  font-weight:700;
  color:#00d4ff;
  letter-spacing:1px;
}

.status{
  font-size:13px;
  padding:6px 14px;
  border-radius:20px;
  font-weight:500;
}

.online{
  background:#0f3d2e;
  color:#00ffb3;
  border:1px solid #00ffb3;
}

.offline{
  background:#3d0f0f;
  color:#ff4d4d;
  border:1px solid #ff4d4d;
}

/* ---------- CARDS ---------- */
.card{
  background:#122836;
  border-radius:12px;
  padding:18px;
  margin-bottom:20px;
  border:1px solid #1e3a4a;
  box-shadow:0 0 10px rgba(0,212,255,0.08);
}

.card h3{
  margin-top:0;
  color:#00d4ff;
}

/* ---------- GRID ---------- */
.grid{
  display:grid;
  grid-template-columns:repeat(2,1fr);
  gap:15px;
}

/* ---------- SWITCH ---------- */
.switch{
  padding:14px;
  border-radius:10px;
  text-align:center;
  font-weight:600;
  transition:0.2s;
}

.on{
  background:#0f3a45;
  color:#00ffd5;
  border:1px solid #00ffd5;
}

.off{
  background:#1a2e3a;
  color:#6c8a99;
}

/* ---------- PROGRESS ---------- */
.bar{
  height:6px;
  background:#1f3c4d;
  border-radius:5px;
  margin-top:10px;
}

.fill{
  height:6px;
  background:#00d4ff;
  border-radius:5px;
}

/* ---------- BUTTONS ---------- */
.btn{
  padding:10px 16px;
  border-radius:6px;
  background:#00d4ff;
  color:#001018;
  text-decoration:none;
  font-size:14px;
  font-weight:600;
  margin-right:10px;
  display:inline-block;
}

.btn:hover{
  background:#00aacc;
}

/* ---------- TABLE ---------- */
.table{
  width:100%;
  border-collapse:collapse;
}

.table th,.table td{
  padding:10px;
  text-align:center;
  border-bottom:1px solid #1f3c4d;
}

.table th{
  color:#00d4ff;
}

.green{color:#00ffb3;}
.red{color:#ff4d4d;}

/* ---------- MOBILE ---------- */
@media(max-width:600px){
  .grid{
    grid-template-columns:1fr;
  }
}

</style>
</head>

<body>
<div class="container">

<!-- HEADER -->
<div class="header">
  <div class="title">⚡ OLAS Dashboard</div>
  {% if logs %}
    <div class="status online">SYSTEM ONLINE</div>
  {% else %}
    <div class="status offline">NO DATA</div>
  {% endif %}
</div>

{% if logs %}
{% set last = logs[0] %}

<!-- SWITCH STATUS -->
<div class="card">
  <h3>🔌 Switch Status</h3>
  <div class="grid">
    {% for sw in ['Switch1','Switch2','Switch3','Switch4'] %}
    {% set on = last.predictions[sw].state %}
    {% set pct = (last.predictions[sw].confidence * 100)|int %}
    <div class="switch {{ 'on' if on else 'off' }}">
      <div>{{ sw }}</div>
      <div style="margin-top:5px;">{{ 'ON' if on else 'OFF' }}</div>
      <div class="bar">
        <div class="fill" style="width:{{ pct }}%"></div>
      </div>
      <div style="font-size:12px;margin-top:5px;">{{ pct }}%</div>
    </div>
    {% endfor %}
  </div>
</div>

<!-- SYSTEM INFO -->
<div class="card">
  <h3> System Info</h3>
  <p><b>Session:</b> {{ last.session }}</p>
  <p><b>Last Run:</b> {{ last.timestamp }}</p>
  <p><b>API Status:</b>
    {% if last.api_status == 'ok' %}
      <span class="green">Connected</span>
    {% else %}
      <span class="red">{{ last.api_status }}</span>
    {% endif %}
  </p>
</div>

{% endif %}

<!-- ACTIONS -->
<div class="card">
  <h3> Actions</h3>
  <a href="/trigger" class="btn">Run Prediction</a>
  <a href="/status" class="btn">View JSON</a>
</div>

<!-- LOGS -->
<div class="card">
  <h3> Logs</h3>
  <table class="table">
    <tr>
      <th>Time</th>
      <th>S1</th><th>S2</th><th>S3</th><th>S4</th>
      <th>API</th>
    </tr>

    {% for entry in logs %}
    <tr>
      <td>{{ entry.timestamp }}</td>

      {% for sw in ['Switch1','Switch2','Switch3','Switch4'] %}
      <td class="{{ 'green' if entry.predictions[sw].state else 'red' }}">
        {{ 'ON' if entry.predictions[sw].state else 'OFF' }}
      </td>
      {% endfor %}

      <td>{{ entry.api_status }}</td>
    </tr>
    {% endfor %}
  </table>
</div>

</div>
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

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

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

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
