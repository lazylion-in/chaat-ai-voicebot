# Chaat AI v2.0 — Render Migration Guide

## What Changed & Why

| v1 (PythonAnywhere)                | v2 (Render)                              |
|------------------------------------|------------------------------------------|
| Flask (sync)                       | FastAPI + Uvicorn (async, WebSocket-native) |
| Turn-based (walkie-talkie)         | ConversationRelay → **barge-in works**   |
| Hardcoded strings in server.py     | All strings in `bot_config.json` (editable from dashboard) |
| Office Hours: drafted, not deployed| Office Hours: **live** ✅                 |
| Port 8080                          | Render auto-assigns PORT via env var     |

### Why Render free tier works for this use case

The big risk with Render's free tier is the 15-min spin-down. But Chaat AI is a **power dialer** — it only runs when you're actively making calls. During a campaign, the dashboard keeps hitting the server every 60s (triggering calls), so it stays awake. When you're not campaigning, it's fine to spin down — nothing needs to run.

The barge-in (interruption) works because **Twilio's ConversationRelay keeps the WebSocket alive on Twilio's side**, not yours. Your server just responds to events. Even if there's a brief hiccup, Twilio retries.

---

## Step 1: Set Up Your GitHub Repo

```bash
# On your computer:
git init chaat-ai
cd chaat-ai

# Copy ALL these files into this folder:
# server.py
# bot_config.json
# prompt.txt          ← Keep your existing prompt.txt from PythonAnywhere
# requirements.txt
# render.yaml
# leads.csv           ← Copy from PythonAnywhere
# call_logs.csv       ← Copy from PythonAnywhere (or start fresh)
# users.csv           ← Copy from PythonAnywhere
# templates/
#   login.html
#   dashboard.html

git add .
git commit -m "Chaat AI v2 - Render migration"
git remote add origin https://github.com/YOUR_USERNAME/chaat-ai.git
git push -u origin main
```

---

## Step 2: Deploy on Render

1. Go to **render.com** → Sign in
2. Click **"New +"** → **Web Service**
3. Connect your GitHub repo (`chaat-ai`)
4. Render auto-detects `render.yaml`. If not, set manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**

---

## Step 3: Add Environment Variables

In Render Dashboard → your service → **Environment** tab, add these ONE BY ONE:

| Key                  | Value                          |
|----------------------|--------------------------------|
| `GROQ_API_KEY`       | Your Groq API key              |
| `TWILIO_ACCOUNT_SID` | From Twilio Console            |
| `TWILIO_AUTH_TOKEN`  | From Twilio Console            |
| `TWILIO_NUMBER`      | Your Twilio phone number (e.g. `+15551234567`) |

> ⚠️ Do NOT put these in `.env` and push to GitHub. Render handles them securely.

After adding env vars, click **"Manual Deploy"** to restart with the new config.

---

## Step 4: Wire Up Twilio

This is the most important step. Twilio needs to know where to send calls.

1. Go to **Twilio Console** → **Phone Numbers** → click your number
2. Under **"A Call Comes In"** (or Voice → Configure):
   - Set the **webhook URL** to:
     ```
     https://YOUR-RENDER-URL.onrender.com/incoming-call-dynamic
     ```
   - Method: **POST**

> 💡 Your Render URL looks like: `https://chaat-ai.onrender.com`
> Find it in your Render dashboard at the top of your service page.

### How the call flow works now:

```
You click "Start" in Dashboard
  → Dashboard calls POST /trigger-call
    → server.py calls Twilio API (outbound call)
      → Person answers
        → Twilio GETs /incoming-call-dynamic
          → server.py returns TwiML with <Connect><Stream url="wss://...">
            → Twilio opens WebSocket to /ws
              → ConversationRelay handles STT, TTS, and barge-in
              → Your server just processes text events via the WebSocket
```

---

## Step 5: Test It

1. Open your Render URL → Login page appears
2. Login with credentials from your `users.csv`
3. You'll see the Power Dialer dashboard
4. Make sure `leads.csv` has at least one lead
5. Click **▶ Start** — it will call the first lead
6. Try talking over the bot — **barge-in should work** (this is the big upgrade!)

---

## Step 6: Keep the Free Tier Alive (Optional but Recommended)

Render's free tier spins down after 15 min of no traffic. During active campaigns this isn't an issue (the dialer hits the server every 60s). But if you want it always-on, you can set up a simple free cron ping:

- Use a free service like **uptimerobot.com** or **cron-job.org**
- Set it to ping `https://YOUR-RENDER-URL.onrender.com/healthz` every 10 minutes
- This keeps the server warm between campaigns at zero cost

---

## Quick Reference: File Roles

| File                    | What it does                                              |
|-------------------------|-----------------------------------------------------------|
| `server.py`             | The brain. FastAPI server, WebSocket handler, all logic   |
| `bot_config.json`       | All editable settings (ghost words, messages, hours)      |
| `prompt.txt`            | The AI sales persona (edit via dashboard or directly)     |
| `leads.csv`             | Phone numbers to call: `Name,Phone`                       |
| `call_logs.csv`         | Auto-generated call history with AI-judged outcomes       |
| `users.csv`             | Login credentials: `Username,Password,Role`               |
| `render.yaml`           | Render deployment config                                  |
| `requirements.txt`      | Python dependencies                                       |
| `templates/`            | Dashboard and login HTML                                  |

---

## What's Still on the Backlog (V3)

- **Redis Semantic Cache** — Cache common Q&A to reduce Groq latency
- **Plivo Migration** — Better India connectivity (parked until scaling)
- **+91 DLT Registration** — For calling Indian numbers from a +91 number
