"""
Chaat AI — v2.0 (Render Migration)
===================================
Architecture: FastAPI (async) + Twilio ConversationRelay (barge-in) + Groq (STT + LLM)
Hosting: Render (Free Web Service)

Key changes from v1 (PythonAnywhere / Flask):
  - Flask  →  FastAPI + Uvicorn  (async, WebSocket-native)
  - Turn-based  →  ConversationRelay  (barge-in / interruption support)
  - Hardcoded strings  →  bot_config.json
  - Office Hours lock deployed (was parked in backlog)
  - All Anti-Jhal filters & race-condition guards carried over
"""

import os, json, time, csv, hashlib, hmac, base64
from datetime import datetime, timezone, timedelta
from functools import wraps

import groq
from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
from dotenv import load_dotenv
import uvicorn


load_dotenv()

# ─── ENV ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
TWILIO_ACCOUNT_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER       = os.getenv("TWILIO_NUMBER", "+1XXXXXXXXXX")
PORT                = int(os.getenv("PORT", 8000))  # Render sets PORT automatically



# ─── CONFIG LOADER ───────────────────────────────────────────────────────────
def load_config() -> dict:
    with open("bot_config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_prompt() -> str:
    with open("prompt.txt", "r", encoding="utf-8") as f:
        return f.read()

CONFIG = load_config()
GROQ_CLIENT = groq.Groq(api_key=GROQ_API_KEY)
TWILIO_CLIENT = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ─── FASTAPI APP ─────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve templates (login.html, dashboard.html) from /templates
app.mount("/static", StaticFiles(directory="templates"), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — OFFICE HOURS LOCK  (was parked in backlog, now live)
# ─────────────────────────────────────────────────────────────────────────────
def is_within_office_hours() -> bool:
    """Returns True if current time is within configured office hours (IST)."""
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    start_h, start_m = CONFIG["office_hours"]["start"]   # e.g. [9, 0]
    end_h,   end_m   = CONFIG["office_hours"]["end"]     # e.g. [20, 0]
    start = now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end   = now.replace(hour=end_h,   minute=end_m,   second=0, microsecond=0)
    return start <= now <= end


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ANTI-JHAL FILTER  (carried over from v1, unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────
GHOST_WORDS = set(CONFIG.get("ghost_words", []))
MIN_INPUT_LENGTH = CONFIG.get("min_input_length", 2)

def is_ghost_input(text: str) -> bool:
    """
    Returns True if the transcribed text is a Whisper hallucination.
    Checks: ghost word list + minimum length.
    """
    cleaned = text.strip().lower()
    if len(cleaned) < MIN_INPUT_LENGTH:
        return True
    # Check each word individually
    words = cleaned.split()
    for w in words:
        if w in GHOST_WORDS:
            return True
    # Also check the entire string (catches multi-word hallucinations)
    if cleaned in GHOST_WORDS:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CSV HELPERS  (flat-file DB, same as v1)
# ─────────────────────────────────────────────────────────────────────────────
def read_csv(filepath: str) -> list[dict]:
    rows = []
    try:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        pass
    return rows

def append_csv(filepath: str, row: dict, fieldnames: list[str]):
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — AUTH  (simple user/password from users.csv)
# ─────────────────────────────────────────────────────────────────────────────
def authenticate(username: str, password: str):
    users = read_csv("users.csv")
    for u in users:
        if u.get("Username") == username and u.get("Password") == password:
            return u
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GROQ: STT (Whisper) + LLM (Llama)
# ─────────────────────────────────────────────────────────────────────────────
def groq_transcribe(audio_bytes: bytes) -> str:
    """Whisper STT via Groq. Same settings as v1."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            transcription = GROQ_CLIENT.audio.transcriptions.create(
                file=("audio.wav", f),
                model="whisper-large-v3",
                prompt=CONFIG["whisper_context_prompt"],
                temperature=0.0,
                language="hi",
            )
        return transcription.text.strip()
    except Exception as e:
        print(f"[STT ERROR] {e}")
        return ""
    finally:
        os.unlink(tmp_path)


def groq_llm(user_input: str, conversation_history: list[dict]) -> str:
    """
    Groq LLM call. Same model + params as v1.
    Returns the raw text (with [ACTION:...] tags still in it).
    """
    system_prompt = load_prompt()
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})

    try:
        response = GROQ_CLIENT.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,  # type: ignore
            temperature=0.6,
            max_tokens=150,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except groq.RateLimitError:
        return CONFIG["messages"]["rate_limit_exit"]
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return CONFIG["messages"]["error_fallback"]


def parse_action(llm_text: str):
    """
    Extracts [ACTION: X] tag from LLM output.
    Returns (clean_text_to_speak, action_or_None).
    """
    action = None
    clean = llm_text
    if "[ACTION:" in llm_text.upper():
        import re
        # Find the action
        match = re.search(r'\[ACTION:\s*(.*?)\]', llm_text, re.IGNORECASE)
        if match:
            action = match.group(1).strip().upper()
        
        # Remove ALL action tags and clean up text
        clean = re.sub(r'\[ACTION:.*?\]', '', llm_text, flags=re.IGNORECASE)
        # Remove any extra whitespace and fix spacing
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Remove any remaining brackets (just in case)
        clean = re.sub(r'\[.*?\]', '', clean).strip()
    return clean, action


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3.5 — SUPABASE LOGGING  (replaces CSV)
# ─────────────────────────────────────────────────────────────────────────────
def log_call_to_csv_new(call_sid: str, final_action: str, conversation_history: list):
    """Log call to CSV with summary."""
    summary = generate_call_summary(conversation_history)
    
    # Get last user/bot messages for compatibility
    last_user = ""
    last_bot = ""
    for msg in reversed(conversation_history):
        if msg["role"] == "user" and not last_user:
            last_user = msg["content"]
        elif msg["role"] == "assistant" and not last_bot:
            last_bot = msg["content"]
    
    append_csv("call_logs.csv", {
        "TIME": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "CALL SID": call_sid,
        "USER SAID": last_user,
        "BOT SAID": last_bot,
        "OUTCOME": final_action,
        "CALL SUMMARY": summary,
    }, ["TIME", "CALL SID", "USER SAID", "BOT SAID", "OUTCOME", "CALL SUMMARY"])
    print(f"[CSV] Call logged: {call_sid} → {final_action}")
    



def generate_call_summary(conversation_history: list) -> str:
    """Generate a 2-3 line summary of the call."""
    if len(conversation_history) < 2:
        return "Brief call - minimal conversation"
    
    # Get key conversation points
    user_messages = [h["content"] for h in conversation_history if h["role"] == "user"]
    bot_messages = [h["content"] for h in conversation_history if h["role"] == "assistant"]
    
    # Simple summary logic
    if len(user_messages) == 1:
        summary = f"Customer said: '{user_messages[0][:50]}...'. "
    else:
        summary = f"Customer had {len(user_messages)} exchanges. "
    
    summary += f"Bot provided information about Google Maps services."
    
    return summary[:200]  # Limit to 200 characters








# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — IN-MEMORY CALL STATE  (replaces v1 race-condition guard)
# ─────────────────────────────────────────────────────────────────────────────
# Stores active call sessions: { call_sid: { "history": [...], "silence_count": int, "active": bool, "start_time": datetime } }
ACTIVE_CALLS = {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — TWILIO ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/incoming-call")
async def handle_incoming_call(request: Request):
    """
    Twilio webhook: called when someone answers an outbound call.
    Returns TwiML with Gather to collect speech, then processes it.
    This is the SIMPLE approach - works like v1 but on Render.
    """
    # Office hours check
    if not is_within_office_hours():
        resp = VoiceResponse()
        resp.say(CONFIG["messages"]["outside_hours"], language="hi-IN", voice="Google.hi-IN-Wavenet-A")
        return HTMLResponse(content=str(resp), media_type="application/xml")

    # Start the conversation
    resp = VoiceResponse()
    resp.say(CONFIG["messages"]["greeting"], language="hi-IN", voice="Google.hi-IN-Wavenet-A")
    
    # Redirect to input handler
    resp.redirect("/voice-input", method="POST")
    return HTMLResponse(content=str(resp), media_type="application/xml")


@app.post("/incoming-call-dynamic")
async def handle_incoming_call_dynamic(request: Request):
    """Same as /incoming-call but named for backward compatibility."""
    return await handle_incoming_call(request)


@app.post("/voice-input")
async def voice_input(request: Request):
    """
    Handles speech input from the caller.
    This uses Twilio's Gather with speech recognition, then we process with Groq LLM.
    """
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    speech_result = str(form.get("SpeechResult", "")).strip()
    
    # Initialize session if new
    if call_sid not in ACTIVE_CALLS:
        ACTIVE_CALLS[call_sid] = {
            "history": [], 
            "silence_count": 0, 
            "active": True,
            "start_time": datetime.now(timezone.utc)
        }
    
    session = ACTIVE_CALLS[call_sid]
    
    print(f"[VOICE] [{call_sid}] User said: '{speech_result}'")
    
    resp = VoiceResponse()
    
    # Anti-Jhal Filter
    if is_ghost_input(speech_result):
        session["silence_count"] += 1
        print(f"[VOICE] [{call_sid}] Ghost input (silence #{session['silence_count']})")
        
        if session["silence_count"] >= 3:
            resp.say(CONFIG["messages"]["silence_check"], language="hi-IN", voice="Google.hi-IN-Wavenet-A")
            session["silence_count"] = 0
        
        # Gather again
        gather = resp.gather(
            input="speech",
            language="hi-IN",
            speech_timeout="auto",
            action="/voice-input",
            method="POST"
        )
        return HTMLResponse(content=str(resp), media_type="application/xml")
    
    # Valid input - reset silence
    session["silence_count"] = 0
    
    # Get LLM response
    llm_raw = groq_llm(speech_result, session["history"])
    speak_text, action = parse_action(llm_raw)
    
    # Update history
    session["history"].append({"role": "user", "content": speech_result})
    session["history"].append({"role": "assistant", "content": llm_raw})
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]
    
    # Store final action in session for logging at call end
    if action:
        session["final_action"] = action
        session["final_transcript"] = speech_result
        session["final_bot_response"] = speak_text
        print(f"[VOICE] [{call_sid}] Action: {action}")
    
    # Speak response
    if speak_text:
        resp.say(speak_text, language="hi-IN", voice="Google.hi-IN-Wavenet-A")
        print(f"[VOICE] [{call_sid}] Bot: '{speak_text}'")
    
    # Terminal actions
    if action in ("NOT_INTERESTED", "RATE_LIMIT_EXIT"):
        resp.say(CONFIG["messages"]["goodbye"], language="hi-IN", voice="Google.hi-IN-Wavenet-A")
        resp.hangup()
        session["active"] = False
        return HTMLResponse(content=str(resp), media_type="application/xml")
    
    # Continue conversation - gather next input
    gather = resp.gather(
        input="speech",
        language="hi-IN",
        speech_timeout="auto",
        action="/voice-input",
        method="POST"
    )
    
    return HTMLResponse(content=str(resp), media_type="application/xml")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — POWER DIALER API  (same logic as v1, adapted for FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/get-next-lead")
async def get_next_lead():
    """Fetch the next un-called lead from leads.csv."""
    leads = read_csv("leads.csv")
    logs  = read_csv("call_logs.csv")
    called_sids = {log.get("CallSID") for log in logs}  # Not perfect, but simple

    # For simplicity: return leads in order. The dashboard tracks index client-side.
    if not leads:
        return JSONResponse({"status": "no_leads", "lead": None})
    return JSONResponse({"status": "ok", "leads": leads})
    
    try:
        # Get leads from CSV for now
        leads = read_csv("leads.csv")
        if not leads:
            return JSONResponse({"status": "no_leads", "lead": None})
        return JSONResponse({"status": "ok", "leads": leads})


@app.post("/trigger-call")
async def trigger_call(request: Request):
    """
    Initiate an outbound call to a lead via Twilio.
    Note: Office hours removed for dashboard - calls can be triggered anytime.
    """

    body = await request.json()
    phone = body.get("phone", "").strip()
    name  = body.get("name", "Unknown")
    lead_id = body.get("id")  # New: lead ID for Supabase

    print(f"[DEBUG] Trigger call request: {body}")  # Debug line
    
    if not phone:
        print(f"[ERROR] No phone number in request: {body}")
        return JSONResponse({"status": "error", "message": "No phone number provided"}, status_code=400)

    try:
        # The /incoming-call-dynamic endpoint returns the TwiML with ConversationRelay
        # Render will auto-set the host, so we use a relative URL trick:
        # Twilio needs an absolute URL, so we build it from the request
        host = request.url.hostname
        scheme = "https"  # Render always serves HTTPS

        call = TWILIO_CLIENT.calls.create(
            to=phone,
            from_=TWILIO_NUMBER,
            url=f"{scheme}://{host}/incoming-call-dynamic",
            status_callback=f"{scheme}://{host}/call-events",
            status_callback_event=["completed", "failed", "no-answer", "canceled"],
            timeout=30,
        )

        # TODO: Update lead status when we have database
    # For now, just log the call
        
        print(f"[DIALER] Call initiated: {call.sid} → {name} ({phone})")
        return JSONResponse({"status": "ok", "call_sid": call.sid, "name": name})

    except Exception as e:
        print(f"[DIALER] Error calling {phone}: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/call-events")
async def call_events(request: Request):
    """Twilio status callback. Logs call completion events."""
    form = await request.form()
    call_sid = str(form.get("CallSid", ""))
    status   = form.get("CallStatus", "")
    print(f"[EVENTS] {call_sid} → {status}")

    # Log final outcome when call completes
    if status in ("completed", "failed", "no-answer", "canceled") and call_sid in ACTIVE_CALLS:
        session = ACTIVE_CALLS[call_sid]
        
        # Get final outcome
        final_action = session.get("final_action", "UNKNOWN")
        conversation_history = session.get("history", [])
        start_time = session.get("start_time", datetime.now(timezone.utc))
        
    # Log to CSV
    log_call_to_csv_new(call_sid, final_action, conversation_history)
    print(f"[FINAL] [{call_sid}] Logged outcome: {final_action}")

    # Clean up in-memory state if call ended
    if call_sid in ACTIVE_CALLS:
        ACTIVE_CALLS[call_sid]["active"] = False
        del ACTIVE_CALLS[call_sid]

    return JSONResponse({"received": True})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — DASHBOARD / AUTH ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("templates/login.html")

@app.get("/login")
async def login_page():
    return FileResponse("templates/login.html")

@app.post("/auth")
async def auth(request: Request):
    body = await request.json()
    user = authenticate(body.get("username", ""), body.get("password", ""))
    if user:
        return JSONResponse({"status": "ok", "role": user.get("Role", "user")})
    return JSONResponse({"status": "error", "message": "Invalid credentials"}, status_code=401)

@app.get("/dashboard")
async def dashboard():
    return FileResponse("templates/dashboard.html")

@app.get("/call-logs")
async def get_call_logs():
    """Get call logs from CSV."""
    return JSONResponse(read_csv("call_logs.csv"))

@app.get("/leads")
async def get_leads():
    """Get leads from CSV."""
    return JSONResponse(read_csv("leads.csv"))

@app.get("/config")
async def get_config():
    return JSONResponse(load_config())

@app.post("/update-config")
async def update_config(request: Request):
    """Update bot_config.json from the dashboard."""
    new_config = await request.json()
    with open("bot_config.json", "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False)
    global CONFIG
    CONFIG = load_config()
    return JSONResponse({"status": "ok"})

@app.get("/prompt")
async def get_prompt():
    return JSONResponse({"prompt": load_prompt()})

@app.post("/update-prompt")
async def update_prompt(request: Request):
    """Update prompt.txt from the dashboard."""
    body = await request.json()
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(body.get("prompt", ""))
    return JSONResponse({"status": "ok"})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — RENDER HEALTH CHECK  (Render pings /healthz to keep free tier alive)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/healthz")
async def health():
    return JSONResponse({"status": "alive", "time": datetime.now(timezone.utc).isoformat()})


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
