import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import torch
import sqlite3
import bcrypt
import json
import base64
import pandas as pd

from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.pdfgen import canvas

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Medical Text Simplifier", layout="wide")

HISTORY_FILE = "history.json"
REMEMBER_FILE = "remember.json"

# ---------------- BACKGROUND ----------------
def set_background(image):
    if not os.path.exists(image):
        return

    with open(image, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image:url("data:image/png;base64,{data}");
        background-size:cover;
        background-attachment:fixed;
    }}
    .block-container {{
        background:rgba(0,0,0,0.6);
        padding:2rem;
        border-radius:10px;
    }}
    h1,h2,h3,h4,p,label,span {{
        color:white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

set_background("background.jpeg")

# ---------------- AUTO LOGIN ----------------
def save_login(username):
    with open(REMEMBER_FILE, "w") as f:
        json.dump({"user": username}, f)

def load_login():
    if os.path.exists(REMEMBER_FILE):
        try:
            return json.load(open(REMEMBER_FILE))["user"]
        except:
            return None
    return None

def clear_login():
    if os.path.exists(REMEMBER_FILE):
        os.remove(REMEMBER_FILE)

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    saved = load_login()
    if saved:
        st.session_state.logged_in = True
        st.session_state.username = saved
    else:
        st.session_state.logged_in = False

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        username TEXT PRIMARY KEY,
        password BLOB
    )
    """)
    conn.commit()
    conn.close()

def add_user(u, p):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        hashed = bcrypt.hashpw(p.encode(), bcrypt.gensalt())
        c.execute("INSERT INTO users VALUES (?,?)", (u, hashed))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def verify_user(u, p):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (u,))
    row = c.fetchone()
    conn.close()

    return row and bcrypt.checkpw(p.encode(), row[0])

def update_password(u, p):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()

    c.execute("SELECT username FROM users WHERE username=?", (u,))
    if not c.fetchone():
        return False

    hashed = bcrypt.hashpw(p.encode(), bcrypt.gensalt())
    c.execute("UPDATE users SET password=? WHERE username=?", (hashed, u))
    conn.commit()
    conn.close()
    return True

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tokenizer, model

# ---------------- VALIDATION ----------------
def is_medical_input(text):
    text = text.lower().strip()

    keywords = [
        "hypertension","diabetes","cancer","infection",
        "asthma","arthritis","anemia","tumor",
        "aspirin","paracetamol","ibuprofen","metformin"
    ]

    suffix = ["itis","osis","emia","pathy","oma","tension"]
    meds = ["cin","azole","statin","pril","sartan"]

    if any(k in text for k in keywords):
        return True
    if any(text.endswith(s) for s in suffix+meds):
        return True
    if len(text.split()) >= 2:
        return True

    return False

# ---------------- AI SCORE ----------------
def ai_confidence_score(inp, out):

    inp_words = set(inp.lower().split())
    out_words = set(out.lower().split())

    # similarity (low weight)
    overlap = len(inp_words & out_words)
    similarity = (overlap / (len(inp_words) + 1)) * 100

    # output length quality (high weight)
    length = len(out.split())
    length_score = min(length * 4, 100)

    # penalty for poor output
    penalty = 30 if length < 6 else 0

    score = (similarity * 0.3) + (length_score * 0.7) - penalty

    return int(max(55, min(score, 95)))
# ---------------- SIMPLIFY ----------------
def simplify_text(text, tokenizer, model, level):

    if level == "Mild":
        prompt = f"Define '{text}' in one medical sentence."
        max_tokens = 50

    elif level == "Medium":
        prompt = f"Explain '{text}' in simple 2-3 sentences for a patient."
        max_tokens = 120

    else:
        prompt = f"Explain '{text}' in very simple language with example in 3-4 sentences."
        max_tokens = 180

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.9,
            repetition_penalty=1.5
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ---------------- HISTORY ----------------
def save_history(text, result, level):
    try:
        history = json.load(open(HISTORY_FILE))
    except:
        history = []

    history.append({
        "text": text,
        "result": result,
        "level": level,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M")
    })

    json.dump(history, open(HISTORY_FILE, "w"))

def load_history():
    try:
        return json.load(open(HISTORY_FILE))
    except:
        return []

# ---------------- PDF ----------------
def create_pdf(o, s):
    file = "result.pdf"
    c = canvas.Canvas(file)
    c.drawString(100, 750, "Medical Simplification")
    c.drawString(100, 700, f"Original: {o}")
    c.drawString(100, 650, f"Simplified: {s}")
    c.save()
    return file

# ---------------- DASHBOARD ----------------
def dashboard():
    st.title("Dashboard")

    data = load_history()
    if not data:
        st.info("No data yet")
        return

    df = pd.DataFrame(data)

    st.metric("Total Simplifications", len(df))
    st.bar_chart(df["level"].value_counts())

# ---------------- HISTORY ----------------
def history_page():
    st.title("History")

    for item in reversed(load_history()):
        st.write("Input:", item["text"])
        st.write("Output:", item["result"])
        st.write("Level:", item["level"])
        st.write("Time:", item["time"])
        st.divider()

# ---------------- SIMPLIFIER ----------------
def simplifier():
    tokenizer, model = load_model()

    st.title("Medical Text Simplifier")

    level = st.select_slider("Simplification Level", ["Mild","Medium","Strong"])
    text = st.text_area("Enter medical term or medicine")

    if st.button("Simplify"):

        text = text.strip()

        if not text:
            st.warning("Enter text")
            return

        if not is_medical_input(text):
            st.error("Invalid medical input")
            return

        result = simplify_text(text, tokenizer, model, level)
        save_history(text, result, level)

        score = ai_confidence_score(text, result)

        col1, col2 = st.columns(2)

        with col1:
            st.info(text)

        with col2:
            st.success(result)

        st.markdown("### 🤖 AI Confidence Score")

        c1, c2 = st.columns([3,1])
        with c1:
            st.progress(score/100)
        with c2:
            st.metric("Confidence", f"{score}%")

        pdf = create_pdf(text, result)

        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f)

# ---------------- AUTH ----------------
def login():
    st.subheader("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(u,p):
            st.session_state.logged_in = True
            st.session_state.username = u
            save_login(u)
            st.rerun()
        else:
            st.error("Invalid")

def signup():
    st.subheader("Sign Up")
    u = st.text_input("Username", key="su")
    p = st.text_input("Password", type="password", key="sp")

    if st.button("Create"):
        if add_user(u,p):
            st.success("Created")
        else:
            st.error("Exists")

def reset_password():
    st.subheader("Reset Password")
    u = st.text_input("Username", key="ru")
    p = st.text_input("New Password", type="password", key="rp")

    if st.button("Reset"):
        if update_password(u,p):
            st.success("Updated")
        else:
            st.error("User not found")

# ---------------- MAIN ----------------
def main():
    init_db()

    if st.session_state.logged_in:

        menu = st.sidebar.selectbox(
            "Navigation",
            ["Simplifier","History","Dashboard"]
        )

        if st.sidebar.button("Logout"):
            clear_login()
            st.session_state.logged_in = False
            st.rerun()

        if menu == "Simplifier":
            simplifier()
        elif menu == "History":
            history_page()
        else:
            dashboard()

    else:
        tabs = st.tabs(["Login","Sign Up","Reset Password"])

        with tabs[0]: login()
        with tabs[1]: signup()
        with tabs[2]: reset_password()

if __name__ == "__main__":
    main()