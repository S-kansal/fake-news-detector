import streamlit as st
import pickle
import numpy as np

# ---------------- Load Model ----------------
model = pickle.load(open("../models/model.pkl", "rb"))
vectorizer = pickle.load(open("../models/vectorizer.pkl", "rb"))

# ------------------- Custom CSS -------------------
st.markdown("""
<style>

body {
    background-color: #0d1117;
    font-family: 'Inter', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 10px;
    letter-spacing: 1px;
}

.sub-title {
    text-align: center;
    font-size: 17px;
    color: #9da5b4;
    margin-top: -10px;
}

/* Card styling */
.card {
    background: #161b22;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #2e343d;
    box-shadow: 0 0 6px rgba(0,0,0,0.5);
    color: #c9d1d9;
    font-size: 16px;
}

.section-header {
    color: white;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 12px;
}

textarea {
    background: #0d1117 !important;
    color: white !important;
}

.verify-btn {
    background-color: #238636 !important;
    border: none;
    width: 100%;
    padding: 13px;
    border-radius: 8px;
    color: white !important;
    font-size: 18px !important;
    font-weight: 600;
    margin-top: 5px;
    cursor: pointer;
}

/* Result boxes */
.result-true {
    background: #0e4429;
    border-left: 6px solid #2ea043;
    padding: 18px;
    margin-top: 20px;
    color: #c9d1d9;
    border-radius: 8px;
    font-size: 18px;
}

.result-fake {
    background: #490202;
    border-left: 6px solid #b62324;
    padding: 18px;
    margin-top: 20px;
    color: #c9d1d9;
    border-radius: 8px;
    font-size: 18px;
}

/* Footer */
.footer {
    text-align:center;
    margin-top:40px;
    color:#8b949e;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ Title Section ------------------
st.markdown("<div class='main-title'>Fake News Detection System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>A machine-learning powered tool that identifies factual vs misleading information.</div>", unsafe_allow_html=True)

st.write("")
st.write("")

# ------------------ Example Cards ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='section-header'>Examples of TRUE News</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='card'>
        • WHO released verified influenza guidelines.<br><br>
        • Finance Ministry updated digital payment rules.<br><br>
        • Heavy rainfall caused waterlogging in Mumbai.<br><br>
        • A new metro line was approved in Pune.<br>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section-header'>Examples of FAKE News</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='card'>
        • Amit Shah is the President of India<br><br>
        • Aliens will land in Delhi next month.<br><br>
        • Hot water cures cancer in 2 weeks.<br><br>
        • Government will replace teachers with robots.<br>
        </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# ------------------ Input Section ------------------
st.markdown("<div class='section-header'>Try the Detector</div>", unsafe_allow_html=True)

text = st.text_area("Paste news text here:", height=160)

if st.button("VERIFY NEWS", key="verify", help="Analyze the article"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        conf = model.predict_proba(vec)[0].max() * 100

        if pred == 1:
            st.markdown(f"<div class='result-true'>✔ TRUE NEWS<br><small>Confidence: {conf:.2f}%</small></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-fake'>✖ FAKE NEWS<br><small>Confidence: {conf:.2f}%</small></div>", unsafe_allow_html=True)

# ------------------ Footer ------------------
st.markdown("<div class='footer'>Developed by <b>JSSR</b></div>", unsafe_allow_html=True)
