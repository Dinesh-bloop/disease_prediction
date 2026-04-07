import streamlit as st
import pickle
import sqlite3
import pandas as pd
from streamlit_option_menu import option_menu
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Healthcare Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# ---------------- CSS ----------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: #00c6ff;}
.stButton>button {
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
    font-weight: bold;
}
.stMetric {
    background-color: #1c1c1c;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("patients.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS patients(
name TEXT,
age INTEGER,
disease TEXT,
result TEXT,
risk REAL
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS users(
username TEXT,
password TEXT
)
""")

conn.commit()

# ---------------- PDF FUNCTION ----------------
def generate_pdf(name, age, disease, result, risk):
    file_path = f"{name}_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Patient Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"Age: {age}", styles["Normal"]))
    content.append(Paragraph(f"Disease: {disease}", styles["Normal"]))
    content.append(Paragraph(f"Result: {result}", styles["Normal"]))
    content.append(Paragraph(f"Risk: {risk:.2f}%", styles["Normal"]))

    doc.build(content)
    return file_path

# ---------------- LOGIN SYSTEM ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 AI Healthcare Login")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
            data = c.fetchone()

            if data:
                st.session_state.logged_in = True
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid Credentials")

    with tab2:
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")

        if st.button("Register"):
            c.execute("INSERT INTO users VALUES (?,?)", (new_user, new_pass))
            conn.commit()
            st.success("User Registered")

    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>🧑‍⚕️ AI Healthcare Assistant</h1>
<p style='text-align:center; color:gray;'>Smart Disease Prediction & Patient Monitoring</p>
<hr>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
diabetes_model = pickle.load(open("saved_models/diabetes_model.sav", "rb"))
heart_model = pickle.load(open("saved_models/heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("saved_models/parkinsons_model.sav", "rb"))

# ---------------- SIDEBAR ----------------
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Diabetes", "Heart", "Parkinsons", "Dashboard"],
        icons=["activity", "heart", "person", "bar-chart"],
        default_index=0
    )

    st.sidebar.info("ML-based disease prediction system")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ---------------- DIABETES ----------------
if selected == "Diabetes":

    st.title("🩺 Diabetes Prediction")

    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 120)

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20)
        glucose = st.number_input("Glucose", 0, 200)
        bp = st.number_input("Blood Pressure", 0, 150)

    with col2:
        skin = st.number_input("Skin Thickness", 0, 100)
        insulin = st.number_input("Insulin", 0, 900)
        bmi = st.number_input("BMI", 0.0, 60.0)
        dpf = st.number_input("Diabetes Pedigree", 0.0, 2.5)
        age_input = st.number_input("Medical Age", 1, 120)

    if st.button("Predict Diabetes"):

        with st.spinner("Analyzing..."):
            data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age_input]
            prediction = diabetes_model.predict([data])
            prob = diabetes_model.predict_proba([data])

        risk = prob[0][1] * 100
        result = "Diabetic" if prediction[0] == 1 else "Healthy"

        st.progress(int(risk))
        st.metric("Risk Probability", f"{risk:.2f}%")

        if risk > 70:
            st.error("⚠️ High Risk — Consult doctor")
        elif risk > 40:
            st.warning("⚠️ Moderate Risk")
        else:
            st.success("✅ Low Risk")

        c.execute("INSERT INTO patients VALUES (?,?,?,?,?)",
                  (name, age, "Diabetes", result, risk))
        conn.commit()

        pdf = generate_pdf(name, age, "Diabetes", result, risk)
        with open(pdf, "rb") as f:
            st.download_button("Download Report", f, file_name=pdf)

# ---------------- HEART ----------------
if selected == "Heart":

    st.title("❤️ Heart Disease Prediction")

    name = st.text_input("Patient Name")
    age_main = st.number_input("Age", 1, 120)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Medical Age", 1, 120)
        sex = st.selectbox("Sex", [0,1])
        cp = st.selectbox("Chest Pain", [0,1,2,3])
        trestbps = st.number_input("BP", 80, 200)

    with col2:
        chol = st.number_input("Cholesterol", 100, 400)
        fbs = st.selectbox("FBS", [0,1])
        restecg = st.selectbox("ECG", [0,1,2])
        thalach = st.number_input("Heart Rate", 60, 220)
        exang = st.selectbox("Angina", [0,1])
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0)
        slope = st.selectbox("Slope", [0,1,2])
        ca = st.selectbox("CA", [0,1,2,3])
        thal = st.selectbox("Thal", [0,1,2,3])

    if st.button("Predict Heart"):

        with st.spinner("Analyzing..."):
            data = [age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal]

            prediction = heart_model.predict([data])
            prob = heart_model.predict_proba([data])

        risk = prob[0][1] * 100
        result = "Heart Disease" if prediction[0] == 1 else "Healthy"

        st.progress(int(risk))
        st.metric("Risk Probability", f"{risk:.2f}%")

        c.execute("INSERT INTO patients VALUES (?,?,?,?,?)",
                  (name, age_main, "Heart", result, risk))
        conn.commit()

        pdf = generate_pdf(name, age_main, "Heart", result, risk)
        with open(pdf, "rb") as f:
            st.download_button("Download Report", f, file_name=pdf)

# ---------------- PARKINSONS ----------------
if selected == "Parkinsons":

    st.title("🧠 Parkinsons Prediction")

    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 120)

    col1, col2 = st.columns(2)

    with col1:
        fo = st.number_input("Fo", 100.0, 300.0)
        fhi = st.number_input("Fhi", 100.0, 300.0)
        flo = st.number_input("Flo", 50.0, 250.0)
        jitter = st.number_input("Jitter", 0.0, 1.0)
        shimmer = st.number_input("Shimmer", 0.0, 1.0)

    with col2:
        nhr = st.number_input("NHR", 0.0, 1.0)
        hnr = st.number_input("HNR", 0.0, 50.0)
        rpde = st.number_input("RPDE", 0.0, 1.0)
        dfa = st.number_input("DFA", 0.0, 1.0)
        ppe = st.number_input("PPE", 0.0, 1.0)

    if st.button("Predict Parkinsons"):

        with st.spinner("Analyzing..."):
            data = [fo, fhi, flo, jitter, 0,0,0,0,
                    shimmer,0,0,0,0,0,nhr,hnr,
                    rpde,dfa,0,0,0,ppe]

            prediction = parkinsons_model.predict([data])
            prob = parkinsons_model.predict_proba([data])

        risk = prob[0][1] * 100
        result = "Parkinsons" if prediction[0] == 1 else "Healthy"

        st.progress(int(risk))
        st.metric("Risk Probability", f"{risk:.2f}%")

        c.execute("INSERT INTO patients VALUES (?,?,?,?,?)",
                  (name, age, "Parkinsons", result, risk))
        conn.commit()

        pdf = generate_pdf(name, age, "Parkinsons", result, risk)
        with open(pdf, "rb") as f:
            st.download_button("Download Report", f, file_name=pdf)

# ---------------- DASHBOARD ----------------
if selected == "Dashboard":

    st.title("📊 Dashboard")

    df = pd.read_sql_query("SELECT * FROM patients", conn)

    if df.empty:
        st.warning("No data available")
    else:
        st.dataframe(df)

        st.bar_chart(df["disease"].value_counts())
        st.area_chart(df["risk"])

        search = st.text_input("Search Patient")

        if search:
            st.dataframe(df[df["name"].str.contains(search, case=False)])

        st.download_button("Download CSV", df.to_csv(index=False), "report.csv")
