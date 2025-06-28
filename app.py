import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load DL model + scaler + encoders
model = load_model("career_dl_model.h5")
scaler = joblib.load("scaler_dl.pkl")
label_encoders = joblib.load("label_encoders_dl.pkl")

st.set_page_config(page_title="Career Path Predictor", layout="centered")
st.title("🤖 Deep Learning Career Path Prediction")
st.write("Enter your details to get a predicted career role and course recommendations.")

def get_user_input():
    logical = st.slider("🧠 Logical Quotient Rating", 1, 10, 5)
    hackathons = st.slider("🏆 Hackathons Attended", 0, 20, 0)
    coding = st.slider("💻 Coding Skills Rating", 1, 10, 5)
    speaking = st.slider("🎤 Public Speaking Points", 1, 10, 5)

    self_learning = st.selectbox("Self-learning Capability?", label_encoders['self-learning capability?'].classes_)
    extra_courses = st.selectbox("Extra Courses Taken?", label_encoders['Extra-courses did'].classes_)
    certifications = st.selectbox("Certifications Done?", label_encoders['certifications'].classes_)
    workshops = st.selectbox("Workshops Attended?", label_encoders['workshops'].classes_)
    reading = st.selectbox("Reading & Writing Skills", label_encoders['reading and writing skills'].classes_)
    memory = st.selectbox("Memory Capability Score", label_encoders['memory capability score'].classes_)
    subjects = st.selectbox("Interested Subjects", label_encoders['Interested subjects'].classes_)
    career_area = st.selectbox("Interested Career Area", label_encoders['interested career area'].classes_)
    company_type = st.selectbox("Preferred Company Type", label_encoders['Type of company want to settle in?'].classes_)
    inputs_from_elders = st.selectbox("Taken Inputs from Seniors?", label_encoders['Taken inputs from seniors or elders'].classes_)
    book_type = st.selectbox("Preferred Type of Books", label_encoders['Interested Type of Books'].classes_)
    mgmt_tech = st.selectbox("Management or Technical", label_encoders['Management or Technical'].classes_)
    work_type = st.selectbox("Hard or Smart Worker", label_encoders['hard/smart worker'].classes_)
    teamwork = st.selectbox("Worked in Teams?", label_encoders['worked in teams ever?'].classes_)
    introvert = st.selectbox("Are You an Introvert?", label_encoders['Introvert'].classes_)

    input_data = [
        logical,
        hackathons,
        coding,
        speaking,
        label_encoders['self-learning capability?'].transform([self_learning])[0],
        label_encoders['Extra-courses did'].transform([extra_courses])[0],
        label_encoders['certifications'].transform([certifications])[0],
        label_encoders['workshops'].transform([workshops])[0],
        label_encoders['reading and writing skills'].transform([reading])[0],
        label_encoders['memory capability score'].transform([memory])[0],
        label_encoders['Interested subjects'].transform([subjects])[0],
        label_encoders['interested career area'].transform([career_area])[0],
        label_encoders['Type of company want to settle in?'].transform([company_type])[0],
        label_encoders['Taken inputs from seniors or elders'].transform([inputs_from_elders])[0],
        label_encoders['Interested Type of Books'].transform([book_type])[0],
        label_encoders['Management or Technical'].transform([mgmt_tech])[0],
        label_encoders['hard/smart worker'].transform([work_type])[0],
        label_encoders['worked in teams ever?'].transform([teamwork])[0],
        label_encoders['Introvert'].transform([introvert])[0],
    ]

    columns = [
        'Logical quotient rating',
        'hackathons',
        'coding skills rating',
        'public speaking points',
        'self-learning capability?',
        'Extra-courses did',
        'certifications',
        'workshops',
        'reading and writing skills',
        'memory capability score',
        'Interested subjects',
        'interested career area',
        'Type of company want to settle in?',
        'Taken inputs from seniors or elders',
        'Interested Type of Books',
        'Management or Technical',
        'hard/smart worker',
        'worked in teams ever?',
        'Introvert'
    ]

    return pd.DataFrame([input_data], columns=columns)

# Get input
user_input = get_user_input()
scaled_input = scaler.transform(user_input)

# Predict
if st.button("🔍 Predict Career"):
    prediction = model.predict(scaled_input)
    predicted_class = np.argmax(prediction)
    career = label_encoders['Suggested Job Role'].inverse_transform([predicted_class])[0]

    st.success(f"🎯 Predicted Career Path: **{career}**")

    st.subheader("📚 Recommended Courses:")
    if "Data" in career:
        st.markdown("- Python\n- SQL\n- Data Science\n- Machine Learning")
    elif "Web" in career:
        st.markdown("- HTML, CSS, JS\n- React\n- Node.js\n- MongoDB")
    elif "Network" in career:
        st.markdown("- Networking Basics\n- Protocols\n- CCNA")
    elif "Security" in career:
        st.markdown("- Cybersecurity\n- Ethical Hacking\n- Firewalls")
    elif "System" in career:
        st.markdown("- OS\n- Comp Arch\n- C/C++\n- OS Internals")
    else:
        st.markdown("- Communication\n- Domain Knowledge\n- Certifications")
