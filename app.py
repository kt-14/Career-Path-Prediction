import streamlit as st
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Career Path Predictor", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ECDC4;
        margin: 0.5rem 0;
    }
    .action-plan {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #000000;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .insight-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model_info = joblib.load("model_info.pkl")
        
        if model_info['model_type'] == 'rf':
            model = joblib.load("career_best_model.pkl")
        elif model_info['model_type'] == 'gb':
            model = joblib.load("career_best_model.pkl")
        else:
            model = load_model("career_dl_model.keras")
        
        scaler = joblib.load("scaler.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        
        return model, scaler, label_encoders, model_info
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

page = "Career Prediction"
model, scaler, label_encoders, model_info = load_models()

if model is None:
    st.error("Could not load the trained model. Please run the training script first.")
    st.stop()

# Header
st.markdown('<h1 class="main-header">AI Career Path Predictor</h1>', unsafe_allow_html=True)

if page == "Career Prediction":
    st.markdown(f"**Model: {model_info['model_type'].upper()}** | **Accuracy: {model_info['accuracy']:.1%}**")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Enter Your Profile")
        
        # Create form
        with st.form("career_prediction_form"):
            # Numerical inputs
            st.markdown("### Skills Assessment")
            col_a, col_b = st.columns(2)
            
            with col_a:
                logical = st.slider("Logical Quotient Rating", 1, 10, 5)
                coding = st.slider("Coding Skills Rating", 1, 10, 5)
            
            with col_b:
                speaking = st.slider("Public Speaking Points", 1, 10, 5)
                hackathons = st.slider("Hackathons Attended", 0, 20, 0)
            
            st.markdown("### 🎓 Learning & Development")
            
            # Categorical inputs
            col_c, col_d = st.columns(2)
            
            with col_c:
                self_learning = st.selectbox(
                    "Self-learning Capability", 
                    options=label_encoders['self-learning capability?'].classes_
                )
                
                extra_courses = st.selectbox(
                    "Extra Courses Taken", 
                    options=label_encoders['Extra-courses did'].classes_
                )
                
                certifications = st.selectbox(
                    "Certifications Done", 
                    options=label_encoders['certifications'].classes_
                )
            
            with col_d:
                workshops = st.selectbox(
                    "Workshops Attended", 
                    options=label_encoders['workshops'].classes_
                )
                
                career_area = st.selectbox(
                    "Interested Career Area", 
                    options=label_encoders['interested career area'].classes_
                )
                
                mgmt_tech = st.selectbox(
                    "Management or Technical", 
                    options=label_encoders['Management or Technical'].classes_
                )
            
            # Submit button
            submitted = st.form_submit_button("Predict My Career Path", type="primary", use_container_width=True)
            
            if submitted:
                # Prepare input data
                input_data = [
                    logical, hackathons, coding, speaking,
                    label_encoders['self-learning capability?'].transform([self_learning])[0],
                    label_encoders['Extra-courses did'].transform([extra_courses])[0],
                    label_encoders['certifications'].transform([certifications])[0],
                    label_encoders['workshops'].transform([workshops])[0],
                    label_encoders['interested career area'].transform([career_area])[0],
                    label_encoders['Management or Technical'].transform([mgmt_tech])[0]
                ]
                
                # Create DataFrame
                feature_names = [
                    'Logical quotient rating', 'hackathons', 'coding skills rating', 
                    'public speaking points', 'self-learning capability?', 'Extra-courses did',
                    'certifications', 'workshops', 'interested career area', 'Management or Technical'
                ]
                
                user_df = pd.DataFrame([input_data], columns=feature_names)
                user_scaled = scaler.transform(user_df)
                
                # Make prediction
                with st.spinner("Analyzing your profile..."):
                    if model_info['model_type'] in ['rf', 'gb']:
                        prediction = model.predict(user_scaled)[0]
                        prediction_proba = model.predict_proba(user_scaled)[0]
                    else:
                        prediction_raw = model.predict(user_scaled)
                        prediction = np.argmax(prediction_raw)
                        prediction_proba = prediction_raw[0]
                    
                    # Get career name
                    career = label_encoders['Suggested Job Role'].inverse_transform([prediction])[0]
                    confidence = max(prediction_proba) * 100
                    
                    # Display result
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>Your Predicted Career Path</h2>
                        <h1>{career}</h1>
                        <p>Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Career probabilities chart
                    st.markdown("### Career Match Analysis")
                    career_names = label_encoders['Suggested Job Role'].classes_
                    prob_df = pd.DataFrame({
                        'Career': career_names,
                        'Match %': prediction_proba * 100
                    }).sort_values('Match %', ascending=False)
                    
                    fig = px.bar(prob_df, x='Match %', y='Career', orientation='h',
                               color='Match %', color_continuous_scale='viridis',
                               title="Career Match Probabilities")
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top 3 recommendations with different colors
                    st.markdown("### Top 3 Career Recommendations")
                    colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze
                    icons = ['🥇', '🥈', '🥉']
                    
                    for i, (_, row) in enumerate(prob_df.head(3).iterrows()):
                        st.markdown(f"""
                        <div class="metric-card">
                            {icons[i]} <strong style="color: {colors[i]};">{row['Career']}</strong> - <span style="color: #4ECDC4;">{row['Match %']:.1f}% match</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ACTION PLAN SECTION
                    st.markdown("### Your Career Action Plan")
                    
                    action_plans = {
                        'Software Engineer': {
                            'immediate': ['Master Python programming', 'Learn Git version control', 'Build 3 portfolio projects'],
                            'short_term': ['Get AWS/Cloud certification', 'Learn React or Angular', 'Contribute to open source'],
                            'long_term': ['Specialize in AI/ML or Full-stack', 'Lead a development team', 'Mentor junior developers']
                        },
                        'Designer': {
                            'immediate': ['Master Figma/Adobe Creative Suite', 'Build a portfolio website', 'Study UI/UX principles'],
                            'short_term': ['Get Google UX certification', 'Work on real client projects', 'Learn design systems'],
                            'long_term': ['Lead design team', 'Specialize in Product Design', 'Start design consultancy']
                        },
                        'Security Engineer': {
                            'immediate': ['Get CompTIA Security+ cert', 'Learn network fundamentals', 'Practice ethical hacking'],
                            'short_term': ['Get CISSP certification', 'Gain SOC experience', 'Learn cloud security'],
                            'long_term': ['Become security architect', 'Lead security team', 'Get advanced certifications']
                        },
                        'QA Engineer': {
                            'immediate': ['Learn test automation tools', 'Understand SDLC', 'Practice manual testing'],
                            'short_term': ['Master Selenium/Cypress', 'Learn performance testing', 'Get ISTQB certification'],
                            'long_term': ['Lead QA team', 'Specialize in test architecture', 'Implement DevOps practices']
                        },
                        'Support Engineer': {
                            'immediate': ['Develop troubleshooting skills', 'Learn ITIL framework', 'Improve communication'],
                            'short_term': ['Get relevant certifications', 'Learn system administration', 'Master ticketing systems'],
                            'long_term': ['Become team lead', 'Move to technical consulting', 'Specialize in specific technologies']
                        }
                    }
                    
                    if career in action_plans:
                        plan = action_plans[career]
                        st.markdown(f"""
                        <div class="action-plan">
                            <h3>Action Plan for {career}</h3>
                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">
                                <div>
                                    <h4>Immediate (0-6 months)</h4>
                                    <ul>{"".join([f"<li>{item}</li>" for item in plan['immediate']])}</ul>
                                </div>
                                <div>
                                    <h4>Short-term (6-18 months)</h4>
                                    <ul>{"".join([f"<li>{item}</li>" for item in plan['short_term']])}</ul>
                                </div>
                                <div>
                                    <h4>Long-term (2-5 years)</h4>
                                    <ul>{"".join([f"<li>{item}</li>" for item in plan['long_term']])}</ul>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## Resources & Guidance")
        
        # Course recommendations
        st.markdown("### Recommended Learning Path")
        default_courses = [
            "💻 Programming Fundamentals",
            "🗃️ Database Basics", 
            "🌐 Web Technologies",
            "🔧 Version Control (Git)",
            "💼 Professional Skills"
        ]
        
        for course in default_courses:
            st.markdown(f"• {course}")
        
        # Skills visualization
        st.markdown("### Your Skills Profile")
        if 'logical' in locals():
            skills_data = pd.DataFrame({
                'Skill': ['Logical Thinking', 'Coding', 'Communication'],
                'Rating': [logical, coding, speaking]
            })
            
            fig = px.bar(skills_data, x='Skill', y='Rating', color='Rating',
                        color_continuous_scale='blues', title="Your Skills Assessment")
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Tips
        st.markdown("### Career Tips")
        st.info("""
        **Success Factors:**
        - Continuous learning is key
        - Build a strong portfolio
        - Network with professionals
        - Gain practical experience
        - Stay updated with trends
        """)

