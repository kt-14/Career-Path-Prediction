# Career Path Prediction and Guidance System using Deep Learning

This project aims to guide students in selecting the most suitable career path by analyzing their academic, technical, and behavioral attributes using machine learning and deep learning models deployes via a user-friendly web interface.

---

## Project Overview

- Predicts suitable career roles for students based on their profile
- Built using multiple machine learning algorithms including Random Forest, Gradient Boosting, deep learning (Keras MLP)
- Deployed via a Streamlit web app for real-time predictions
- Recommends relevant courses based on the predicted career

---

## Problem Statement

> The Career Path Prediction System evaluates student records including skills, interests, academic performance, and extracurricular activities using machine learning concepts to provide professional career recommendations. The system guides students on which academic track to pursue and provides actionable plans toward achieving their occupational goals.

---

## Project Structure

```
career_path_predictor/
├── eda.py                    # Exploratory Data Analysis
├── train_model.py            # Train ML/DL models
├── app.py                    # Streamlit app interface
├── career_best_model.pkl        # Saved deep learning model
├── label_encoders.pkl     # Encoders for categorical features
├── scaler.pkl             # Scaler for input normalization
├── PS2_Dataset.csv           # Dataset used for training
├── eda_plots/               # Generated EDA visualizations
├── requirements.txt         # Python dependencies
├── README.md                 # Project documentation
```

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas & NumPy**
- **Streamlit** 

## Supporting Libraries

- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive charts
- **Imbalanced-learn** - SMOTE for class balancing
- **Pickle** - Model serialization

---

## Dataset Description

### Dataset Characteristics:

- Total Records: 6,901 student profiles
- Features: 20 total (19 input features + 1 target variable)
- Data Quality: No missing values
- Target Classes: 12 distinct career paths

### Feature Categories:

- Numerical Features (4): Logical quotient rating, hackathons attended, coding skills rating, public speaking points
- Categorical Features (16): Self-learning capability, certifications, workshops, management vs technical preference, interested subjects, career area preferences, and more

### Career Categories:
- Network Security Engineer, Software Engineer, UX Designer, Software Developer, Database Developer, QA/Testing, Web Developer, CRM Developer, Technical Support, Systems Security Administrator, Applications Developer, Mobile Applications Developer

## Model Summary

| Model           | Accuracy |  Status   | 
|-----------------|----------|-----------|
| Gradient Boost  |   78%    | Selected  |
| RandomForest    |   76%    | Baseline  |
| MLP (Keras)     |   39%    | Evaluated |

> Traditional ensemble methods outperformed deep learning for this tabular dataset. Gradient Boosting Classifier selected as final model based on comprehensive evaluation

---

## Features

- Accepts user input through Streamlit UI
- Encodes & normalizes the input
- Predicts one of 12 career paths
- Displays suggested learning paths based on result

---

## Setup Instructions

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/career-path-predictor.git
   cd career-path-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run exploratory data analysis (optional):
   ```bash
   python eda.py
   ```

4. Train models (optional - pre-trained models included):
   ```bash
   python train_model.py
   ```

5. Run Streamlit app:
   ```bash
   python -m streamlit run app.py
   ```
6. Access the application at http://localhost:8501

---

## Future Scope

- Implement advanced ensemble and deep learning techniques
- Add resume parsing using NLP
- Deploy to Streamlit Cloud / Azure Web Apps
- Integrate user account management