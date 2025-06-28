# Career Path Prediction and Guidance System using Deep Learning

This project aims to guide students in selecting the most suitable career path by analyzing their academic, technical, and behavioral attributes using machine learning and deep learning models.

---

## Project Overview

- Predicts suitable career roles for students based on their profile
- Built using deep learning (Keras MLP) and deployed via a Streamlit web app
- Recommends relevant courses based on the predicted career

---

## Problem Statement

> The tool, Career Path Prediction and Guidance System, evaluates student data (skills, interests, academic performance) using deep learning to suggest the most suitable professional tracks. It also provides a recommendation of relevant courses for future growth.

---

## Project Structure

```
career_path_predictor/
├── train_model.py            # Train ML/DL models
├── app.py                    # Streamlit app interface
├── career_dl_model.h5        # Saved deep learning model
├── label_encoders_dl.pkl     # Encoders for categorical features
├── scaler_dl.pkl             # Scaler for input normalization
├── PS2_Dataset.csv           # Dataset used for training
├── README.md                 # Project documentation
```

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Pandas & Scikit-learn**
- **Streamlit** (for frontend)
- **Joblib** (for saving encoders & scalers)

---

## Model Summary

| Model Used      | Accuracy  |
|-----------------|-----------|
| RandomForest    | ~8.5%     |
| MLP (Keras)     | ~9.0%     |

> Accuracy is low due to poor feature-label correlation in the dataset. However, the focus of the project was on building a complete ML pipeline and real-time deployment.

---

## Features

- Accepts user input through Streamlit UI
- Encodes & normalizes the input
- Predicts one of 12 career paths
- Displays suggested learning paths based on result

---

## Sample Recommended Careers

| Predicted Career             | Recommended Courses                           |
|-----------------------------|-----------------------------------------------|
| Data Analyst / Scientist    | Python, SQL, Machine Learning                 |
| Web Developer               | HTML, CSS, JavaScript, React, Node.js         |
| System Engineer             | OS, C/C++, DBMS, Comp Architecture            |
| Network Administrator       | Networking, CCNA, Protocols                   |

---

## How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/career-path-predictor.git
   cd career-path-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Future Scope

- Collect better-labeled data from real students
- Add resume parsing using NLP
- Deploy to Streamlit Cloud / Azure Web Apps
- Integrate user account management
