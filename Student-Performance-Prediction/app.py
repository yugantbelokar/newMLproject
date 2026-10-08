"""
===================================================================
STUDENT PERFORMANCE PREDICTION SYSTEM - FLASK WEB APP
===================================================================
This Flask web application provides a web interface for predicting
whether a student is likely to PASS or FAIL based on 5 key features:
1. Attendance Percentage (0-100)
2. Internal Marks (0-100)
3. Study Hours Per Day (0-24)
4. Previous Performance Percentage (0-100)
5. Assignment Score (0-100)
===================================================================
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Path to the trained Machine Learning model
MODEL_PATH = os.path.join(PROJECT_DIR, 'model.pkl')

def load_ml_model():
    """Load the trained machine learning model from model.pkl."""
    if os.path.exists(MODEL_PATH):
        try:
            loaded_model = joblib.load(MODEL_PATH)
            return loaded_model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

model = load_ml_model()

@app.route('/', methods=['GET'])
def index():
    """Render home page with the student performance form."""
    return render_template('index.html', error=None, form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission, validate inputs, run ML prediction, and display result."""
    global model
    if model is None:
        model = load_ml_model()

    if model is None:
        return render_template(
            'index.html',
            error="Trained model file ('model.pkl') not found! Please run 'python train_model.py' first.",
            form_data=request.form
        )

    # Retrieve form input values
    attendance_raw = request.form.get('attendance', '').strip()
    internal_marks_raw = request.form.get('internal_marks', '').strip()
    study_hours_raw = request.form.get('study_hours', '').strip()
    previous_perf_raw = request.form.get('previous_performance', '').strip()
    assignment_raw = request.form.get('assignment_score', '').strip()

    form_data = {
        'attendance': attendance_raw,
        'internal_marks': internal_marks_raw,
        'study_hours': study_hours_raw,
        'previous_performance': previous_perf_raw,
        'assignment_score': assignment_raw
    }

    # 1. Validation: Ensure no empty inputs
    if not all([attendance_raw, internal_marks_raw, study_hours_raw, previous_perf_raw, assignment_raw]):
        return render_template('index.html', error="All fields are required. Please enter values for all features.", form_data=form_data)

    # 2. Validation: Ensure values are numerical and within required ranges
    try:
        attendance = float(attendance_raw)
        internal_marks = float(internal_marks_raw)
        study_hours = float(study_hours_raw)
        previous_performance = float(previous_perf_raw)
        assignment_score = float(assignment_raw)
    except ValueError:
        return render_template('index.html', error="Invalid input. Please enter valid numeric values.", form_data=form_data)

    # Range Validations
    if not (0 <= attendance <= 100):
        return render_template('index.html', error="Attendance Percentage must be between 0 and 100.", form_data=form_data)
    if not (0 <= internal_marks <= 100):
        return render_template('index.html', error="Internal Marks must be between 0 and 100.", form_data=form_data)
    if not (0 <= study_hours <= 24):
        return render_template('index.html', error="Study Hours per Day must be between 0 and 24.", form_data=form_data)
    if not (0 <= previous_performance <= 100):
        return render_template('index.html', error="Previous Performance Percentage must be between 0 and 100.", form_data=form_data)
    if not (0 <= assignment_score <= 100):
        return render_template('index.html', error="Assignment Score must be between 0 and 100.", form_data=form_data)

    # 3. Format input for Machine Learning model prediction as DataFrame with feature names
    feature_cols = ['attendance', 'internal_marks', 'study_hours', 'previous_performance', 'assignment_score']
    input_data = pd.DataFrame([[attendance, internal_marks, study_hours, previous_performance, assignment_score]], columns=feature_cols)

    # 4. Make prediction using trained model
    prediction = model.predict(input_data)[0]

    # Calculate Pass probability if model supports predict_proba
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        # Class index 1 corresponds to 'Pass', index 0 to 'Fail'
        pass_prob = round(probabilities[1] * 100, 1)
    else:
        pass_prob = 100.0 if prediction == 1 else 0.0

    result_text = "PASS" if prediction == 1 else "FAIL"

    # 5. Render result page with prediction and entered student details
    return render_template(
        'result.html',
        result=result_text,
        pass_probability=pass_prob,
        inputs={
            'attendance': attendance,
            'internal_marks': internal_marks,
            'study_hours': study_hours,
            'previous_performance': previous_performance,
            'assignment_score': assignment_score
        }
    )

if __name__ == '__main__':
    print("Starting Flask Web Application...")
    print("Access the web app at: http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)
