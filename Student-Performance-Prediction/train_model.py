"""
===================================================================
STUDENT PERFORMANCE PREDICTION SYSTEM - MODEL TRAINING SCRIPT
===================================================================
This script loads student data from dataset.csv, preprocesses it,
trains Decision Tree Classifier and Logistic Regression models,
evaluates both models using multiple classification metrics,
selects the best performing model, and saves it to model.pkl.
===================================================================
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate():
    print("=" * 60)
    print("  STUDENT PERFORMANCE PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # ---------------------------------------------------------------
    # STEP 1: Load Dataset
    # ---------------------------------------------------------------
    csv_file = 'dataset.csv'
    print(f"\n[STEP 1] Loading dataset from '{csv_file}'...")
    try:
        df = pd.read_csv(csv_file)
        print(f"-> Dataset loaded successfully with {len(df)} records.")
    except Exception as e:
        print(f"-> Error loading dataset: {e}")
        return

    # Display dataset sample
    print("\nDataset Sample (First 5 records):")
    print(df.head())

    # ---------------------------------------------------------------
    # STEP 2: Separate Features and Target Variable
    # ---------------------------------------------------------------
    print("\n[STEP 2] Separating features (X) and target (y)...")
    
    # Feature columns: Academic factors influencing student performance
    feature_cols = ['attendance', 'internal_marks', 'study_hours', 'previous_performance', 'assignment_score']
    X = df[feature_cols]
    
    # Target column: Pass or Fail
    # Map 'Pass' -> 1 and 'Fail' -> 0 for machine learning algorithms
    y = df['result'].map({'Pass': 1, 'Fail': 0})

    print(f"-> Features: {feature_cols}")
    print(f"-> Target distribution:\n{df['result'].value_counts()}")

    # ---------------------------------------------------------------
    # STEP 3: Train-Test Split
    # ---------------------------------------------------------------
    print("\n[STEP 3] Splitting data into Training (80%) and Testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"-> Training samples: {len(X_train)}")
    print(f"-> Testing samples : {len(X_test)}")

    # ---------------------------------------------------------------
    # STEP 4: Train Machine Learning Models
    # ---------------------------------------------------------------
    print("\n[STEP 4] Training Machine Learning Models...")
    
    # 1. Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_model.fit(X_train, y_train)
    
    # 2. Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)

    print("-> Decision Tree Classifier trained.")
    print("-> Logistic Regression Model trained.")

    # ---------------------------------------------------------------
    # STEP 5: Evaluate Models on Test Set
    # ---------------------------------------------------------------
    print("\n[STEP 5] Evaluating Models on Test Set...")

    # Predictions
    dt_preds = dt_model.predict(X_test)
    lr_preds = lr_model.predict(X_test)

    # Decision Tree Metrics
    dt_acc = accuracy_score(y_test, dt_preds)
    dt_prec = precision_score(y_test, dt_preds)
    dt_rec = recall_score(y_test, dt_preds)
    dt_f1 = f1_score(y_test, dt_preds)

    # Logistic Regression Metrics
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds)
    lr_rec = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)

    # Print Detailed Evaluation Results
    print("\n" + "=" * 60)
    print("                 MODEL EVALUATION RESULTS")
    print("=" * 60)

    print("\n1. Decision Tree Classifier:")
    print(f"   - Decision Tree Accuracy  : {dt_acc * 100:.2f}%")
    print(f"   - Decision Tree Precision : {dt_prec * 100:.2f}%")
    print(f"   - Decision Tree Recall    : {dt_rec * 100:.2f}%")
    print(f"   - Decision Tree F1 Score  : {dt_f1 * 100:.2f}%")

    print("\n2. Logistic Regression:")
    print(f"   - Logistic Regression Accuracy  : {lr_acc * 100:.2f}%")
    print(f"   - Logistic Regression Precision : {lr_prec * 100:.2f}%")
    print(f"   - Logistic Regression Recall    : {lr_rec * 100:.2f}%")
    print(f"   - Logistic Regression F1 Score  : {lr_f1 * 100:.2f}%")

    # ---------------------------------------------------------------
    # STEP 6: Select Best Model and Save to File
    # ---------------------------------------------------------------
    print("\n[STEP 6] Selecting the best model based on F1 Score and Accuracy...")
    
    if lr_f1 >= dt_f1:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_acc = lr_acc
        best_f1 = lr_f1
    else:
        best_model = dt_model
        best_name = "Decision Tree Classifier"
        best_acc = dt_acc
        best_f1 = dt_f1

    print(f"\n==================================================")
    print(f" Best Model Selected: {best_name}")
    print(f" Best Accuracy      : {best_acc * 100:.2f}%")
    print(f" Best F1 Score      : {best_f1 * 100:.2f}%")
    print(f"==================================================")

    # Save model to model.pkl
    model_filename = 'model.pkl'
    joblib.dump(best_model, model_filename)
    print(f"\n-> Saved best model ('{best_name}') to '{model_filename}' successfully.")
    print("==================================================\n")

if __name__ == '__main__':
    train_and_evaluate()
