import joblib
import pandas as pd
import numpy as np

def get_user_input():
    print("\n--- Enter Student Details for Prediction ---")
    try:
        data = {
            'age': int(input("Age (e.g., 17): ")),
            'gender': input("Gender (male/female/other): ").lower(),
            'school_type': input("School Type (public/private): ").lower(),
            'parent_education': input("Parent Education (high school/graduate/post graduate/etc.): ").lower(),
            'study_hours': float(input("Weekly Study Hours (e.g., 5.5): ")),
            'attendance_percentage': float(input("Attendance % (0-100): ")),
            'internet_access': input("Internet Access (yes/no): ").lower(),
            'travel_time': input("Travel Time (<15 min/15-30 min/30-60 min/>60 min): ").lower(),
            'extra_activities': input("Extra Activities (yes/no): ").lower(),
            'study_method': input("Study Method (notes/textbook/group study/coaching/mixed): ").lower(),
            'math_score': float(input("Current Math Score: ")),
            'science_score': float(input("Current Science Score: ")),
            'english_score': float(input("Current English Score: "))
        }
        return data
    except ValueError:
        print("\n❌ Invalid input! Please enter numbers for age, hours, and scores.")
        return None

def test_prediction():
    # 1. Load the saved model and encoders
    try:
        model = joblib.load('student_model.pkl')
        encoders = joblib.load('encoders.pkl')
    except FileNotFoundError:
        print("Error: student_model.pkl or encoders.pkl not found. Run train_model.py first.")
        return

    # 2. Get data from user
    user_data = get_user_input()
    if not user_data:
        return

    # 3. Process the data
    input_df = pd.DataFrame([user_data])

    # Apply the same LabelEncoding used during training
    for col, le in encoders.items():
        try:
            # We strip whitespace to handle accidental spaces in user input
            val = str(input_df[col].iloc[0]).strip()
            input_df[col] = le.transform([val])
        except ValueError:
            # If the user enters a category the model hasn't seen
            print(f"⚠️ Warning: '{val}' is an unrecognized category for {col}. Using default.")
            input_df[col] = 0 

    # 4. Predict
    prediction = model.predict(input_df)[0]

    # 5. Output Result
    print("\n" + "="*40)
    print("      PREDICTION RESULT")
    print("="*40)
    print(f"Student: {user_data['age']}yo, {user_data['study_hours']}hrs study/week")
    print(f"Predicted Overall Score: {prediction:.2f}%")
    
    # Optional: Logic to show a hypothetical grade based on score
    grade = 'F'
    if prediction >= 90: grade = 'A'
    elif prediction >= 80: grade = 'B'
    elif prediction >= 70: grade = 'C'
    elif prediction >= 60: grade = 'D'
    elif prediction >= 50: grade = 'E'
    
    print(f"Estimated Grade: {grade}")
    print("="*40)

if __name__ == "__main__":
    test_prediction()