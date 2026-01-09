import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

def train_edu_model():
    print("Loading data...")
    df = pd.read_csv('Student_Performance.csv')

    # 1. Feature Selection
    # Dropping ID and the target variables (overall_score and final_grade)
    X = df.drop(['student_id', 'overall_score', 'final_grade'], axis=1)
    y = df['overall_score']

    # 2. Encoding Categorical Data
    # We save the encoders to use them exactly the same way during testing
    encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train Model
    print("Training Random Forest Regressor for maximum accuracy...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 5. Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\nModel Performance:")
    print(f"- Mean Absolute Error: {mae:.2f} points")
    print(f"- Accuracy (R2 Score): {r2*100:.2f}%")

    # 6. Save Model and Encoders
    joblib.dump(model, 'student_model.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    print("\n✓ Model and Encoders saved successfully.")

if __name__ == "__main__":
    train_edu_model()